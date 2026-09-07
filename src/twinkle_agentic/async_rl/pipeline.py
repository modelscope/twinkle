# Copyright (c) ModelScope Contributors. All rights reserved.
"""Driver for the independent async-RL Ray workers."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from functools import partial
from pydoc import locate
from typing import Any, Sequence

from twinkle.metric import MetricRecord, MetricsReporter, create_metrics_reporter
from .context_manager import LoraContextManager
from .data_plane import TQDataPlane
from .scheduler import ContextSchedulePolicy, SchedulerConfig
from .types import LoraContext, PartitionAdmission
from .utils import (TrainBatchConfig, build_native_fsdp_model_kwargs, configure_lora_lr_scheduler,
                    resolve_context_learning_rate, resolve_context_lora_target_modules, resolve_context_loss_config,
                    resolve_model_attention_implementation, resolve_sequence_parallel_size, sampler_data_parallel_size,
                    validate_context_batch_config)
from .workers import AdvantageWorker, RolloutWorker, TrainerWorker


@dataclass(frozen=True)
class AsyncMultiLoraGRPOConfig:
    metrics_drain_interval_s: float = 1.0


class AsyncMultiLoraGRPOPipeline:
    """Owns the production async-RL runtime and drives its worker services.

    ``from_config`` is the real training construction path.  The explicit
    constructor remains available to fake-TQ tests, where injecting a fake
    sampler/model is the point of the test.
    """

    def __init__(self,
                 *,
                 context_manager: LoraContextManager,
                 rollout_worker: RolloutWorker,
                 advantage_worker: AdvantageWorker,
                 trainer_worker: TrainerWorker,
                 metrics: MetricsReporter | None = None,
                 config: AsyncMultiLoraGRPOConfig = AsyncMultiLoraGRPOConfig(),
                 sampler: Any | None = None,
                 model: Any | None = None,
                 contexts: Sequence[LoraContext] = ()):
        self.context_manager = context_manager
        self.rollout_worker = rollout_worker
        self.advantage_worker = advantage_worker
        self.trainer_worker = trainer_worker
        self.sampler = sampler
        self.model = model
        self.contexts = tuple(contexts)
        self.metrics = metrics
        self.config = config

    @classmethod
    def from_config(
        cls,
        raw_config: dict[str, Any],
        *,
        persistent: bool = False,
    ) -> AsyncMultiLoraGRPOPipeline:
        """Build the complete Ray/TQ runtime from the async-RL YAML mapping."""
        from omegaconf import OmegaConf

        raw_config = OmegaConf.to_container(OmegaConf.create(raw_config), resolve=True)
        if not isinstance(raw_config, dict):
            raise TypeError('async-RL config must resolve to a mapping')

        import ray
        import transfer_queue as tq
        from peft import LoraConfig

        import twinkle
        from twinkle import DeviceGroup, DeviceMesh
        from twinkle.data_format import SamplingParams
        from twinkle.model import MultiLoraTransformersModel
        from twinkle.processor import InputProcessor
        from .native_tq import ContextGRPOGroupNSampler

        runtime = raw_config['runtime']
        model_config = raw_config['model']
        lora_config_data = raw_config['lora']
        loss_config_data = raw_config.get('loss')
        template_config = raw_config.get('template', {})
        template_cls = template_config.get('cls', 'Qwen3_5Template')
        enable_thinking = bool(template_config.get('enable_thinking', False))
        rollout_output_config = dict(raw_config.get('rollout_output') or {})
        sampler_gpus = int(runtime['sampler_gpus'])
        sampler_tp = int(runtime['sampler_tp'])
        sampler_dp = sampler_data_parallel_size(sampler_gpus, sampler_tp)
        model_dp = int(runtime['model_gpus'])
        sequence_parallel_size = resolve_sequence_parallel_size(
            model_dp,
            int(model_config['sequence_parallel_size']),
        )
        padding_free = bool(model_config['padding_free'])
        attn_implementation = resolve_model_attention_implementation(
            model_config,
            padding_free=padding_free,
            sequence_parallel_size=sequence_parallel_size,
        )
        model_max_length = int(model_config['max_length'])
        sampler_config = raw_config['sampler']
        total_gpus = model_dp + sampler_gpus
        device_groups = [
            DeviceGroup('model', list(range(int(runtime['model_gpus']))), device_type='GPU'),
            DeviceGroup(
                'sampler',
                list(range(int(runtime['model_gpus']), total_gpus)),
                device_type='GPU',
                gpus_per_worker=sampler_tp,
            ),
        ]
        twinkle.initialize(mode='ray', nproc_per_node=total_gpus, groups=device_groups, lazy_collect=False)
        tq.init(
            OmegaConf.create(
                {
                    'controller': {
                        'sampler': ContextGRPOGroupNSampler,
                        'polling_mode': bool(raw_config['tq'].get('polling_mode', True)),
                    },
                    'backend': {
                        'SimpleStorage': {
                            'num_data_storage_units': raw_config['tq']['storage_units']
                        }
                    },
                },
                flags={'allow_objects': True}))

        model_mesh = DeviceMesh.from_sizes(
            world_size=model_dp,
            dp_size=model_dp,
            ulysses_size=sequence_parallel_size,
        )
        model_data_parallel_size = model_mesh.data_world_size
        sampler_mesh = DeviceMesh.from_sizes(world_size=sampler_gpus, dp_size=sampler_dp, tp_size=sampler_tp)
        model_kwargs = build_native_fsdp_model_kwargs(model_config)
        if attn_implementation is not None:
            model_kwargs['attn_implementation'] = attn_implementation
        model = MultiLoraTransformersModel(
            model_id=runtime['model_id'],
            device_mesh=model_mesh,
            remote_group='model',
            max_length=model_max_length,
            **model_kwargs,
        )
        contexts: list[LoraContext] = []
        prompt_sources: dict[str, Any] = {}
        rollout_config: dict[str, dict[str, Any]] = {}
        train_batch_configs: dict[str, TrainBatchConfig] = {}
        rewards: dict[str, Any] = {}
        evaluation_config: dict[str, dict[str, Any]] = {}
        evaluation_rewards: dict[str, Any] = {}
        initial_paths: dict[str, str] = {}
        global_evaluation = dict(raw_config.get('evaluation') or {})
        for item in raw_config['lora_contexts']:
            train = item['train']
            context = LoraContext(
                item['tenant_id'],
                item['training_run_id'],
                runtime['model_id'],
                item['adapter_name'],
            )
            contexts.append(context)
            adapter_lora_config = LoraConfig(
                target_modules=resolve_context_lora_target_modules(item, lora_config_data),
                r=lora_config_data['r'],
                lora_alpha=lora_config_data['alpha'],
                lora_dropout=lora_config_data['dropout'],
            )
            model.add_adapter_to_model(
                context.adapter_name,
                adapter_lora_config,
                gradient_accumulation_steps=1,
            )
            model.set_optimizer(
                'AdamW',
                lr=resolve_context_learning_rate(train, lora_config_data),
                adapter_name=context.adapter_name,
            )
            configure_lora_lr_scheduler(model, context.adapter_name, lora_config_data)
            loss_cls, loss_kwargs = resolve_context_loss_config(item, loss_config_data)
            model.set_loss(
                loss_cls,
                adapter_name=context.adapter_name,
                **loss_kwargs,
            )
            model.set_processor(
                InputProcessor,
                adapter_name=context.adapter_name,
                padding_free=padding_free,
            )
            model.set_template(
                template_cls,
                model_id=runtime['model_id'],
                adapter_name=context.adapter_name,
                enable_thinking=enable_thinking,
                max_length=model_max_length,
            )

            rollout = item['rollout']
            rollout_batch_size = int(rollout['batch_size'])
            num_generations = int(rollout['num_generations'])
            train_batch_config = TrainBatchConfig(
                mini_batch_size=int(train['mini_batch_size']),
                micro_batch_size=int(train['micro_batch_size']),
                dynamic_batching=bool(train.get('dynamic_batching', False)),
                max_tokens_per_micro_batch=(int(train['max_tokens_per_micro_batch'])
                                            if train.get('max_tokens_per_micro_batch') is not None else None),
                packing_algorithm=str(train.get('packing_algorithm', 'ffd')),
            )
            validate_context_batch_config(
                context.key,
                rollout_groups=rollout_batch_size,
                num_generations=num_generations,
                train=train_batch_config,
                sampler_dp=sampler_dp,
                model_dp=model_data_parallel_size,
            )
            prompt_sources[context.key] = partial(
                _prompt_batches,
                item['dataset'],
                model_id=runtime['model_id'],
                batch_size=rollout_batch_size,
                template_cls=template_cls,
                enable_thinking=enable_thinking,
            )
            rollout_config[context.key] = {
                'context':
                context,
                'batch_size':
                rollout_batch_size,
                'num_generations':
                num_generations,
                'sampling_params':
                SamplingParams(
                    max_tokens=rollout['max_tokens'],
                    temperature=rollout['temperature'],
                    top_p=rollout['top_p'],
                    repetition_penalty=float(rollout.get('repetition_penalty', 1.0)),
                    logprobs=1,
                    num_samples=1,
                ),
            }
            train_batch_configs[context.key] = train_batch_config
            rewards[context.key] = _reward_for_context(
                item.get('reward'),
                context_key=context.key,
            )
            if bool(global_evaluation.get('enabled', False)):
                eval_dataset = item.get('eval_dataset')
                if eval_dataset is None:
                    raise ValueError(f'eval_dataset is required for periodic evaluation of {context.key}')
                eval_batch_size = int(global_evaluation.get('batch_size', 16))
                eval_interval = int(global_evaluation.get('interval', 1))
                if eval_batch_size <= 0 or eval_interval <= 0:
                    raise ValueError('evaluation.batch_size and evaluation.interval must be positive')
                eval_sampling = dict(global_evaluation.get('sampling_params') or {})
                evaluation_config[context.key] = {
                    'interval':
                    eval_interval,
                    'dataset_name':
                    eval_dataset.get('name', eval_dataset['dataset_id']),
                    'prompt_batches':
                    partial(
                        _prompt_batches,
                        eval_dataset,
                        model_id=runtime['model_id'],
                        batch_size=eval_batch_size,
                        template_cls=template_cls,
                        enable_thinking=enable_thinking,
                        full_batches_only=False,
                    ),
                    'sampling_params':
                    SamplingParams(
                        max_tokens=int(eval_sampling.get('max_tokens', rollout['max_tokens'])),
                        temperature=float(eval_sampling.get('temperature', 0.0)),
                        top_p=float(eval_sampling.get('top_p', 1.0)),
                        repetition_penalty=float(eval_sampling.get('repetition_penalty', 1.0)),
                        logprobs=0,
                        num_samples=1,
                    ),
                }
                evaluation_rewards[context.key] = _reward_for_context(
                    eval_dataset.get('reward'),
                    context_key=f'{context.key} evaluation',
                )
            initial_paths[context.key] = _collect_adapter_path(
                model.save(
                    f'async-{context.adapter_name}-initial',
                    output_dir=runtime['output_dir'],
                    adapter_name=context.adapter_name,
                ),
                operation=f'initial adapter save for {context.key}',
            )

        manager = create_cpu_actor(
            LoraContextManager,
            max_staleness=runtime['max_staleness'],
            max_steps=runtime['max_steps'],
        )
        for context in contexts:
            ray.get(manager.register_context.remote(context, adapter_path=initial_paths[context.key]))

        from .vllm_sampler_tq import VLLMSamplerTQ
        sampler_engine_args = {
            'tensor_parallel_size': sampler_tp,
            'enable_lora': True,
            'max_loras': int(runtime['sampler_max_loras']),
            'max_lora_rank': lora_config_data['r'],
            'max_model_len': int(sampler_config['max_model_len']),
            'gpu_memory_utilization': float(sampler_config['gpu_memory_utilization']),
            'max_num_seqs': int(sampler_config['max_num_seqs']),
            'enforce_eager': bool(sampler_config['enforce_eager']),
            'seed': int(runtime.get('seed', 1)),
        }
        if sampler_config.get('max_num_batched_tokens') is not None:
            sampler_engine_args['max_num_batched_tokens'] = int(sampler_config['max_num_batched_tokens'])
        sampler = VLLMSamplerTQ(
            model_id=runtime['model_id'],
            remote_group='sampler',
            device_mesh=sampler_mesh,
            engine_args=sampler_engine_args,
            reward_registry=rewards,
            context_manager=manager,
            rollout_max_retries=int(runtime.get('rollout_max_retries', 2)),
            rollout_retry_delay_s=float(runtime.get('rollout_retry_delay_s', 0.5)),
            rollout_output_dir=(rollout_output_config.get('output_dir') if bool(
                rollout_output_config.get('enabled', False)) else None),
            rollout_output_include_token_ids=bool(rollout_output_config.get('include_token_ids', False)),
        )
        sampler.set_template(
            template_cls,
            model_id=runtime['model_id'],
            enable_thinking=enable_thinking,
            max_length=model_max_length,
        )

        rollout_worker = create_cpu_actor(
            RolloutWorker,
            context_manager=manager,
            data_plane=TQDataPlane(),
            sampler=sampler,
            prompt_batches=prompt_sources,
            rollout_config=rollout_config,
            scheduler=_scheduler(raw_config['scheduler']['rollout']),
            allow_partial_rollout=runtime['allow_partial_rollout'],
            persistent=persistent,
        )
        advantage_worker = create_cpu_actor(
            AdvantageWorker,
            context_manager=manager,
            data_plane=TQDataPlane(),
            advantage_fn=_compute_advantages,
            scheduler=_scheduler(raw_config['scheduler']['advantage']),
            persistent=persistent,
        )
        trainer_worker = create_cpu_actor(
            TrainerWorker,
            context_manager=manager,
            data_plane=TQDataPlane(),
            train_fn=partial(
                _train_batch,
                model,
                train_batch_configs,
                model_data_parallel_size=model_data_parallel_size,
            ),
            train_with_config_fn=partial(
                _train_batch_with_config,
                model,
                model_data_parallel_size=model_data_parallel_size,
            ),
            train_batch_configs=train_batch_configs,
            save_adapter=partial(_save_adapter, model, runtime['output_dir']),
            mini_batch_sizes={
                key: config.mini_batch_size
                for key, config in train_batch_configs.items()
            },
            scheduler=_scheduler(raw_config['scheduler']['train']),
            keep_adapter_versions=runtime['keep_adapter_versions'],
            initial_adapter_paths=initial_paths,
            remove_adapter=partial(_remove_adapter_snapshot, sampler),
            evaluation_config=evaluation_config,
            evaluate_batch=partial(_evaluate_batch, sampler, evaluation_rewards) if evaluation_config else None,
            evaluate_with_reward_fn=partial(_evaluate_batch_with_reward, sampler),
            evaluation_rewards=evaluation_rewards,
            persistent=persistent,
        )
        raw_metrics_config = raw_config.get('metrics')
        metrics_config = dict(raw_metrics_config or {})
        metrics = create_metrics_reporter(
            raw_metrics_config,
            run_id=str(runtime.get('run_id', 'async_multi_lora_grpo')),
        )
        return cls(
            context_manager=manager,
            rollout_worker=rollout_worker,
            advantage_worker=advantage_worker,
            trainer_worker=trainer_worker,
            sampler=sampler,
            metrics=metrics,
            config=AsyncMultiLoraGRPOConfig(
                metrics_drain_interval_s=float(metrics_config.get('drain_interval_s', 1.0)), ),
            model=model,
            contexts=contexts,
        )

    async def run_async(self) -> dict[str, Any]:
        started = time.perf_counter()
        workers = [self.rollout_worker, self.advantage_worker, self.trainer_worker]
        await asyncio.gather(*(worker.start.remote() for worker in workers))
        try:
            while True:
                await self._drain_metrics()
                states = await asyncio.gather(*(worker.get_service_state.remote() for worker in workers))
                failures = [state['failure'] for state in states if state['failure']]
                if failures:
                    raise RuntimeError(f'async RL worker failed: {failures[0]}')
                if self.sampler is not None:
                    await asyncio.to_thread(self.sampler.check_health)
                running = any(bool(state['running']) for state in states)
                if not running:
                    if await self.context_manager.is_run_finished.remote():
                        break
                    raise RuntimeError('async RL workers stopped before all contexts were drained')
                await asyncio.sleep(self.config.metrics_drain_interval_s)
        except Exception as exc:
            if self.metrics is not None:
                self.metrics.record(
                    MetricRecord(
                        stage='run',
                        status='failed',
                        values={'wall_time_s': time.perf_counter() - started},
                        attributes={'error': f'{type(exc).__name__}: {exc}'},
                    ))
                self.metrics.flush()
            raise
        finally:
            await asyncio.gather(*(worker.stop.remote() for worker in workers), return_exceptions=True)
            await self._drain_metrics()
        result = {
            'trained_partitions': await self.context_manager.get_completed_partitions.remote(),
            'wall_time_s': time.perf_counter() - started,
        }
        if self.metrics is not None:
            self.metrics.record(MetricRecord(stage='run', values=result))
            self.metrics.flush()
            result['metrics_health'] = self.metrics.health()
        return result

    def run(self) -> dict[str, Any]:
        try:
            return asyncio.run(self.run_async())
        finally:
            if self.metrics is not None:
                self.metrics.close()

    async def _drain_metrics(self) -> None:
        workers = [self.rollout_worker, self.advantage_worker, self.trainer_worker]
        for worker in workers:
            records = await worker.drain_metric_records.remote()
            if self.metrics is not None:
                self.metrics.record_many(records)
        if self.sampler is not None:
            records = await asyncio.to_thread(self.sampler.drain_metric_records)
            if self.metrics is not None:
                self.metrics.record_many(records)


def create_cpu_actor(cls: type, *args: Any, **kwargs: Any) -> Any:
    """Deploy a CPU service as one raw Ray actor; local tests use the class directly."""

    import ray
    actor_class = ray.remote(
        num_cpus=1,
        runtime_env={'env_vars': {
            'TWINKLE_MODE': 'ray'
        }},
    )(
        cls)
    return actor_class.remote(*args, **kwargs)


def _scheduler(config: dict[str, Any]) -> SchedulerConfig:
    return SchedulerConfig(ContextSchedulePolicy(config['policy']), config.get('max_consecutive_units'))


def _prompt_batches(
    dataset_config: dict[str, Any],
    *,
    model_id: str,
    batch_size: int,
    template_cls: str,
    enable_thinking: bool,
    full_batches_only: bool = True,
):
    """Create a lazy, full-batch-only prompt source for one context."""
    from twinkle.dataloader import DataLoader
    from twinkle.dataset import Dataset, DatasetMeta
    from twinkle.preprocessor import llm as llm_processors

    def batches():
        data_num = dataset_config.get('data_num')
        dataset = Dataset(
            DatasetMeta(
                dataset_config['dataset_id'],
                subset_name=dataset_config.get('subset_name'),
                split=dataset_config.get('split', 'train'),
                data_slice=range(int(data_num)) if data_num is not None else None,
            ))
        dataset.set_template(
            template_cls,
            model_id=model_id,
            max_length=dataset_config['max_length'],
            enable_thinking=enable_thinking,
        )
        processor_name = dataset_config.get('processor', 'GSM8KProcessor')
        processor_cls = getattr(llm_processors, processor_name)
        if processor_name == 'GSM8KProcessor':
            processor = processor_cls(system=dataset_config['system_prompt'])
        else:
            processor = processor_cls()
        dataset.map(processor)
        dataset.encode(add_generation_prompt=True)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            min_batch_size=batch_size if full_batches_only else 1,
        )
        remaining = data_num
        remaining = None if remaining is None else int(remaining)
        for batch in loader:
            if full_batches_only and (len(batch) != batch_size or (remaining is not None and remaining < batch_size)):
                return
            yield batch
            if remaining is not None:
                remaining -= batch_size

    return batches()


def _evaluate_batch(
    sampler: Any,
    reward_registry: dict[str, Any],
    prompts: Sequence[dict[str, Any]],
    admission: PartitionAdmission,
    adapter_path: str,
    policy_version: int,
    sampling_params: Any,
) -> dict[str, Any]:
    reward_fn = reward_registry[admission.context.key]
    return _evaluate_batch_with_reward(
        sampler,
        prompts,
        admission,
        adapter_path,
        policy_version,
        sampling_params,
        reward_fn,
    )


def _evaluate_batch_with_reward(
    sampler: Any,
    prompts: Sequence[dict[str, Any]],
    admission: PartitionAdmission,
    adapter_path: str,
    policy_version: int,
    sampling_params: Any,
    reward_fn: Any,
) -> dict[str, Any]:
    from .utils import sample_responses_to_rollout_rows

    responses = sampler.evaluate(
        list(prompts),
        sampling_params,
        admission.context.adapter_name,
        adapter_path,
    )
    rows = sample_responses_to_rollout_rows(list(prompts), responses, policy_version=policy_version)
    rewards = list(reward_fn(rows, context=admission.context))
    return {
        'rewards': rewards,
        'completion_lengths': [int(row['completion_length']) for row in rows],
    }


def _reward_for_context(
    reward_config: dict[str, Any] | None = None,
    *,
    context_key: str,
) -> Any:
    from twinkle.reward import Reward

    config = dict(reward_config or {})
    class_path = config.get('class_path', '')
    reward_cls = locate(class_path)
    if not isinstance(reward_cls, type) or not issubclass(reward_cls, Reward):
        raise TypeError(f'reward.class_path {class_path!r} for {context_key} must reference a Reward subclass')
    return reward_cls(**dict(config.get('kwargs') or {}))


def _compute_advantages(data: Any, admission: PartitionAdmission) -> tuple[list[float], list[float]]:
    from twinkle.advantage import GRPOAdvantage
    rewards = [float(value) for value in data['rewards']]
    advantages = GRPOAdvantage()(rewards, num_generations=admission.num_generations, scale='group').tolist()
    return advantages, rewards


def _train_batch(
    model: Any,
    train_batch_configs: dict[str, TrainBatchConfig],
    data: Any,
    admission: PartitionAdmission,
    *,
    model_data_parallel_size: int = 1,
) -> dict[str, Any]:
    config = train_batch_configs[admission.context.key]
    return _train_batch_with_config(
        model,
        data,
        admission,
        config,
        model_data_parallel_size=model_data_parallel_size,
    )


def _train_batch_with_config(
    model: Any,
    data: Any,
    admission: PartitionAdmission,
    config: TrainBatchConfig,
    *,
    model_data_parallel_size: int = 1,
) -> dict[str, Any]:
    from .tq_utils import REQUIRED_MODEL_INPUT_FIELDS

    size = int(data.batch_size[0])
    inputs = [{name: data[name][index] for name in REQUIRED_MODEL_INPUT_FIELDS} for index in range(size)]
    old_logps = list(data['logprobs'])
    advantages = list(data['advantages'])
    if size != config.mini_batch_size:
        raise ValueError(f'train batch for {admission.context.key} has {size} samples; '
                         f'expected mini_batch_size={config.mini_batch_size}')

    if size % model_data_parallel_size:
        raise ValueError(f'train batch size {size} must be divisible by model DP size '
                         f'{model_data_parallel_size}')
    model.forward_backward(
        inputs=inputs,
        old_logps=old_logps,
        advantages=advantages,
        adapter_name=admission.context.adapter_name,
        micro_batch_size=config.micro_batch_size,
        dynamic_batching=config.dynamic_batching,
        max_tokens_per_micro_batch=config.max_tokens_per_micro_batch,
        packing_algorithm=config.packing_algorithm,
        sync_gradients=True,
        loss_scale=1.0,
    )

    model.clip_grad_and_step(adapter_name=admission.context.adapter_name)
    metrics = dict(model.calculate_metric(is_training=True, adapter_name=admission.context.adapter_name))
    metrics['mini_batch_size'] = config.mini_batch_size
    metrics['micro_batch_size_per_rank'] = config.micro_batch_size
    metrics['dynamic_batching'] = config.dynamic_batching
    return metrics


def _save_adapter(model: Any, output_dir: str, admission: PartitionAdmission) -> str:
    return _collect_adapter_path(
        model.save(
            f'async-{admission.context.adapter_name}-v{admission.step + 1}',
            output_dir=output_dir,
            adapter_name=admission.context.adapter_name,
        ),
        operation=f'adapter save for {admission.partition_id}',
    )


def _collect_adapter_path(value: Any, *, operation: str) -> str:
    """Collect a lazy model.save result at the async-RL publication boundary."""
    if callable(value) and getattr(value, '_is_lazy_collect', False):
        value = value()
    return _require_adapter_path(value, operation=operation)


def _require_adapter_path(value: Any, *, operation: str) -> str:
    """Fail at the save boundary instead of publishing an invalid policy."""
    if not isinstance(value, str) or not value:
        raise TypeError(f'{operation} must return a non-empty checkpoint path string, '
                        f'got {type(value).__name__}: {value!r}')
    return value


def _remove_adapter_snapshot(sampler: Any, adapter_path: str) -> None:
    """Unload an unreferenced policy from vLLM before deleting its checkpoint."""
    from .workers import _remove_local_adapter

    sampler.unload_adapter_paths([adapter_path])
    _remove_local_adapter(adapter_path)
