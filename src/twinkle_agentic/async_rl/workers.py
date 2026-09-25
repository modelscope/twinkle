# Copyright (c) ModelScope Contributors. All rights reserved.
"""Long-lived, context-scheduling async-RL workers."""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from twinkle.metric import MetricBuffer, MetricRecord
from .context_manager import ContextStatus, LoraContextManager
from .data_plane import TQDataPlane
from .metrics import advantage_signal_metrics, training_policy_metrics
from .scheduler import ContextScheduler, ScheduleCandidate, SchedulerConfig
from .types import LoraContext, PartitionAdmission


class _Worker:
    """A long-lived Ray service with one privately owned background loop."""

    def __init__(self):
        self._service_task: asyncio.Task[None] | None = None
        self._stop_requested = False
        self._failure: str | None = None
        self.metric_buffer = MetricBuffer()

    async def start(self) -> None:
        if self._service_task is not None and not self._service_task.done():
            return
        self._stop_requested = False
        self._failure = None
        self._service_task = asyncio.create_task(self._run_service())

    async def stop(self) -> None:
        self._stop_requested = True
        if self._service_task is None or self._service_task.done():
            return
        self._service_task.cancel()
        try:
            await self._service_task
        except asyncio.CancelledError:
            pass

    async def get_service_state(self) -> dict[str, str | bool | None]:
        return {
            'running': self._service_task is not None and not self._service_task.done(),
            'failure': self._failure,
        }

    def drain_metric_records(self) -> list[MetricRecord]:
        return self.metric_buffer.drain()

    def _record_metric(
        self,
        stage: str,
        *,
        context: LoraContext | None = None,
        admission: PartitionAdmission | None = None,
        partition_id: str | None = None,
        values: dict[str, Any] | None = None,
        status: str = 'completed',
        attributes: dict[str, Any] | None = None,
        optimizer_step: int | None = None,
        policy_version: int | None = None,
    ) -> None:
        self.metric_buffer.record(
            MetricRecord(
                stage=stage,
                values=dict(values or {}),
                context_key=context.key if context is not None else None,
                partition_id=admission.partition_id if admission is not None else partition_id,
                partition_index=admission.step if admission is not None else None,
                optimizer_step=optimizer_step,
                policy_version=policy_version,
                status=status,
                attributes=dict(attributes or {}),
            ))

    async def _run_service(self) -> None:
        try:
            await self._serve()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self._failure = f'{type(exc).__name__}: {exc}'

    async def _serve(self) -> None:
        raise NotImplementedError


class RolloutWorker(_Worker):
    """Admits full prompt batches and submits them to the sampler without waiting."""

    def __init__(self,
                 *,
                 context_manager: LoraContextManager,
                 data_plane: TQDataPlane,
                 sampler: Any,
                 prompt_batches: dict[str, Iterable[Sequence[dict[str, Any]]]
                                      | Callable[[], Iterable[Sequence[dict[str, Any]]]]],
                 rollout_config: dict[str, dict[str, Any]],
                 scheduler: SchedulerConfig,
                 allow_partial_rollout: bool = False,
                 persistent: bool = False,
                 idle_delay_s: float = 0.05):
        super().__init__()
        self.data_plane = data_plane
        self.sampler = sampler
        self.context_manager = context_manager
        self.idle_delay_s = idle_delay_s
        self.rollout_config = rollout_config
        self.scheduler = ContextScheduler(scheduler)
        self.allow_partial_rollout = allow_partial_rollout
        self.persistent = persistent
        self._prompt_batch_iterators = {
            key: iter(value() if callable(value) else value)
            for key, value in prompt_batches.items()
        }
        self._next_batch_tasks: dict[str, asyncio.Task[Sequence[dict[str, Any]] | None]] = {}
        self._exhausted: set[str] = set()
        self._contexts_changed = asyncio.Event()

    async def register_context(
        self,
        context: LoraContext,
        prompt_batches: Iterable[Sequence[dict[str, Any]]] | Callable[[], Iterable[Sequence[dict[str, Any]]]],
        rollout_config: dict[str, Any],
    ) -> None:
        key = context.key
        if key in self._prompt_batch_iterators:
            raise KeyError(f'rollout context already exists: {key}')
        self._prompt_batch_iterators[key] = iter(prompt_batches() if callable(prompt_batches) else prompt_batches)
        self.rollout_config[key] = dict(rollout_config)
        self._exhausted.discard(key)
        if self._service_task is not None and not self._service_task.done():
            self._start_next_batch(key)
        self._contexts_changed.set()

    async def unregister_context(self, context: LoraContext | str) -> None:
        key = context if isinstance(context, str) else context.key
        task = self._next_batch_tasks.pop(key, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        iterator = self._prompt_batch_iterators.pop(key, None)
        self.rollout_config.pop(key, None)
        self._exhausted.discard(key)
        close = getattr(iterator, 'close', None)
        if callable(close):
            await asyncio.to_thread(close)
        self._contexts_changed.set()

    def _start_next_batch(self, key: str) -> None:
        if key in self._prompt_batch_iterators and key not in self._next_batch_tasks:
            self._next_batch_tasks[key] = asyncio.create_task(
                asyncio.to_thread(next, self._prompt_batch_iterators[key], None))

    async def stop(self) -> None:
        await super().stop()
        pending = list(self._next_batch_tasks.values())
        self._next_batch_tasks.clear()
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _serve(self) -> None:
        for key in self._prompt_batch_iterators:
            self._start_next_batch(key)
        while not self._stop_requested:
            if not self.persistent and await self.context_manager.is_rollout_admission_closed.remote():
                return
            candidates = []
            for key in list(self._prompt_batch_iterators):
                if key in self._exhausted:
                    continue
                status = await self.context_manager.context_status.remote(key)
                if status is not ContextStatus.ACTIVE:
                    if status in (ContextStatus.EXHAUSTED, ContextStatus.FINISHED):
                        self._exhausted.add(key)
                    continue
                config = self.rollout_config[key]
                task = self._next_batch_tasks.get(key)
                if task is None:
                    self._start_next_batch(key)
                    task = self._next_batch_tasks.get(key)
                if task is not None and task.done():
                    candidates.append(ScheduleCandidate(config['context']))
            candidate = self.scheduler.choose(candidates)
            if candidate is None:
                if not self.persistent and len(self._exhausted) == len(self._prompt_batch_iterators):
                    return
                self._contexts_changed.clear()
                try:
                    await asyncio.wait_for(self._contexts_changed.wait(), timeout=self.idle_delay_s)
                except TimeoutError:
                    pass
                continue
            key = candidate.context.key
            config = self.rollout_config[key]
            batch_task = self._next_batch_tasks[key]
            try:
                batch = batch_task.result()
            except Exception as exc:
                self._next_batch_tasks.pop(key)
                self._record_metric(
                    'rollout',
                    context=candidate.context,
                    status='failed',
                    attributes={'error': f'prompt loading failed: {exc}'},
                )
                raise RuntimeError(f'prompt loading failed for {key}: {exc}') from exc
            if batch is None or len(batch) != int(config['batch_size']):
                self._next_batch_tasks.pop(key)
                self._exhausted.add(key)
                await self.context_manager.on_dataset_exhausted.remote(candidate.context)
                self.scheduler.on_blocked(candidate)
                continue
            admission = await self.context_manager.request_rollout_partition.remote(
                candidate.context,
                target_groups=len(batch),
                num_generations=int(config['num_generations']),
            )
            if admission is None:
                self.scheduler.on_blocked(candidate)
                await asyncio.sleep(self.idle_delay_s)
                continue
            self._next_batch_tasks.pop(key)
            submission_started = time.perf_counter()
            try:
                prepared = await self.data_plane.prepare_rollout_partition(
                    admission,
                    list(batch),
                    config['sampling_params'],
                )
                await asyncio.to_thread(
                    self.sampler.submit_prompt_groups,
                    list(prepared.groups),
                    prepared.sampling_params,
                    self.allow_partial_rollout,
                )
            except Exception as exc:
                self._record_metric(
                    'rollout',
                    context=admission.context,
                    admission=admission,
                    status='failed',
                    attributes={'error': str(exc)},
                )
                raise RuntimeError(f'rollout submission failed for {admission.partition_id}: {exc}') from exc
            self._start_next_batch(key)
            self.scheduler.on_success(candidate)
            self._record_metric(
                'rollout',
                context=admission.context,
                admission=admission,
                status='submitted',
                values={
                    'prompt_count': admission.target_groups,
                    'sample_count': admission.sample_count,
                    'num_generations': admission.num_generations,
                    'rollout_submission_latency_s': time.perf_counter() - submission_started,
                },
                attributes={'scope': 'partition'},
            )


class AdvantageWorker(_Worker):

    def __init__(self,
                 *,
                 context_manager: LoraContextManager,
                 data_plane: TQDataPlane,
                 advantage_fn: Callable[[Any, PartitionAdmission], tuple[Sequence[float], Sequence[float]]],
                 scheduler: SchedulerConfig,
                 persistent: bool = False,
                 idle_delay_s: float = 0.05):
        super().__init__()
        self.data_plane = data_plane
        self.context_manager = context_manager
        self.idle_delay_s = idle_delay_s
        self.advantage_fn = advantage_fn
        self.scheduler = ContextScheduler(scheduler)
        self.persistent = persistent

    async def _serve(self) -> None:
        while not self._stop_requested:
            if not self.persistent and await self.context_manager.is_run_finished.remote():
                return
            admissions = await self.context_manager.list_live_partitions.remote()
            blocked: set[str] = set()
            progressed = False
            for _ in range(len(admissions)):
                candidates = [
                    ScheduleCandidate(admission.context, admission) for admission in admissions
                    if admission.partition_id not in blocked
                ]
                candidate = self.scheduler.choose(candidates)
                if candidate is None:
                    break
                admission = candidate.partition
                batch = await self.data_plane.claim_advantage_batch(admission, 1)
                if batch is None:
                    blocked.add(admission.partition_id)
                    self.scheduler.on_blocked(candidate)
                    continue
                started = time.perf_counter()
                try:
                    advantages, returns = self.advantage_fn(batch.data, admission)
                    await self.data_plane.write_advantages(batch, advantages=advantages, returns=returns)
                except Exception as exc:
                    self._record_metric(
                        'advantage',
                        context=admission.context,
                        admission=admission,
                        status='failed',
                        attributes={'error': str(exc)},
                    )
                    raise RuntimeError(f'advantage failed for {admission.partition_id}: {exc}') from exc
                self.scheduler.on_success(candidate)
                policy = await self.context_manager.get_rollout_policy.remote(admission.context)
                advantage_metrics = advantage_signal_metrics(
                    batch.data['rewards'],
                    advantages,
                    num_generations=admission.num_generations,
                )
                advantage_metrics.update({
                    'sample_count': len(advantages),
                    'advantage_latency_s': time.perf_counter() - started,
                })
                self._record_metric(
                    'advantage',
                    context=admission.context,
                    admission=admission,
                    values=advantage_metrics,
                    policy_version=policy.version,
                )
                progressed = True
                break
            if not progressed:
                await asyncio.sleep(self.idle_delay_s)


class TrainerWorker(_Worker):

    def __init__(self,
                 *,
                 context_manager: LoraContextManager,
                 data_plane: TQDataPlane,
                 train_fn: Callable[[Any, PartitionAdmission], dict[str, Any] | None],
                 save_adapter: Callable[[PartitionAdmission], str],
                 mini_batch_sizes: dict[str, int],
                 scheduler: SchedulerConfig,
                 train_with_config_fn: Callable[[Any, PartitionAdmission, Any], dict[str, Any] | None] | None = None,
                 train_batch_configs: dict[str, Any] | None = None,
                 keep_adapter_versions: int = 0,
                 initial_adapter_paths: dict[str, str] | None = None,
                 remove_adapter: Callable[[str], None] | None = None,
                 evaluation_config: dict[str, dict[str, Any]] | None = None,
                 evaluate_batch: Callable[[Sequence[dict[str, Any]], PartitionAdmission, str, int, Any], dict[str, Any]]
                 | None = None,
                 evaluate_with_reward_fn: Callable[[Sequence[dict[str, Any]], PartitionAdmission, str, int, Any, Any],
                                                   dict[str, Any]] | None = None,
                 evaluation_rewards: dict[str, Any] | None = None,
                 persistent: bool = False,
                 idle_delay_s: float = 0.05):
        super().__init__()
        self.data_plane = data_plane
        self.context_manager = context_manager
        self.idle_delay_s = idle_delay_s
        self.train_fn = train_fn
        self.train_with_config_fn = train_with_config_fn
        self.train_batch_configs = dict(train_batch_configs or {})
        self.save_adapter = save_adapter
        self.mini_batch_sizes = mini_batch_sizes
        self.scheduler = ContextScheduler(scheduler)
        self.keep_adapter_versions = max(0, int(keep_adapter_versions))
        self._adapter_history: dict[str, list[str]] = defaultdict(list)
        for context_key, path in (initial_adapter_paths or {}).items():
            if path:
                self._adapter_history[context_key].append(path)
        self.remove_adapter = remove_adapter or _remove_local_adapter
        self._adapter_removal_tasks: set[asyncio.Task[None]] = set()
        self.evaluation_config = dict(evaluation_config or {})
        self.evaluate_batch = evaluate_batch
        self.evaluate_with_reward_fn = evaluate_with_reward_fn
        self.evaluation_rewards = dict(evaluation_rewards or {})
        self._evaluation_batches: dict[str, list[Sequence[dict[str, Any]]]] = {}
        self._optimizer_steps: dict[str, int] = defaultdict(int)
        self.persistent = persistent

    async def register_context(
        self,
        context: LoraContext,
        *,
        mini_batch_size: int,
        train_batch_config: Any | None = None,
        initial_adapter_path: str | None = None,
        evaluation_config: dict[str, Any] | None = None,
        evaluation_reward: Any | None = None,
    ) -> None:
        key = context.key
        if key in self.mini_batch_sizes:
            raise KeyError(f'trainer context already exists: {key}')
        self.mini_batch_sizes[key] = int(mini_batch_size)
        if train_batch_config is not None:
            self.train_batch_configs[key] = train_batch_config
        if initial_adapter_path:
            self._adapter_history[key].append(initial_adapter_path)
        if evaluation_config is not None:
            self.evaluation_config[key] = dict(evaluation_config)
        if evaluation_reward is not None:
            self.evaluation_rewards[key] = evaluation_reward

    async def unregister_context(self, context: LoraContext | str) -> None:
        key = context if isinstance(context, str) else context.key
        self.mini_batch_sizes.pop(key, None)
        self.train_batch_configs.pop(key, None)
        self.evaluation_config.pop(key, None)
        self.evaluation_rewards.pop(key, None)
        self._evaluation_batches.pop(key, None)
        self._adapter_history.pop(key, None)
        self._optimizer_steps.pop(key, None)

    async def stop(self) -> None:
        await super().stop()
        pending = tuple(self._adapter_removal_tasks)
        if pending:
            await asyncio.gather(*pending)

    async def _serve(self) -> None:
        while not self._stop_requested:
            if not self.persistent and await self.context_manager.is_run_finished.remote():
                return
            admissions = await self.context_manager.list_trainable_partitions.remote()
            blocked: set[str] = set()
            progressed = False
            for _ in range(len(admissions)):
                candidates = [
                    ScheduleCandidate(admission.context, admission) for admission in admissions
                    if admission.partition_id not in blocked
                ]
                candidate = self.scheduler.choose(candidates)
                if candidate is None:
                    break
                admission = candidate.partition
                mini_batch_size = self.mini_batch_sizes[admission.context.key]
                batch = await self.data_plane.claim_training_batch(
                    admission,
                    mini_batch_size // admission.num_generations,
                )
                if batch is None:
                    if await self.data_plane.is_training_consumed(admission):
                        try:
                            await self.context_manager.on_partition_training_started.remote(admission)
                            await self._finish_partition(admission)
                            self.scheduler.on_success(candidate)
                            progressed = True
                            break
                        except Exception as exc:
                            self._record_metric(
                                'train',
                                context=admission.context,
                                admission=admission,
                                status='failed',
                                attributes={'error': str(exc)},
                            )
                            raise RuntimeError(
                                f'training completion failed for {admission.partition_id}: {exc}') from exc
                    blocked.add(admission.partition_id)
                    self.scheduler.on_blocked(candidate)
                    continue
                try:
                    await self.context_manager.on_partition_training_started.remote(admission)
                    policy = await self.context_manager.get_rollout_policy.remote(admission.context)
                    sample_count = len(batch.data['input_ids'])
                    if sample_count != mini_batch_size:
                        raise RuntimeError(
                            f'training claim for {admission.partition_id} returned {sample_count} samples; '
                            f'expected mini_batch_size={mini_batch_size}')
                    started = time.perf_counter()
                    if self.train_with_config_fn is not None:
                        config = self.train_batch_configs[admission.context.key]
                        metrics = dict(self.train_with_config_fn(batch.data, admission, config) or {})
                    else:
                        metrics = dict(self.train_fn(batch.data, admission) or {})
                except Exception as exc:
                    self._record_metric(
                        'train',
                        context=admission.context,
                        admission=admission,
                        status='failed',
                        attributes={'error': str(exc)},
                    )
                    raise RuntimeError(f'training failed for {admission.partition_id}: {exc}') from exc
                self.scheduler.on_success(candidate)
                metrics['sample_count'] = sample_count
                metrics['reward'] = (sum(float(value) for value in batch.data['rewards']) / sample_count)
                metrics['train_latency_s'] = time.perf_counter() - started
                metrics.update(training_policy_metrics(batch.sample_tags, policy.version))
                context_key = admission.context.key
                self._optimizer_steps[context_key] += 1
                optimizer_step = self._optimizer_steps[context_key]
                self._record_metric(
                    'train',
                    context=admission.context,
                    admission=admission,
                    values=metrics,
                    optimizer_step=optimizer_step,
                    policy_version=policy.version,
                )
                progressed = True
                break
            if not progressed:
                await asyncio.sleep(self.idle_delay_s)

    async def _finish_partition(self, admission: PartitionAdmission) -> None:
        finalize_started = time.perf_counter()
        save_started = time.perf_counter()
        adapter_path = self.save_adapter(admission)
        adapter_save_latency_s = time.perf_counter() - save_started
        publish_started = time.perf_counter()
        policy = await self.context_manager.on_partition_trained.remote(admission, adapter_path=adapter_path)
        policy_publish_latency_s = time.perf_counter() - publish_started
        self._record_metric(
            'policy',
            context=admission.context,
            admission=admission,
            values={
                'adapter_save_latency_s': adapter_save_latency_s,
                'policy_publish_latency_s': policy_publish_latency_s,
            },
            attributes={
                'operation': 'publish',
                'adapter_path': adapter_path
            },
            optimizer_step=self._optimizer_steps[admission.context.key],
            policy_version=policy.version,
        )
        await self._evaluate_policy(admission, adapter_path, policy.version)
        clear_started = time.perf_counter()
        await self.data_plane.clear_partition(admission)
        tq_clear_latency_s = time.perf_counter() - clear_started
        release_started = time.perf_counter()
        await self.context_manager.on_partition_cleared.remote(admission)
        partition_release_latency_s = time.perf_counter() - release_started
        self._adapter_history[admission.context.key].append(adapter_path)
        prune_started = time.perf_counter()
        await self._prune_adapter_history(admission.context)
        adapter_prune_schedule_latency_s = time.perf_counter() - prune_started
        self._record_metric(
            'partition',
            context=admission.context,
            admission=admission,
            values={
                'adapter_save_latency_s': adapter_save_latency_s,
                'policy_publish_latency_s': policy_publish_latency_s,
                'tq_clear_latency_s': tq_clear_latency_s,
                'partition_release_latency_s': partition_release_latency_s,
                'adapter_prune_schedule_latency_s': adapter_prune_schedule_latency_s,
                'partition_finalize_latency_s': time.perf_counter() - finalize_started,
            },
            optimizer_step=self._optimizer_steps[admission.context.key],
            policy_version=policy.version,
        )

    async def _evaluate_policy(self, admission: PartitionAdmission, adapter_path: str, policy_version: int) -> None:
        config = self.evaluation_config.get(admission.context.key)
        if config is None or (self.evaluate_batch is None and self.evaluate_with_reward_fn is None):
            return
        interval = int(config['interval'])
        if policy_version % interval:
            return

        context_key = admission.context.key
        if context_key not in self._evaluation_batches:
            source = config['prompt_batches']
            self._evaluation_batches[context_key] = list(source() if callable(source) else source)
        batches = self._evaluation_batches[context_key]
        started = time.perf_counter()
        rewards: list[float] = []
        completion_lengths: list[int] = []
        prompt_count = 0
        for batch in batches:
            if self.evaluate_with_reward_fn is not None:
                result = await asyncio.to_thread(
                    self.evaluate_with_reward_fn,
                    batch,
                    admission,
                    adapter_path,
                    policy_version,
                    config['sampling_params'],
                    self.evaluation_rewards[context_key],
                )
            else:
                result = await asyncio.to_thread(
                    self.evaluate_batch,
                    batch,
                    admission,
                    adapter_path,
                    policy_version,
                    config['sampling_params'],
                )
            rewards.extend(float(value) for value in result['rewards'])
            completion_lengths.extend(int(value) for value in result['completion_lengths'])
            prompt_count += len(batch)
        if not rewards:
            raise ValueError(f'evaluation dataset is empty for {context_key}')
        self._record_metric(
            'evaluation',
            context=admission.context,
            admission=admission,
            values={
                'accuracy': sum(rewards) / len(rewards),
                'sample_count': len(rewards),
                'prompt_count': prompt_count,
                'completion_length': sum(completion_lengths) / len(completion_lengths),
                'eval_latency_s': time.perf_counter() - started,
            },
            attributes={'eval_dataset': config['dataset_name']},
            optimizer_step=self._optimizer_steps[context_key],
            policy_version=policy_version,
        )

    async def _prune_adapter_history(self, context: LoraContext) -> None:
        protected = set(await self.context_manager.adapter_paths_to_keep.remote())
        context_key = context.key
        history = self._adapter_history[context_key]
        retained_history = set(history[-self.keep_adapter_versions:]) if self.keep_adapter_versions else set()
        retained = protected | retained_history
        stale = [path for path in history if path not in retained]
        self._adapter_history[context_key] = [path for path in history if path in retained]
        for path in stale:
            task = asyncio.create_task(self._remove_adapter(context, path))
            self._adapter_removal_tasks.add(task)
            task.add_done_callback(self._adapter_removal_tasks.discard)

    async def _remove_adapter(self, context: LoraContext, path: str) -> None:
        started = time.perf_counter()
        try:
            await asyncio.to_thread(self.remove_adapter, path)
        except OSError as exc:
            self._record_metric(
                'policy',
                context=context,
                status='failed',
                values={
                    'adapter_prune_latency_s': time.perf_counter() - started,
                },
                attributes={
                    'operation': 'adapter_prune',
                    'adapter_path': path,
                    'error': str(exc)
                },
            )
            return
        self._record_metric(
            'policy',
            context=context,
            values={
                'adapter_prune_latency_s': time.perf_counter() - started,
            },
            attributes={
                'operation': 'adapter_prune',
                'adapter_path': path
            },
        )


def _remove_local_adapter(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
