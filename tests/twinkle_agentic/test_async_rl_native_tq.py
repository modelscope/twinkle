from __future__ import annotations

import asyncio
import inspect

import pytest

from twinkle.data_format import SampledSequence, SampleResponse
from twinkle.metric import MetricRecord
from twinkle_agentic.async_rl import (AsyncMultiLoraGRPOPipeline, ContextSchedulePolicy, ContextScheduler,
                                      ContextStatus, LoraContext, LoraContextManager, ScheduleCandidate,
                                      SchedulerConfig, TQDataPlane, TrainerWorker)
from twinkle_agentic.async_rl.metrics import training_policy_metrics
from twinkle_agentic.async_rl.native_tq import ContextGRPOGroupNSampler
from twinkle_agentic.async_rl.pipeline import (
    _collect_adapter_path,
    _require_adapter_path,
    _train_batch,
    create_cpu_actor,
)
from twinkle_agentic.async_rl.types import PartitionAdmission, PreparedPartition
from twinkle_agentic.async_rl.utils import (
    TrainBatchConfig,
    sample_responses_to_rollout_rows,
)
from twinkle_agentic.async_rl.workers import RolloutWorker


class LocalActorHandle:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        method = getattr(self.target, name)

        class RemoteMethod:
            async def remote(_, *args, **kwargs):
                result = method(*args, **kwargs)
                return await result if inspect.isawaitable(result) else result

        return RemoteMethod()


def test_cpu_service_actor_uses_twinkle_ray_mode(monkeypatch):
    import ray

    captured = {}

    class ActorClass:

        @staticmethod
        def remote(*args, **kwargs):
            captured['actor_args'] = args
            captured['actor_kwargs'] = kwargs
            return 'actor'

    def fake_remote(**options):
        captured['options'] = options
        return lambda cls: ActorClass

    monkeypatch.setattr(ray, 'remote', fake_remote)

    assert create_cpu_actor(object, 'value', enabled=True) == 'actor'
    assert captured['options'] == {
        'num_cpus': 1,
        'runtime_env': {
            'env_vars': {
                'TWINKLE_MODE': 'ray'
            }
        },
    }
    assert captured['actor_args'] == ('value', )
    assert captured['actor_kwargs'] == {'enabled': True}


def test_train_batch_preserves_position_ids_from_tq():
    class Batch(dict):
        batch_size = (1, )

    class Model:
        inputs = None

        def forward_backward(self, *, inputs, **_kwargs):
            self.inputs = inputs

        def clip_grad_and_step(self, **_kwargs):
            return None

        def calculate_metric(self, **_kwargs):
            return {}

    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 1, 0)
    data = Batch({
        'input_ids': [[1, 2]],
        'labels': [[-100, 2]],
        'attention_mask': [[1, 1]],
        'position_ids': [[0, 1]],
        'logprobs': [[-.1]],
        'advantages': [1.],
        'rewards': [1.],
    })
    model = Model()

    _train_batch(model, {context.key: TrainBatchConfig(1, 1)}, data, admission)

    assert model.inputs == [{
        'input_ids': [1, 2],
        'labels': [-100, 2],
        'attention_mask': [1, 1],
        'position_ids': [0, 1],
    }]


def test_train_batch_accumulates_real_micro_batches_before_one_optimizer_step():
    class Batch(dict):
        batch_size = (4, )

    class Model:
        def __init__(self):
            self.calls = []
            self.optimizer_steps = 0

        def forward_backward(self, **kwargs):
            self.calls.append(kwargs)
            return lambda: {}

        def clip_grad_and_step(self, **_kwargs):
            self.optimizer_steps += 1

        def calculate_metric(self, **_kwargs):
            return {}

    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 4, 0)
    data = Batch({
        'input_ids': [[index] for index in range(4)],
        'labels': [[index] for index in range(4)],
        'attention_mask': [[1] for _ in range(4)],
        'position_ids': [[0] for _ in range(4)],
        'logprobs': [[-.1] for _ in range(4)],
        'advantages': [1., 2., 3., 4.],
        'rewards': [1., 1., 1., 1.],
    })
    model = Model()

    metrics = _train_batch(
        model,
        {context.key: TrainBatchConfig(4, 1)},
        data,
        admission,
        model_data_parallel_size=1,
    )

    assert [len(call['inputs']) for call in model.calls] == [4]
    assert model.calls[0]['advantages'] == [1., 2., 3., 4.]
    assert model.calls[0]['micro_batch_size'] == 1
    assert model.calls[0]['loss_scale'] == 1.0
    assert model.optimizer_steps == 1
    assert metrics['micro_batch_size_per_rank'] == 1


def _context(name: str = 'adapter') -> LoraContext:
    return LoraContext('tenant', f'run_{name}', 'model', name)


def _sample_response(tokens, stop_reason, input_ids):
    return SampleResponse(
        prompt_token_ids=[1, 2],
        sequences=[
            SampledSequence(
                stop_reason=stop_reason,
                tokens=tokens,
                logprobs=[[(token, -.1)] for token in tokens],
                new_input_feature={
                    'input_ids': input_ids,
                    'labels': [-100, -100, *tokens],
                },
            )
        ],
    )


def test_evaluation_rows_do_not_require_rollout_group_metadata():
    prompt = {'input_ids': [1, 2], 'labels': [-100, -100]}

    rows = sample_responses_to_rollout_rows(
        [prompt],
        [_sample_response([3], 'stop', [1, 2, 3])],
        policy_version=10,
    )

    assert len(rows) == 1
    assert 'group_id' not in rows[0]
    assert 'generation_idx' not in rows[0]
    assert rows[0]['rollout_policy_version'] == 10


def test_training_rows_preserve_rollout_group_metadata():
    source = {
        'input_ids': [1, 2],
        'labels': [-100, -100],
        'group_id': 'partition/group_0',
        'generation_idx': 2,
    }

    rows = sample_responses_to_rollout_rows(
        [source],
        [_sample_response([3], 'stop', [1, 2, 3])],
        policy_version=4,
    )

    assert rows[0]['group_id'] == 'partition/group_0'
    assert rows[0]['generation_idx'] == 2


def test_adapter_path_rejects_uncollected_remote_result():
    def lazy_result():
        return '/tmp/policy'

    with pytest.raises(TypeError, match='must return a non-empty checkpoint path string'):
        _require_adapter_path(lazy_result, operation='test save')


def test_adapter_path_collects_lazy_remote_result():
    def lazy_result():
        return '/tmp/policy'

    lazy_result._is_lazy_collect = True
    assert _collect_adapter_path(lazy_result, operation='test save') == '/tmp/policy'


def test_training_policy_metrics_use_final_version_and_partial_span():
    metrics = training_policy_metrics((
        {
            'final_policy_version': 3,
            'policy_version_span': 1
        },
        {
            'final_policy_version': 4,
            'policy_version_span': 0
        },
    ), train_policy_version=5)

    assert metrics == {
        'policy_version_gap_mean': 1.5,
        'policy_version_gap_p95': 2,
        'policy_version_gap_max': 2,
        'rollout_policy_span_mean': 0.5,
        'rollout_policy_span_max': 1,
    }


def test_training_policy_metrics_reject_future_rollout_version():
    try:
        training_policy_metrics(({
            'final_policy_version': 6,
            'policy_version_span': 0
        }, ), train_policy_version=5)
    except ValueError as exc:
        assert 'older than rollout versions' in str(exc)
    else:
        raise AssertionError('expected a future rollout policy version to fail')


def test_unified_staleness_admission():
    context = _context()
    zero = LoraContextManager(max_staleness=0)
    zero.register_context(context)
    first = zero.request_rollout_partition(context, target_groups=1, num_generations=2)
    assert first is not None
    assert zero.request_rollout_partition(context, target_groups=1, num_generations=2) is None

    one = LoraContextManager(max_staleness=1)
    one.register_context(context)
    assert one.request_rollout_partition(context, target_groups=1, num_generations=2) is not None
    assert one.request_rollout_partition(context, target_groups=1, num_generations=2) is not None
    assert one.request_rollout_partition(context, target_groups=1, num_generations=2) is None


def test_rollout_worker_retains_prefetched_batch_until_admission_succeeds():
    context = _context()

    class AdmissionGate:
        def __init__(self):
            self.blocked = True
            self.attempts = 0
            self.accepted = False

        def is_rollout_admission_closed(self):
            return False

        def context_status(self, _context):
            return ContextStatus.ACTIVE

        def request_rollout_partition(self, _context, *, target_groups, num_generations):
            self.attempts += 1
            if self.blocked or self.accepted:
                return None
            self.accepted = True
            return PartitionAdmission(context, context.partition_id(0), 0, target_groups, num_generations, 0)

    class DataPlane:
        async def prepare_rollout_partition(self, admission, _prompts, sampling_params):
            return PreparedPartition(admission, (), sampling_params)

    class Sampler:
        def __init__(self, loop):
            self.submitted = asyncio.Event()
            self.loop = loop

        def submit_prompt_groups(self, _groups, _sampling_params, _allow_partial_rollout):
            self.loop.call_soon_threadsafe(self.submitted.set)

    loaded_batches = []

    def batches():
        for value in (1, 2):
            loaded_batches.append(value)
            yield [{'input_ids': [value]}]

    async def run():
        manager = AdmissionGate()
        sampler = Sampler(asyncio.get_running_loop())
        worker = RolloutWorker(
            context_manager=LocalActorHandle(manager),
            data_plane=DataPlane(),
            sampler=sampler,
            prompt_batches={context.key: batches()},
            rollout_config={
                context.key: {
                    'context': context,
                    'batch_size': 1,
                    'num_generations': 2,
                    'sampling_params': {},
                }
            },
            scheduler=SchedulerConfig(ContextSchedulePolicy.ROUND_ROBIN, 1),
            idle_delay_s=.001,
        )
        await worker.start()
        while manager.attempts == 0:
            await asyncio.sleep(.001)
        prefetched_task = worker._next_batch_tasks[context.key]
        await asyncio.sleep(.01)
        assert worker._next_batch_tasks[context.key] is prefetched_task
        assert loaded_batches == [1]

        manager.blocked = False
        await asyncio.wait_for(sampler.submitted.wait(), timeout=1)
        while loaded_batches == [1]:
            await asyncio.sleep(.001)
        assert loaded_batches == [1, 2]
        await worker.stop()

    asyncio.run(run())


def test_partition_clear_releases_capacity_only_after_publish():
    context = _context()
    manager = LoraContextManager(max_staleness=0)
    manager.register_context(context, adapter_path='initial')
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    manager.on_partition_training_started(admission)
    policy = manager.on_partition_trained(admission, adapter_path='v1')
    assert policy.version == 1
    assert manager.request_rollout_partition(context, target_groups=1, num_generations=2) is None
    manager.on_partition_cleared(admission)
    assert manager.request_rollout_partition(context, target_groups=1, num_generations=2) is not None


def test_context_trains_partitions_in_step_order():
    context = _context()
    manager = LoraContextManager(max_staleness=1)
    manager.register_context(context)
    first = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    second = manager.request_rollout_partition(context, target_groups=1, num_generations=2)

    assert manager.list_trainable_partitions() == [first]
    manager.on_partition_training_started(first)
    assert manager.list_trainable_partitions() == [first]

    try:
        manager.on_partition_training_started(second)
    except RuntimeError as exc:
        assert f'already trains {first.partition_id}' in str(exc)
    else:
        raise AssertionError('expected the next partition to remain blocked')

    manager.on_partition_trained(first, adapter_path='v1')
    manager.on_partition_cleared(first)
    assert manager.list_trainable_partitions() == [second]
    manager.on_partition_training_started(second)


def test_scheduler_supports_round_robin_sticky_and_oldest():
    a, b = _context('a'), _context('b')
    candidates = [ScheduleCandidate(a), ScheduleCandidate(b)]
    round_robin = ContextScheduler(SchedulerConfig(ContextSchedulePolicy.ROUND_ROBIN, 1))
    assert round_robin.choose(candidates).context == a
    round_robin.on_success(candidates[0])
    assert round_robin.choose(candidates).context == b

    sticky = ContextScheduler(SchedulerConfig(ContextSchedulePolicy.STICKY, None))
    sticky.on_success(candidates[1])
    assert sticky.choose(candidates).context == b
    sticky.on_blocked(candidates[1])
    assert sticky.choose(candidates).context == a

    capped = ContextScheduler(SchedulerConfig(ContextSchedulePolicy.STICKY, 1))
    first = capped.choose(candidates)
    capped.on_success(first)
    assert capped.choose(candidates).context == b

    manager = LoraContextManager(max_staleness=2)
    manager.register_context(a)
    manager.register_context(b)
    old = manager.request_rollout_partition(a, target_groups=1, num_generations=2)
    new = manager.request_rollout_partition(b, target_groups=1, num_generations=2)
    oldest = ContextScheduler(SchedulerConfig(ContextSchedulePolicy.OLDEST_PARTITION, 1))
    assert oldest.choose([ScheduleCandidate(b, new), ScheduleCandidate(a, old)]).partition == old


def test_context_group_sampler_uses_request_generation_count():
    sampler = ContextGRPOGroupNSampler()

    selected, consumed = sampler.sample(
        [0, 1, 4, 5, 6, 7],
        batch_size=4,
        partition_id='train_0',
        task_name='advantage/context',
        n_samples_per_prompt=4,
    )

    assert selected == [4, 5, 6, 7]
    assert consumed == selected

    selected, consumed = sampler.sample(
        [8, 9, 12],
        batch_size=2,
        partition_id='train_1',
        task_name='advantage/context',
        n_samples_per_prompt=2,
    )

    assert selected == [8, 9]
    assert consumed == selected


def test_context_finishes_after_exhaustion_and_clear():
    context = _context()
    manager = LoraContextManager()
    manager.register_context(context)
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    manager.on_dataset_exhausted(context)
    assert not manager.is_run_finished()
    manager.on_partition_training_started(admission)
    manager.on_partition_trained(admission, adapter_path='v1')
    manager.on_partition_cleared(admission)
    assert manager.is_run_finished()


def test_pipeline_fails_fast_when_a_worker_service_fails():
    context = _context()
    manager = LoraContextManager()
    manager.register_context(context)

    class FailedWorker:
        async def start(self):
            return None

        async def stop(self):
            return None

        async def get_service_state(self):
            return {'running': False, 'failure': 'CUDA out of memory'}

        def drain_metric_records(self):
            return []

    worker = LocalActorHandle(FailedWorker())
    pipeline = AsyncMultiLoraGRPOPipeline(
        context_manager=LocalActorHandle(manager),
        rollout_worker=worker,
        advantage_worker=worker,
        trainer_worker=worker,
    )

    try:
        asyncio.run(pipeline.run_async())
    except RuntimeError as exc:
        assert 'CUDA out of memory' in str(exc)
    else:
        raise AssertionError('expected worker failure to fail the pipeline')


def test_pipeline_drains_actor_metric_buffers_when_reporting_is_disabled():
    class BufferedWorker:
        def __init__(self):
            self.drain_count = 0

        def drain_metric_records(self):
            self.drain_count += 1
            return [MetricRecord(stage='train', values={'loss': 1.0})]

    class BufferedSampler:
        def __init__(self):
            self.drain_count = 0

        def drain_metric_records(self):
            self.drain_count += 1
            return [MetricRecord(stage='rollout', values={'sample_count': 1})]

    workers = [BufferedWorker() for _ in range(3)]
    sampler = BufferedSampler()
    pipeline = AsyncMultiLoraGRPOPipeline(
        context_manager=object(),
        rollout_worker=LocalActorHandle(workers[0]),
        advantage_worker=LocalActorHandle(workers[1]),
        trainer_worker=LocalActorHandle(workers[2]),
        sampler=sampler,
        metrics=None,
    )

    asyncio.run(pipeline._drain_metrics())

    assert [worker.drain_count for worker in workers] == [1, 1, 1]
    assert sampler.drain_count == 1


def test_global_max_steps_limits_admission_and_closes_after_completion():
    first, second = _context('a'), _context('b')
    manager = LoraContextManager(max_staleness=1, max_steps=1)
    manager.register_context(first)
    manager.register_context(second)
    first_admission = manager.request_rollout_partition(first, target_groups=1, num_generations=2)
    assert manager.request_rollout_partition(second, target_groups=1, num_generations=2) is None
    manager.on_partition_training_started(first_admission)
    manager.on_partition_trained(first_admission, adapter_path='v1')
    manager.on_partition_cleared(first_admission)
    assert manager.is_rollout_admission_closed()
    assert manager.is_run_finished()


def test_zero_max_steps_finishes_without_admission():
    context = _context()
    manager = LoraContextManager(max_steps=0)
    manager.register_context(context)
    assert manager.request_rollout_partition(context, target_groups=1, num_generations=2) is None
    assert manager.is_rollout_admission_closed()
    assert manager.is_run_finished()


def test_checkpoint_retention_preserves_current_policy_and_history_window():
    context = _context()
    manager = LoraContextManager()
    manager.register_context(context, adapter_path='initial')
    removed = []
    worker = TrainerWorker(
        context_manager=LocalActorHandle(manager),
        data_plane=TQDataPlane(),
        train_fn=lambda _data, _admission: {},
        save_adapter=lambda _admission: 'unused',
        mini_batch_sizes={context.key: 2},
        scheduler=SchedulerConfig(ContextSchedulePolicy.STICKY, None),
        keep_adapter_versions=1,
        initial_adapter_paths={context.key: 'initial'},
        remove_adapter=removed.append,
    )
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    manager.on_partition_training_started(admission)
    manager.on_partition_trained(admission, adapter_path='current')
    manager.on_partition_cleared(admission)
    worker._adapter_history[context.key].append('current')
    async def prune():
        await worker._prune_adapter_history(context)
        await worker.stop()

    asyncio.run(prune())
    assert removed == ['initial']
    assert worker._adapter_history[context.key] == ['current']
    prune_events = [
        record for record in worker.drain_metric_records()
        if record.stage == 'policy' and record.attributes.get('operation') == 'adapter_prune'
    ]
    assert len(prune_events) == 1
    assert prune_events[0].context_key == context.key
    assert prune_events[0].attributes['adapter_path'] == 'initial'
    assert prune_events[0].values['adapter_prune_latency_s'] >= 0


def test_policy_retention_keeps_only_current_and_actively_referenced_paths():
    context = _context()
    manager = LoraContextManager(max_staleness=1)
    manager.register_context(context, adapter_path='initial')
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)

    acquired = manager.acquire_rollout_policy(context)
    manager.on_partition_training_started(admission)
    manager.on_partition_trained(admission, adapter_path='current')

    assert manager.adapter_paths_to_keep() == {'initial', 'current'}
    manager.release_rollout_policy(acquired)
    assert manager.adapter_paths_to_keep() == {'current'}


def test_trainer_periodically_evaluates_published_policy():
    context = _context()
    manager = LoraContextManager()
    manager.register_context(context, adapter_path='initial')
    admission = manager.request_rollout_partition(context, target_groups=1, num_generations=2)
    calls = []

    def evaluate_batch(batch, evaluated_admission, adapter_path, policy_version, sampling_params):
        calls.append((list(batch), evaluated_admission, adapter_path, policy_version, sampling_params))
        return {
            'rewards': [1.0] * len(batch),
            'completion_lengths': [10] * len(batch),
        }

    worker = TrainerWorker(
        context_manager=LocalActorHandle(manager),
        data_plane=TQDataPlane(),
        train_fn=lambda _data, _admission: {},
        save_adapter=lambda _admission: 'unused',
        mini_batch_sizes={context.key: 2},
        scheduler=SchedulerConfig(ContextSchedulePolicy.STICKY, None),
        evaluation_config={
            context.key: {
                'interval': 5,
                'dataset_name': 'validation',
                'prompt_batches': lambda: [[{'input_ids': [1]}], [{'input_ids': [2]}]],
                'sampling_params': 'params',
            }
        },
        evaluate_batch=evaluate_batch,
    )
    worker._optimizer_steps[context.key] = 50

    async def evaluate():
        await worker._evaluate_policy(admission, 'adapter-v4', 4)
        await worker._evaluate_policy(admission, 'adapter-v5', 5)

    asyncio.run(evaluate())
    assert len(calls) == 2
    records = [record for record in worker.drain_metric_records() if record.stage == 'evaluation']
    assert len(records) == 1
    assert records[0].policy_version == 5
    assert records[0].optimizer_step == 50
    assert records[0].values['accuracy'] == 1.0
    assert records[0].values['prompt_count'] == 2
    assert records[0].values['sample_count'] == 2
    assert records[0].values['completion_length'] == 10
