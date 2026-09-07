from __future__ import annotations

import asyncio
import inspect
import json
import time
from concurrent.futures import Future

import pytest

from twinkle import DeviceMesh
from twinkle.data_format import SampledSequence, SampleResponse, SamplingParams
from twinkle.infra import _dispatch_args
from twinkle.server.sampler.twinkle_handlers import _await_generation
from twinkle_agentic.async_rl import LoraContext
from twinkle_agentic.async_rl.types import PartitionAdmission, PromptGroup, RolloutPolicy
from twinkle_agentic.async_rl.vllm_sampler_tq import (
    VLLMSamplerTQ,
    _GeneratedSample,
    _PromptGroupRolloutStats,
    _dispatch_generation,
)


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


class PolicyProvider:
    def __init__(self, policies):
        self.policies = iter(policies)
        self.released = []

    def get_rollout_policy(self, _context):
        return next(self.policies)

    def acquire_rollout_policy(self, context):
        return self.get_rollout_policy(context)

    def release_rollout_policy(self, policy):
        self.released.append(policy)


class GenerationHarness:
    _merge_partial_responses = VLLMSamplerTQ._merge_partial_responses

    def __init__(self, policies, responses):
        self.context_manager = LocalActorHandle(PolicyProvider(policies))
        self.responses = iter(responses)
        self.rollout_max_retries = 1
        self.rollout_retry_delay_s = 0
        self.calls = []
        self.template = type('Template', (), {'decode': staticmethod(lambda tokens: str(tokens))})()

    async def _load_lora_for_policy(self, policy):
        return policy.version

    async def _sample_single(self, feat, sampling_params, *, lora_request, multi_modal_data, logprobs_only):
        self.calls.append((list(feat['input_ids']), sampling_params.max_tokens, lora_request))
        return next(self.responses)


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


def _bare_sampler() -> VLLMSamplerTQ:
    sampler = object.__new__(VLLMSamplerTQ)
    sampler._generation_submissions = {}
    return sampler


def test_generation_dispatch_allows_one_prompt_with_multiple_dp_workers() -> None:
    assert VLLMSamplerTQ.submit_generation._dispatch is _dispatch_generation
    assert VLLMSamplerTQ.submit_prompt_groups._dispatch == 'slice_dp'
    shards = [
        _dispatch_generation(
            3,
            worker_index,
            ('submission', [{'input_ids': [1]}], 'params'),
            {},
        )[0][1]
        for worker_index in range(3)
    ]

    assert shards == [[{'input_ids': [1]}], [], []]


def test_generation_submission_returns_before_generation_finishes() -> None:
    sampler = _bare_sampler()
    pending = Future()
    submitted_coroutines = []

    def submit(coro):
        submitted_coroutines.append(coro)
        coro.close()
        return pending

    sampler._submit_in_loop = submit

    result = sampler.submit_generation(
        'submission-1',
        [{'input_ids': [1]}],
        SamplingParams(max_tokens=4),
    )

    assert result == {'submission_id': 'submission-1', 'status': 'running'}
    assert not pending.done()
    assert len(submitted_coroutines) == 1
    assert sampler.get_generation_status('submission-1')['status'] == 'running'

    responses = [object()]
    pending.set_result(responses)
    assert sampler.get_generation_status('submission-1')['status'] == 'completed'
    assert sampler.collect_generation('submission-1') == responses
    assert 'submission-1' not in sampler._generation_submissions


def test_generation_keeps_one_response_per_prompt() -> None:
    sampler = _bare_sampler()
    sampler.template = None

    async def sample_single(feat, _params, **_kwargs):
        await asyncio.sleep(0)
        return feat['input_ids'][0]

    sampler._sample_single = sample_single
    responses = asyncio.run(
        sampler._generate_inputs(
            [{'input_ids': [10]}, {'input_ids': [20]}],
            SamplingParams(max_tokens=4),
            adapter_name='',
            adapter_path=None,
            use_base_model=False,
        ))

    assert responses == [10, 20]


def test_generation_failure_is_isolated_and_consumed() -> None:
    sampler = _bare_sampler()
    failed = Future()
    failed.set_exception(ValueError('bad prompt'))
    sampler._generation_submissions['failed'] = failed

    state = sampler.get_generation_status('failed')
    assert state['status'] == 'failed'
    assert state['error'] == 'ValueError: bad prompt'

    with pytest.raises(ValueError, match='bad prompt'):
        sampler.collect_generation('failed')
    assert 'failed' not in sampler._generation_submissions


def test_generation_can_be_cancelled_without_waiting() -> None:
    sampler = _bare_sampler()
    pending = Future()
    sampler._generation_submissions['pending'] = pending

    state = sampler.cancel_generation('pending')

    assert state == {'submission_id': 'pending', 'status': 'cancelled'}
    assert pending.cancelled()
    assert 'pending' not in sampler._generation_submissions


def test_all_generations_are_cancelled_on_shutdown() -> None:
    sampler = _bare_sampler()
    first = Future()
    second = Future()
    sampler._generation_submissions.update(first=first, second=second)

    state = sampler.cancel_all_generations()

    assert state == {'submissions': 2, 'cancelled': 2}
    assert first.cancelled()
    assert second.cancelled()
    assert sampler._generation_submissions == {}


def test_native_prompt_group_sampling_requires_context_manager() -> None:
    sampler = _bare_sampler()
    sampler.context_manager = None

    with pytest.raises(RuntimeError, match='context_manager is required'):
        sampler.submit_prompt_groups([], SamplingParams(max_tokens=4))


def test_server_waiter_admits_later_submission_before_first_finishes() -> None:

    class Sampler:

        def __init__(self):
            self.futures: dict[str, Future] = {}
            self.submission_order = []

        def submit_generation(self, submission_id, *_args, **_kwargs):
            self.submission_order.append(submission_id)
            self.futures[submission_id] = Future()

        def get_generation_status(self, submission_id):
            future = self.futures[submission_id]
            return {'status': 'completed' if future.done() else 'running'}

        def collect_generation(self, submission_id):
            return self.futures[submission_id].result()

        def cancel_generation(self, submission_id):
            self.futures.pop(submission_id, None)

    sampler = Sampler()

    async def run():
        sampler.submit_generation('first')
        sampler.submit_generation('second')
        first = asyncio.create_task(
            _await_generation(sampler, 'first'))
        second = asyncio.create_task(
            _await_generation(sampler, 'second'))
        while len(sampler.submission_order) < 2:
            await asyncio.sleep(0)
        assert not first.done()
        sampler.futures['second'].set_result(['short'])
        assert await second == ['short']
        assert not first.done()
        sampler.futures['first'].set_result(['long'])
        assert await first == ['long']

    asyncio.run(run())
    assert set(sampler.submission_order) == {'first', 'second'}


def test_server_waiter_retries_cancelled_status_poll() -> None:
    from ray.exceptions import TaskCancelledError

    class Sampler:

        def __init__(self):
            self.status_calls = 0
            self.cancelled = False

        def submit_generation(self, *_args, **_kwargs):
            return None

        def get_generation_status(self, _submission_id):
            self.status_calls += 1
            if self.status_calls == 1:
                raise TaskCancelledError()
            return {'status': 'completed'}

        def collect_generation(self, _submission_id):
            return ['completed']

        def cancel_generation(self, _submission_id):
            self.cancelled = True

    sampler = Sampler()
    sampler.submit_generation('submission')
    result = asyncio.run(
        _await_generation(sampler, 'submission'))

    assert result == ['completed']
    assert sampler.status_calls == 2
    assert not sampler.cancelled


def test_sampler_dp_dispatch_slices_complete_groups_without_duplication():
    mesh = DeviceMesh.from_sizes(world_size=4, dp_size=2, tp_size=2)
    groups = ['group_0', 'group_1', 'group_2', 'group_3']
    dispatched = _dispatch_args(
        workers=['dp_0', 'dp_1'],
        dispatch='slice_dp',
        execute='all',
        device_mesh=mesh,
        args=(groups, 'sampling_params', False),
        kwargs={},
    )

    assert [worker for worker, _, _ in dispatched] == ['dp_0', 'dp_1']
    assert [args[0] for _, args, _ in dispatched] == [groups[:2], groups[2:]]
    assert [group for _, args, _ in dispatched for group in args[0]] == groups


@pytest.mark.parametrize(('dp_size', 'expected_scope'), [(1, 'partition'), (2, 'shard')])
def test_sampler_reports_submission_throughput_at_partition_or_shard_scope(dp_size, expected_scope):
    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 2, 2, 0)
    groups = [
        PromptGroup(context, admission, f'{admission.partition_id}/group_{index}', {}, object())
        for index in range(2)
    ]

    class RolloutMetricsHarness:
        def __init__(self):
            self.device_mesh = DeviceMesh.from_sizes(world_size=dp_size, dp_size=dp_size)
            self.events = []

        async def _run_prompt_group(self, *, group, **_kwargs):
            index = int(group.group_id.rsplit('_', 1)[1])
            lengths = ((10, 20), (30, 40))[index]
            reasons = (('stop', 'length'), ('stop', 'stop'))[index]
            return _PromptGroupRolloutStats(lengths, reasons, (index + 1, index + 1))

        def _record_metrics(self, group, values, **kwargs):
            self.events.append((group, values, kwargs))

    sampler = RolloutMetricsHarness()
    asyncio.run(
        VLLMSamplerTQ._sample_prompt_groups(
            sampler,
            'submission',
            groups,
            SamplingParams(max_tokens=64),
            False,
            time.perf_counter() - 1,
        ))

    recorded_group, metrics, record_options = sampler.events[-1]
    assert recorded_group.context == context
    assert recorded_group.partition_id == admission.partition_id
    assert record_options['attributes']['scope'] == expected_scope
    assert metrics['prompt_group_count'] == 2
    assert metrics['sample_count'] == 4
    assert metrics['output_tokens'] == 100
    assert metrics['completion_length_mean'] == 25
    assert metrics['completion_truncated_count'] == 1
    assert metrics['policy_version_min'] == 1
    assert metrics['policy_version_max'] == 2
    assert metrics['sampler_dp_size'] == dp_size
    assert metrics['output_tokens_per_s'] == pytest.approx(100 / metrics['rollout_latency_s'])


def test_sampler_writes_one_atomic_rollout_file_per_prompt_group(tmp_path):
    context = _context()
    admission = PartitionAdmission(context, context.partition_id(3), 3, 1, 2, 0)
    group = PromptGroup(
        context,
        admission,
        f'{admission.partition_id}/group_0',
        {'user_data': [('ground_truth', '"42"')]},
        object(),
    )
    policy = RolloutPolicy(context.key, context.adapter_name, 7, '/tmp/adapter-v7')
    generated = [
        _GeneratedSample(
            SampleResponse(
                sequences=[SampledSequence('stop', [20 + index], decoded=f'completion-{index}')],
                prompt_token_ids=[10, 11],
            ),
            (policy,),
            attempts=1,
            was_aborted=False,
            resumed_partial_output=False,
        )
        for index in range(2)
    ]
    rows = [
        {
            'generation_idx': index,
            'rollout_policy_version': 7,
            'initial_policy_version': 7,
            'final_policy_version': 7,
            'rollout_policy_versions': [7],
            'rollout_adapter_path': '/tmp/adapter-v7',
            'stop_reason': 'stop',
            'logprobs': [-0.1],
        }
        for index in range(2)
    ]

    class Template:
        @staticmethod
        def decode(token_ids, **_kwargs):
            return ' '.join(map(str, token_ids))

    sampler = object.__new__(VLLMSamplerTQ)
    sampler.rollout_output_dir = tmp_path
    sampler.rollout_output_include_token_ids = False
    sampler.template = Template()

    sampler._write_rollout_group('submission-1', group, generated, rows, [1.0, 0.0])
    sampler._write_rollout_group('submission-2', group, generated, rows, [1.0, 0.0])

    output_path = (
        tmp_path
        / context.tenant_id
        / context.training_run_id
        / context.adapter_name
        / 'policy_7'
        / 'train_3-group_0.jsonl'
    )
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]['submission_id'] == 'submission-2'
    assert records[0]['prompt'] == '10 11'
    assert records[0]['completion'] == '20'
    assert records[0]['ground_truth'] == '42'
    assert records[0]['reward'] == 1.0
    assert records[0]['head_version'] == 7
    assert records[0]['tail_version'] == 7
    assert 'prompt_token_ids' not in records[0]


def test_aborted_generation_restarts_from_original_prompt_when_partial_is_disabled():
    context = _context()
    policies = [
        RolloutPolicy(context.key, context.adapter_name, 3, 'adapter-v3'),
        RolloutPolicy(context.key, context.adapter_name, 4, 'adapter-v4'),
    ]
    sampler = GenerationHarness(
        policies,
        [
            _sample_response([7], 'abort', [1, 2, 7]),
            _sample_response([8], 'stop', [1, 2, 8]),
        ],
    )
    generated = asyncio.run(
        VLLMSamplerTQ._generate_sample(
            sampler,
            context,
            {'input_ids': [1, 2], 'labels': [-100, -100]},
            SamplingParams(max_tokens=4, logprobs=1),
            multi_modal_data=None,
            logprobs_only=False,
            allow_partial_rollout=False,
        ))

    assert sampler.calls == [([1, 2], 4, 3), ([1, 2], 4, 4)]
    assert generated.response.sequences[0].tokens == [8]
    assert [policy.version for policy in generated.policies] == [4]
    assert generated.retry_count == 1
    assert generated.was_aborted
    assert not generated.resumed_partial_output


def test_aborted_generation_continues_from_partial_tokens_when_enabled():
    context = _context()
    policies = [
        RolloutPolicy(context.key, context.adapter_name, 3, 'adapter-v3'),
        RolloutPolicy(context.key, context.adapter_name, 4, 'adapter-v4'),
    ]
    sampler = GenerationHarness(
        policies,
        [
            _sample_response([7], 'abort', [1, 2, 7]),
            _sample_response([8], 'stop', [1, 2, 7, 8]),
        ],
    )
    generated = asyncio.run(
        VLLMSamplerTQ._generate_sample(
            sampler,
            context,
            {'input_ids': [1, 2], 'labels': [-100, -100]},
            SamplingParams(max_tokens=4, logprobs=1),
            multi_modal_data=None,
            logprobs_only=False,
            allow_partial_rollout=True,
        ))

    assert sampler.calls == [([1, 2], 4, 3), ([1, 2, 7], 3, 4)]
    assert generated.response.sequences[0].tokens == [7, 8]
    assert [policy.version for policy in generated.policies] == [3, 4]
    assert generated.initial_policy.version == 3
    assert generated.final_policy.version == 4
    assert generated.retry_count == 1
    assert generated.was_aborted
    assert generated.resumed_partial_output
