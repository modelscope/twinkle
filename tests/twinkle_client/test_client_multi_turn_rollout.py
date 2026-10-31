# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import replace

import pytest

from twinkle.data_format.sampling import SampledSequence
from twinkle.protocol.types.sampler import SampledSequenceModel, SampleResponseModel
from twinkle.template.contract import (
    CompletionEncoding,
    DecodedResponse,
    MediaBundle,
    PromptEncoding,
    RolloutState,
    refresh,
    splice,
    splice_state,
)
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_client.rollout.multi_turn import ClientMultiTurnRollout


class _Template:
    truncation_strategy = 'delete'

    def __init__(self, *, multimodal=False, has_media=None, engine_delta=0, bridge_engine_delta=0):
        self.is_multimodal = multimodal
        self.has_media = multimodal if has_media is None else has_media
        self.engine_delta = engine_delta
        self.bridge_engine_delta = bridge_engine_delta

    def encode_prompt(self, trajectory, *, backend='vllm', add_generation_prompt=True):
        feature = {'input_ids': [1]}
        state = RolloutState(
            feature=feature,
            engine_input_ids=(1 + self.engine_delta, ),
            messages=tuple(trajectory.get('messages') or ()),
            tools=tuple(trajectory.get('tools') or ()),
            reasoning_started=False,
            media=MediaBundle(images=('image.png', )) if self.has_media else MediaBundle(),
        )
        return PromptEncoding(state)

    def append_completion(self, state, tokens, *, appended_as='completion', tool_calls=None):
        next_state = splice_state(state, tokens, tokens, scored=True, sampled=True)
        calls = ({'id': 'call-1', 'type': 'function',
                  'function': {'name': 'tool', 'arguments': {}}}, ) if tokens == [10] else ()
        response = DecodedResponse(content=f'answer-{tokens[0]}', tool_calls=calls, reasoning_started=False)
        message = {'role': 'assistant', 'content': response.content}
        if calls:
            message['tool_calls'] = list(calls)
        return CompletionEncoding(replace(next_state, messages=state.messages + (message, )), response)

    def render_delta(self, messages_before, appended, *, engine=False, **kwargs):
        return [90 + (self.bridge_engine_delta if engine else 0)]

    def finalize_appended_features(self, rows):
        return [refresh(row) for row in rows]

    def parse_tool_call(self, decoded):
        return []

    def tool_call_errors(self, decoded):
        return []

    @staticmethod
    def _reasoning_started(input_ids):
        return False


class _Sampler:

    def __init__(self, scripts=((10, 'stop'), (20, 'stop'))):
        self.scripts = list(scripts)
        self.calls = []
        self.policies = []
        self.params = []

    def sample(self, features, sampling_params=None, adapter_name='', adapter_uri=None, num_samples=1):
        self.calls.append(features)
        self.policies.append((adapter_name, adapter_uri, num_samples))
        self.params.append(sampling_params)
        token, stop_reason = self.scripts.pop(0)
        responses = []
        for feature in features:
            updated = splice(feature, [token], scored=True, sampled=True)
            sequence = SampledSequenceModel(
                stop_reason=stop_reason,
                tokens=[token],
                logprobs=[[(token, -0.1)]],
                decoded=f'answer-{token}',
                new_input_feature=updated,
            )
            responses.append(SampleResponseModel(sequences=[sequence]))
        return responses


class _ToolManager:

    @staticmethod
    def tool_infos():
        return [{'type': 'function', 'function': {'name': 'tool', 'parameters': {}}}]

    @staticmethod
    def call_many(tool_calls):
        return ['observed'] * len(tool_calls)


def _trajectory():
    return {'messages': [{'role': 'user', 'content': 'question'}]}


def test_text_http_multi_turn_preserves_tokens_messages_and_logprobs():
    sampler = _Sampler()
    rollout = ClientMultiTurnRollout(sampler, _Template(), tool_manager=_ToolManager(), max_turns=2)

    result = rollout([_trajectory()])[0]

    assert len(sampler.calls) == 2
    assert result['input_ids'] == [1, 10, 90, 20]
    assert result['logprobs'] == [[(10, -0.1)], [(20, -0.1)]]
    assert [message['role'] for message in result['messages']] == ['user', 'assistant', 'tool', 'assistant']
    assert result['turns'] == 2
    assert result['truncated'] is False


def test_core_and_wire_characterization_share_scripted_trajectory():
    def wire_callback(sampler, api, sampling_params, *, input_feature, adapter_kwargs, **kwargs):
        response = sampler.sample([input_feature], sampling_params=sampling_params)[0]
        sequence = response.sequences[0]
        return SampledSequence(
            stop_reason=sequence.stop_reason,
            tokens=sequence.tokens,
            logprobs=sequence.logprobs,
            decoded=sequence.decoded,
            new_input_feature=sequence.new_input_feature,
        )

    wire_sampler = _Sampler()
    core = MultiTurnRollout(
        sampler=None,
        template=_Template(),
        tool_manager=_ToolManager(),
        max_turns=2,
        response_callback=wire_callback,
    )
    core_result = core([_trajectory()], response_callback=lambda _sampler, _api, params, **kwargs: wire_callback(
        wire_sampler, _api, params, **kwargs))[0]
    http_result = ClientMultiTurnRollout(
        _Sampler(), _Template(), tool_manager=_ToolManager(), max_turns=2)([_trajectory()])[0]

    for key in ('input_ids', 'labels', 'completion_mask', 'messages', 'tools', 'logprobs', 'rollout_logprobs'):
        assert core_result[key] == http_result[key]
    assert core_result['stop_reason'] == http_result['stop_reason'] == 'stop'


def test_http_multi_turn_complete_trajectory_golden():
    result = ClientMultiTurnRollout(
        _Sampler(), _Template(), tool_manager=_ToolManager(), max_turns=2)([_trajectory()])[0]
    assert result == {
        'input_ids': [1, 10, 90, 20],
        'labels': [10, -100, 20, -100],
        'completion_mask': [1, 0, 1, 0],
        'attention_mask': [1, 1, 1, 1],
        'length': 4,
        'messages': [
            {'role': 'user', 'content': 'question'},
            {'role': 'assistant', 'content': 'answer-10', 'tool_calls': [{
                'id': 'call-1', 'type': 'function', 'function': {'name': 'tool', 'arguments': {}}
            }]},
            {'role': 'tool', 'content': 'observed', 'tool_call_id': 'call-1', 'name': 'tool'},
            {'role': 'assistant', 'content': 'answer-20'},
        ],
        'tools': [{'type': 'function', 'function': {'name': 'tool', 'parameters': {}}}],
        'logprobs': [[(10, -0.1)], [(20, -0.1)]],
        'response_token_ids': [10, 20],
        'rollout_logprobs': [-0.1, -0.1],
        'turns': 2,
        'stop_reason': 'stop',
        'truncated': False,
        'stuck_stop': False,
        'tool_stop': None,
        'followups': 0,
    }


@pytest.mark.parametrize('template,match', [
    (_Template(multimodal=True), 'state_id.*media'),
    (_Template(engine_delta=1), 'state_id.*engine ids differ'),
])
def test_http_multi_turn_rejects_state_that_cannot_be_resubmitted(template, match):
    sampler = _Sampler()
    with pytest.raises(NotImplementedError, match=match):
        ClientMultiTurnRollout(sampler, template)([_trajectory()])
    assert sampler.calls == []


def test_multimodal_capability_template_allows_text_only_trajectory():
    sampler = _Sampler(((30, 'stop'), ))
    result = ClientMultiTurnRollout(sampler, _Template(multimodal=True, has_media=False))([_trajectory()])[0]
    assert result['input_ids'] == [1, 30]
    assert result['turns'] == 1


def test_http_multi_turn_rechecks_feature_engine_equality_after_bridge():
    sampler = _Sampler()
    rollout = ClientMultiTurnRollout(
        sampler, _Template(bridge_engine_delta=1), tool_manager=_ToolManager(), max_turns=2)
    with pytest.raises(NotImplementedError, match='state_id.*engine ids differ'):
        rollout([_trajectory()])
    assert len(sampler.calls) == 1


def test_http_multi_turn_requires_new_input_feature():
    class BrokenSampler:

        @staticmethod
        def sample(features, sampling_params=None, **kwargs):
            sequence = SampledSequenceModel(
                stop_reason='stop', tokens=[10], logprobs=[[(10, -0.1)]], decoded='x')
            return [SampleResponseModel(sequences=[sequence])]

    with pytest.raises(RuntimeError, match='new_input_feature.*trajectory 0'):
        ClientMultiTurnRollout(BrokenSampler(), _Template())([_trajectory()])


def test_http_multi_turn_requires_tool_manager_only_when_called():
    with pytest.raises(ValueError, match='tool_calls.*ToolManager'):
        ClientMultiTurnRollout(_Sampler(), _Template(), max_turns=2)([_trajectory()])


def test_http_multi_turn_marks_length_stop_as_truncated():
    result = ClientMultiTurnRollout(_Sampler(((10, 'length'), )), _Template())([_trajectory()])[0]
    assert result['turns'] == 1
    assert result['stop_reason'] == 'length'
    assert result['truncated'] is True


def test_http_multi_turn_preserves_batch_output_order():
    trajectories = [dict(_trajectory(), sample_id=index) for index in range(4)]
    results = ClientMultiTurnRollout(_Sampler(((30, 'stop'), ) * 4), _Template())(trajectories)
    assert [result['sample_id'] for result in results] == [0, 1, 2, 3]


@pytest.mark.parametrize('stop_reason', ['stop', 'length', 'abort', 'error'])
def test_http_multi_turn_stop_reason_stays_in_public_range(stop_reason):
    result = ClientMultiTurnRollout(_Sampler(((30, stop_reason), )), _Template())([_trajectory()])[0]
    assert result['stop_reason'] in {'stop', 'length', 'abort', 'error', 'max_turns'}


@pytest.mark.parametrize('scripts', [
    ((10, 'stop'), (20, 'stop')),
    ((10, 'stop'), (21, 'length')),
])
def test_http_multi_turn_scripts_keep_logprobs_aligned_with_policy_labels(scripts):
    result = ClientMultiTurnRollout(
        _Sampler(scripts), _Template(), tool_manager=_ToolManager(), max_turns=2)([_trajectory()])[0]
    policy_positions = sum(
        label != -100 and mask
        for label, mask in zip(result['labels'], result['completion_mask'])
    )
    assert len(result['logprobs']) == policy_positions == result['turns']


def test_http_multi_turn_max_turns_one_truncates_before_tool_dispatch():
    class RecordingTools(_ToolManager):
        calls = 0

        @classmethod
        def call_many(cls, tool_calls):
            cls.calls += 1
            return super().call_many(tool_calls)

    result = ClientMultiTurnRollout(
        _Sampler(((10, 'stop'), )), _Template(), tool_manager=RecordingTools(), max_turns=1)([_trajectory()])[0]
    assert result['turns'] == 1
    assert result['stop_reason'] == 'stop'
    assert result['tool_stop'] == 'max_turns'
    assert result['truncated'] is True
    assert RecordingTools.calls == 0


def test_http_multi_turn_propagates_network_error_unchanged():
    class NetworkError(Exception):
        pass

    sentinel = NetworkError('connection reset')

    class FailingSampler:

        @staticmethod
        def sample(features, sampling_params=None, **kwargs):
            raise sentinel

    with pytest.raises(NetworkError) as error:
        ClientMultiTurnRollout(FailingSampler(), _Template())([_trajectory()])
    assert error.value is sentinel


def test_http_multi_turn_validates_public_arguments():
    with pytest.raises(ValueError, match='Template'):
        ClientMultiTurnRollout(object(), None)
    with pytest.raises(ValueError, match='max_turns'):
        ClientMultiTurnRollout(object(), _Template(), max_turns=0)


def test_policy_and_sampling_params_are_snapshotted_per_call():
    sampler = _Sampler(((30, 'stop'), (31, 'stop')))
    rollout = ClientMultiTurnRollout(
        sampler, _Template(), adapter_name='trained', adapter_uri='twinkle://policy-0')
    rollout([_trajectory()])
    override = {'temperature': 0.25}
    rollout([_trajectory()], adapter_name='base', adapter_uri=None, sampling_params=override)

    assert sampler.policies == [('trained', 'twinkle://policy-0', 1), ('base', None, 1)]
    assert sampler.params[0].temperature == 1.0
    assert sampler.params[1].temperature == 0.25
    assert override == {'temperature': 0.25}


def test_invalid_call_options_are_rejected_before_http():
    sampler = _Sampler()
    rollout = ClientMultiTurnRollout(sampler, _Template())
    with pytest.raises(ValueError, match='unknown.*adapter_path'):
        rollout([_trajectory()], adapter_path='not-a-cs-option')
    with pytest.raises(ValueError, match='num_samples=1'):
        rollout([_trajectory()], sampling_params={'num_samples': 2})
    with pytest.raises(ValueError, match='tool_manager list length'):
        rollout([_trajectory()], tool_manager=[])
    assert sampler.calls == []
    assert rollout([]) == []


def test_sync_and_async_calls_return_the_same_core_trajectory():
    sync_rollout = ClientMultiTurnRollout(_Sampler(), _Template(), tool_manager=_ToolManager(), max_turns=2)
    async_rollout = ClientMultiTurnRollout(_Sampler(), _Template(), tool_manager=_ToolManager(), max_turns=2)

    expected = sync_rollout([_trajectory()])

    async def run():
        return await async_rollout.acall([_trajectory()])

    assert asyncio.run(run()) == expected


def test_async_cancellation_drains_http_and_does_not_poison_later_calls():
    started = threading.Event()
    release = threading.Event()

    class BlockingSampler(_Sampler):
        def sample(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().sample(*args, **kwargs)

    sampler = BlockingSampler(((10, 'stop'), (20, 'stop')))
    rollout = ClientMultiTurnRollout(sampler, _Template())

    async def run():
        task = asyncio.create_task(rollout.acall([_trajectory()]))
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        return await rollout.acall([_trajectory()])

    result = asyncio.run(run())
    assert result[0]['input_ids'] == [1, 20]
    assert len(sampler.calls) == 2


def test_duplicate_env_and_bare_harness_are_rejected_before_side_effects():
    env = object()

    class EnvManager(_ToolManager):
        @staticmethod
        def bound_envs():
            return (env, )

    class Harness:
        calls = 0

        def before_generate(self, trajectory):
            type(self).calls += 1
            return trajectory

    sampler = _Sampler()
    harness = Harness()
    rollout = ClientMultiTurnRollout(sampler, _Template())
    with pytest.raises(ValueError, match='same Env'):
        rollout([_trajectory(), _trajectory()], tool_manager=[EnvManager(), EnvManager()])
    with pytest.raises(ValueError, match='same bare harness'):
        rollout([_trajectory(), _trajectory()], harness=[harness, harness])
    assert sampler.calls == []
    assert harness.calls == 0


def test_active_env_borrow_is_released_only_after_runner_finishes():
    started = threading.Event()
    release = threading.Event()

    class BlockingSampler(_Sampler):
        def sample(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().sample(*args, **kwargs)

    class Env:
        close_calls = 0

        def close(self):
            self.close_calls += 1

    class EnvManager(_ToolManager):
        def __init__(self, env):
            self.env = env

        def bound_envs(self):
            return (self.env, )

    env = Env()
    manager = EnvManager(env)
    sampler = BlockingSampler(((10, 'stop'), (20, 'stop'), (30, 'stop')))
    rollout = ClientMultiTurnRollout(sampler, _Template(), max_turns=2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(rollout, [_trajectory()], tool_manager=manager)
        assert started.wait(timeout=5)
        with pytest.raises(ValueError, match='still borrowed'):
            rollout([_trajectory()], tool_manager=manager)
        release.set()
        assert first.result(timeout=5)[0]['turns'] == 2
    result = rollout([_trajectory()], tool_manager=manager)
    assert result[0]['turns'] == 1
    assert env.close_calls == 0


def test_active_bare_harness_borrow_covers_concurrent_single_episode_calls():
    started = threading.Event()
    release = threading.Event()

    class BlockingSampler(_Sampler):
        def sample(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().sample(*args, **kwargs)

    class Harness:
        def before_generate(self, trajectory):
            return trajectory

        @staticmethod
        def after_generate(context, decoded, tool_calls):
            return {'messages': context['messages'] + [{'role': 'assistant', 'content': decoded}]}

    sampler = BlockingSampler(((30, 'stop'), (40, 'stop')))
    rollout = ClientMultiTurnRollout(sampler, _Template(), harness=Harness())
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(rollout, [_trajectory()])
        assert started.wait(timeout=5)
        with pytest.raises(ValueError, match='still borrowed'):
            rollout([_trajectory()])
        release.set()
        assert first.result(timeout=5)[0]['input_ids'] == [1, 30]


def test_independent_episodes_do_not_wait_for_a_slow_tool_round():
    started = threading.Event()
    release = threading.Event()
    fast_finished = threading.Event()

    class KeyedTemplate(_Template):
        def encode_prompt(self, trajectory, *, backend='vllm', add_generation_prompt=True):
            encoding = super().encode_prompt(
                trajectory, backend=backend, add_generation_prompt=add_generation_prompt)
            token = 1 if trajectory['messages'][0]['content'] == 'slow' else 2
            state = replace(
                encoding.state,
                feature={'input_ids': [token]},
                engine_input_ids=(token, ),
            )
            return PromptEncoding(state)

    class KeyedSampler:
        def __init__(self):
            self.calls = {1: 0, 2: 0}

        def sample(self, features, sampling_params=None, **kwargs):
            feature = features[0]
            key = feature['input_ids'][0]
            self.calls[key] += 1
            if key == 1 and self.calls[key] == 1:
                started.set()
                assert release.wait(timeout=5)
            token = 10 if self.calls[key] == 1 else 20
            updated = splice(feature, [token], scored=True, sampled=True)
            sequence = SampledSequenceModel(
                stop_reason='stop',
                tokens=[token],
                logprobs=[[(token, -0.1)]],
                decoded=f'answer-{token}',
                new_input_feature=updated,
            )
            if key == 2 and self.calls[key] == 2:
                fast_finished.set()
            return [SampleResponseModel(sequences=[sequence])]

    rollout = ClientMultiTurnRollout(
        KeyedSampler(), KeyedTemplate(), tool_manager=_ToolManager(), max_turns=2, concurrency=2)
    trajectories = [
        {'messages': [{'role': 'user', 'content': 'slow'}]},
        {'messages': [{'role': 'user', 'content': 'fast'}]},
    ]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rollout, trajectories)
        assert started.wait(timeout=5)
        assert fast_finished.wait(timeout=5)
        release.set()
        results = future.result(timeout=5)
    assert [result['input_ids'][0] for result in results] == [1, 2]
    assert all(result['turns'] == 2 for result in results)


def test_first_episode_error_is_preserved_after_peer_drain():
    peer_started = threading.Event()
    failing = threading.Event()
    release = threading.Event()
    sentinel = RuntimeError('sample failed')

    class KeyedTemplate(_Template):
        def encode_prompt(self, trajectory, *, backend='vllm', add_generation_prompt=True):
            encoding = super().encode_prompt(
                trajectory, backend=backend, add_generation_prompt=add_generation_prompt)
            token = 1 if trajectory['messages'][0]['content'] == 'slow' else 2
            state = replace(encoding.state, feature={'input_ids': [token]}, engine_input_ids=(token, ))
            return PromptEncoding(state)

    class FailingSampler:
        @staticmethod
        def sample(features, sampling_params=None, **kwargs):
            key = features[0]['input_ids'][0]
            if key == 2:
                assert peer_started.wait(timeout=5)
                failing.set()
                raise sentinel
            peer_started.set()
            assert failing.wait(timeout=5)
            assert release.wait(timeout=5)
            feature = features[0]
            updated = splice(feature, [10], scored=True, sampled=True)
            sequence = SampledSequenceModel(
                stop_reason='stop',
                tokens=[10],
                logprobs=[[(10, -0.1)]],
                decoded='late',
                new_input_feature=updated,
            )
            return [SampleResponseModel(sequences=[sequence])]

    rollout = ClientMultiTurnRollout(FailingSampler(), KeyedTemplate(), concurrency=2)
    trajectories = [
        {'messages': [{'role': 'user', 'content': 'slow'}]},
        {'messages': [{'role': 'user', 'content': 'fail'}]},
    ]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rollout, trajectories)
        assert failing.wait(timeout=5)
        assert not future.done()
        release.set()
        with pytest.raises(RuntimeError) as error:
            future.result(timeout=5)
    assert error.value is sentinel


def test_cancel_during_followup_prevents_another_generation():
    followup_started = threading.Event()
    release = threading.Event()
    sampler = _Sampler(((30, 'stop'), (40, 'stop')))

    def followup(trajectory, count):
        followup_started.set()
        assert release.wait(timeout=5)
        return 'continue'

    rollout = ClientMultiTurnRollout(sampler, _Template(), followup_fn=followup)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(rollout, [_trajectory()])
        assert followup_started.wait(timeout=5)
        rollout.cancel()
        release.set()
        with pytest.raises(CancelledError):
            future.result(timeout=5)
    assert len(sampler.calls) == 1


def test_sampler_response_cardinality_and_logprob_alignment_are_validated():
    class EmptySampler:
        @staticmethod
        def sample(features, sampling_params=None, **kwargs):
            return []

    with pytest.raises(RuntimeError, match='expected one'):
        ClientMultiTurnRollout(EmptySampler(), _Template())([_trajectory()])

    class MisalignedSampler:
        @staticmethod
        def sample(features, sampling_params=None, **kwargs):
            updated = splice(features[0], [10, 11], scored=True, sampled=True)
            sequence = SampledSequenceModel(
                stop_reason='stop',
                tokens=[10, 11],
                logprobs=[[(10, -0.1)]],
                decoded='misaligned',
                new_input_feature=updated,
            )
            return [SampleResponseModel(sequences=[sequence])]

    with pytest.raises(RuntimeError, match='logprob'):
        ClientMultiTurnRollout(MisalignedSampler(), _Template())([_trajectory()])


def test_concurrent_policy_capture_and_completed_call_cancel_race():
    slow_started = threading.Event()
    release = threading.Event()
    policies = []

    class KeyedTemplate(_Template):
        def encode_prompt(self, trajectory, *, backend='vllm', add_generation_prompt=True):
            encoding = super().encode_prompt(
                trajectory, backend=backend, add_generation_prompt=add_generation_prompt)
            token = {'slow': 1, 'fast': 2, 'later': 3}[trajectory['messages'][0]['content']]
            state = replace(encoding.state, feature={'input_ids': [token]}, engine_input_ids=(token, ))
            return PromptEncoding(state)

    class PolicySampler:
        @staticmethod
        def sample(features, sampling_params=None, adapter_name='', adapter_uri=None, num_samples=1):
            policies.append((features[0]['input_ids'][0], adapter_name, adapter_uri))
            if adapter_uri == 'slow':
                slow_started.set()
                assert release.wait(timeout=5)
            updated = splice(features[0], [30], scored=True, sampled=True)
            sequence = SampledSequenceModel(
                stop_reason='stop',
                tokens=[30],
                logprobs=[[(30, -0.1)]],
                decoded='done',
                new_input_feature=updated,
            )
            return [SampleResponseModel(sequences=[sequence])]

    def trajectory(name):
        return {'messages': [{'role': 'user', 'content': name}]}

    rollout = ClientMultiTurnRollout(PolicySampler(), KeyedTemplate())
    with ThreadPoolExecutor(max_workers=1) as pool:
        slow = pool.submit(rollout, [trajectory('slow')], adapter_name='policy', adapter_uri='slow')
        assert slow_started.wait(timeout=5)
        fast = rollout([trajectory('fast')], adapter_name='policy', adapter_uri='fast')
        assert fast[0]['input_ids'] == [2, 30]
        rollout.cancel()
        release.set()
        with pytest.raises(CancelledError):
            slow.result(timeout=5)
    later = rollout([trajectory('later')], adapter_name='base', adapter_uri=None)
    assert later[0]['input_ids'] == [3, 30]
    assert policies == [(1, 'policy', 'slow'), (2, 'policy', 'fast'), (3, 'base', None)]
