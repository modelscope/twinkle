# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import gc
import threading
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
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


def test_sync_repeated_interrupts_keep_draining_until_core_finishes():
    started = threading.Event()
    release = threading.Event()

    class BlockingSampler(_Sampler):
        def sample(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().sample(*args, **kwargs)

    class InterruptingFuture:

        def __init__(self, future, interrupts):
            self._future = future
            self._interrupts = interrupts

        def done(self):
            return self._future.done()

        def result(self):
            if self._interrupts:
                self._interrupts -= 1
                assert started.wait(timeout=5)
                raise KeyboardInterrupt
            return self._future.result()

    sampler = BlockingSampler(((10, 'stop'), (20, 'stop')))
    rollout = ClientMultiTurnRollout(sampler, _Template(), tool_manager=_ToolManager(), max_turns=2)
    calls = []
    original_start = rollout._start

    def start(trajectories, kwargs):
        call = original_start(trajectories, kwargs)
        calls.append(call)
        call.future = InterruptingFuture(call.future, interrupts=2)
        return call

    rollout._start = start
    outcome = {}

    def run():
        try:
            rollout([_trajectory()])
        except BaseException as error:
            outcome['error'] = error

    worker = threading.Thread(target=run)
    worker.start()
    assert started.wait(timeout=5)
    worker.join(timeout=0.3)
    assert worker.is_alive(), 'sync call returned before the in-flight sample finished'
    release.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert isinstance(outcome['error'], KeyboardInterrupt)
    assert calls[0].future._future.done()
    assert len(sampler.calls) == 1


@pytest.mark.parametrize('control_error', [KeyboardInterrupt(), SystemExit(2)])
def test_sync_completed_future_control_error_is_not_retried_forever(control_error, monkeypatch):
    from types import SimpleNamespace
    from twinkle_agentic.rollout.base import _RolloutExecution

    future = Future()
    future.set_exception(control_error)
    rollout = ClientMultiTurnRollout(_Sampler(), _Template())
    execution = _RolloutExecution()
    monkeypatch.setattr(rollout, '_start', lambda *args: SimpleNamespace(future=future, execution=execution))
    with pytest.raises(type(control_error)) as error:
        rollout([_trajectory()])
    assert error.value is control_error
    assert execution.cancelled


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


def _wait_for_handle(rollout, handle):
    ticker = threading.Event()
    for _ in range(1000):
        completed = rollout.poll([handle])
        if completed:
            return completed[0]
        ticker.wait(0.005)
    raise AssertionError('episode did not finish')


def test_admit_poll_collect_matches_batch_and_preserves_admission_snapshot():
    sampler = _Sampler()
    rollout = ClientMultiTurnRollout(sampler, _Template(), tool_manager=_ToolManager(), max_turns=2)
    handle = rollout.admit(
        _trajectory()['messages'],
        3,
        prompt_idx=17,
        prompt_extras={'task_id': 'task-1'},
        adapter_path='twinkle://policy-0',
        policy_version=4,
    )

    _wait_for_handle(rollout, handle)
    sample = rollout.collect(handle)

    assert sample.prompt_id == '17'
    assert sample.policy_version == 4
    assert sample.version_span == 0
    assert sample.prompt == _trajectory()['messages']
    assert sample.extra == {
        'task_id': 'task-1',
        'trajectory_idx': 3,
        'adapter_uri': 'twinkle://policy-0',
        'submission_id': handle.submission_id,
    }
    assert sample.encoded['input_ids'] == [1, 10, 90, 20]
    assert sample.response_token_ids == [10, 20]
    assert sample.rollout_logprobs == [-0.1, -0.1]
    assert sampler.policies == [('', 'twinkle://policy-0', 1), ('', 'twinkle://policy-0', 1)]
    assert sampler.params[0].logprobs == 1
    assert rollout.stop(handle.submission_id) is True
    with pytest.raises(RuntimeError, match='already consumed'):
        rollout.collect(handle)


def test_admit_and_poll_do_not_wait_for_sampler_or_idle_confirmation():
    sample_started = threading.Event()
    release_sample = threading.Event()
    idle_started = threading.Event()
    release_idle = threading.Event()

    class BlockingSampler(_Sampler):
        def sample(self, *args, **kwargs):
            sample_started.set()
            assert release_sample.wait(timeout=5)
            return super().sample(*args, **kwargs)

    class Env:
        def ensure_idle(self):
            idle_started.set()
            assert release_idle.wait(timeout=5)

    class EnvManager(_ToolManager):
        @staticmethod
        def bound_envs():
            return (env, )

    env = Env()
    rollout = ClientMultiTurnRollout(BlockingSampler(((30, 'stop'), )), _Template())
    handle = rollout.admit(_trajectory()['messages'], tool_manager=EnvManager())
    assert sample_started.wait(timeout=5)
    assert rollout.poll([handle]) == []
    with pytest.raises(RuntimeError, match='not ready'):
        rollout.collect(handle)
    release_sample.set()
    assert idle_started.wait(timeout=5)
    assert rollout.poll([handle]) == [], 'Future completed before runner finished ensure_idle'
    release_idle.set()
    _wait_for_handle(rollout, handle)
    assert rollout.collect(handle).response_token_ids == [30]


def test_failed_collect_preserves_business_error_and_settled_stop_is_idempotent():
    class EmptySampler:
        @staticmethod
        def sample(features, sampling_params=None, **kwargs):
            return []

    rollout = ClientMultiTurnRollout(EmptySampler(), _Template())
    handle = rollout.admit(_trajectory()['messages'])
    _wait_for_handle(rollout, handle)
    with pytest.raises(RuntimeError, match='expected one'):
        rollout.collect(handle)
    assert rollout.stop(handle.submission_id) is True
    assert rollout.stop(handle.submission_id) is True


def test_uncertain_generation_quarantines_resources_and_caches_stop_error(monkeypatch):
    sentinel = ConnectionError('client wait expired')
    close_calls = []

    class FailingSampler:
        @staticmethod
        def sample(features, sampling_params=None, **kwargs):
            raise sentinel

    class Env:
        def ensure_idle(self):
            return None

        def close(self):
            close_calls.append('close')

    class EnvManager(_ToolManager):
        @staticmethod
        def bound_envs():
            return (env, )

    env = Env()
    rollout = ClientMultiTurnRollout(FailingSampler(), _Template())
    handle = rollout.admit(_trajectory()['messages'], tool_manager=EnvManager())
    _wait_for_handle(rollout, handle)
    with pytest.raises(ConnectionError) as business_error:
        rollout.collect(handle)
    assert business_error.value is sentinel
    with pytest.raises(RuntimeError, match='unconfirmed') as first:
        rollout.stop(handle.submission_id)
    monkeypatch.setattr(rollout, '_drain_entry', lambda entry: (_ for _ in ()).throw(
        AssertionError('cached uncertainty must not drain again')))
    with pytest.raises(RuntimeError, match='unconfirmed') as second:
        rollout.stop(handle.submission_id)
    assert second.value is first.value
    with pytest.raises(ValueError, match='still borrowed'):
        rollout.admit(_trajectory()['messages'], tool_manager=EnvManager())
    assert close_calls == []


def test_settlement_credential_tracks_handle_lifetime():
    rollout = ClientMultiTurnRollout(_Sampler(((30, 'stop'), )), _Template())
    handle = rollout.admit(_trajectory()['messages'])
    _wait_for_handle(rollout, handle)
    rollout.collect(handle)
    submission_id = handle.submission_id
    assert rollout.stop(submission_id) is True
    del handle
    gc.collect()
    assert rollout.stop(submission_id) is False


def test_handle_tampering_cross_instance_use_and_unknown_stop_are_rejected():
    rollout = ClientMultiTurnRollout(_Sampler(((30, 'stop'), (31, 'stop'))), _Template())
    handle = rollout.admit(_trajectory()['messages'], policy_version=2)
    other = ClientMultiTurnRollout(_Sampler(), _Template())
    with pytest.raises(RuntimeError, match='unknown'):
        other.poll([handle])
    handle.policy_version = 99
    with pytest.raises(RuntimeError, match='metadata was modified'):
        rollout.poll([handle])
    handle.policy_version = 2
    forged = replace(handle, submission_id='forged')
    with pytest.raises(RuntimeError, match='unknown'):
        rollout.poll([forged])
    assert rollout.stop('unknown-submission') is False
    _wait_for_handle(rollout, handle)
    rollout.collect(handle)


def test_stop_cancels_the_next_turn_and_waits_for_the_inflight_sample():
    started = threading.Event()
    release = threading.Event()

    class BlockingSampler(_Sampler):
        def sample(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().sample(*args, **kwargs)

    sampler = BlockingSampler()
    rollout = ClientMultiTurnRollout(sampler, _Template(), max_turns=2)
    handle = rollout.admit(_trajectory()['messages'])
    assert started.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stopped = pool.submit(rollout.stop, handle.submission_id)
        assert not stopped.done()
        release.set()
        assert stopped.result(timeout=5) is True
    assert len(sampler.calls) == 1


def test_concurrent_stop_shares_one_actual_drain(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    class BlockingSampler(_Sampler):
        def sample(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().sample(*args, **kwargs)

    rollout = ClientMultiTurnRollout(BlockingSampler(((30, 'stop'), )), _Template())
    handle = rollout.admit(_trajectory()['messages'])
    assert started.wait(timeout=5)
    drain_calls = []
    original_drain = rollout._drain_entry

    def drain(entry):
        drain_calls.append(entry.handle.submission_id)
        return original_drain(entry)

    monkeypatch.setattr(rollout, '_drain_entry', drain)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(rollout.stop, handle.submission_id)
        second = pool.submit(rollout.stop, handle.submission_id)
        release.set()
        assert first.result(timeout=5) is True
        assert second.result(timeout=5) is True
    assert drain_calls == [handle.submission_id]


def test_collect_loses_to_an_inprogress_stop_without_double_consumption():
    started = threading.Event()
    release = threading.Event()

    class BlockingSampler(_Sampler):
        def sample(self, *args, **kwargs):
            started.set()
            assert release.wait(timeout=5)
            return super().sample(*args, **kwargs)

    rollout = ClientMultiTurnRollout(BlockingSampler(((30, 'stop'), )), _Template())
    handle = rollout.admit(_trajectory()['messages'])
    assert started.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=1) as pool:
        stopped = pool.submit(rollout.stop, handle.submission_id)
        with pytest.raises(RuntimeError, match='being stopped'):
            rollout.collect(handle)
        release.set()
        assert stopped.result(timeout=5) is True
    assert rollout.stop(handle.submission_id) is True


def test_idle_failure_is_separate_from_generation_and_remains_quarantined():
    sentinel = RuntimeError('provider cannot confirm command exit')

    class Env:
        def ensure_idle(self):
            raise sentinel

    class EnvManager(_ToolManager):
        @staticmethod
        def bound_envs():
            return (env, )

    env = Env()
    rollout = ClientMultiTurnRollout(_Sampler(((30, 'stop'), )), _Template())
    handle = rollout.admit(_trajectory()['messages'], tool_manager=EnvManager())
    _wait_for_handle(rollout, handle)
    with pytest.raises(RuntimeError, match='unconfirmed') as error:
        rollout.collect(handle)
    assert error.value.__cause__ is sentinel
    with pytest.raises(RuntimeError, match='unconfirmed'):
        rollout.stop(handle.submission_id)
    with pytest.raises(ValueError, match='still borrowed'):
        rollout.admit(_trajectory()['messages'], tool_manager=EnvManager())


def test_admit_threads_tool_schema_into_core():
    class ToolCaptureTemplate(_Template):
        seen = None

        def encode_prompt(self, trajectory, **kwargs):
            self.seen = trajectory.get('tools')
            return super().encode_prompt(trajectory, **kwargs)

    template = ToolCaptureTemplate()
    rollout = ClientMultiTurnRollout(_Sampler(), template, tool_manager=_ToolManager(), max_turns=2)
    schema = _ToolManager.tool_infos()
    handle = rollout.admit(_trajectory()['messages'], prompt_extras={'tools': schema})
    _wait_for_handle(rollout, handle)
    rollout.collect(handle)
    assert template.seen == schema


def test_admit_rejects_conflicting_configuration_before_side_effects():
    sampler = _Sampler()
    rollout = ClientMultiTurnRollout(sampler, _Template())
    with pytest.raises(ValueError, match='num_samples=1'):
        rollout.admit(_trajectory()['messages'], sampling_params={'num_samples': 2})
    with pytest.raises(ValueError, match='reserved fields'):
        rollout.admit(_trajectory()['messages'], prompt_extras={'adapter_uri': 'other'})
    assert sampler.calls == []


def test_streaming_scope_attempts_all_handles_before_abort(monkeypatch):
    rollout = ClientMultiTurnRollout(_Sampler(((30, 'stop'),) * 4), _Template())
    handles = [rollout.admit(_trajectory()['messages']) for _ in range(4)]
    for handle in handles:
        _wait_for_handle(rollout, handle)
    events = []
    original = rollout.stop

    def stop(submission_id):
        events.append(submission_id)
        if submission_id == handles[0].submission_id:
            raise RuntimeError('first cleanup failed')
        return original(submission_id)

    monkeypatch.setattr(rollout, 'stop', stop)
    driver = type('Driver', (), {'abort': lambda self: events.append('abort')})()
    with pytest.raises(BaseExceptionGroup, match='streaming cleanup'):
        with rollout.streaming(driver):
            pass
    assert events == [handle.submission_id for handle in handles] + ['abort']
    with pytest.raises(RuntimeError, match='admission is closed'):
        rollout.admit(_trajectory()['messages'])


def test_streaming_scope_preserves_primary_and_unknown_resources():
    class FailingSampler:
        def sample(self, *args, **kwargs):
            raise ConnectionError('unknown remote generation')

    rollout = ClientMultiTurnRollout(FailingSampler(), _Template())
    handle = rollout.admit(_trajectory()['messages'])
    _wait_for_handle(rollout, handle)
    aborted = []
    driver = type('Driver', (), {'abort': lambda self: aborted.append(True)})()
    with pytest.raises(ConnectionError, match='unknown remote') as error:
        with rollout.streaming(driver):
            rollout.collect(handle)
    assert aborted == [True]
    assert not rollout.is_idle
    assert any('streaming cleanup' in note for note in error.value.__notes__)
    assert any(handle.submission_id in note for note in error.value.__notes__)
    with pytest.raises(BaseExceptionGroup):
        rollout.drain()


def test_drain_waits_for_admission_already_in_progress(monkeypatch):
    rollout = ClientMultiTurnRollout(_Sampler(((30, 'stop'),)), _Template())
    starting, release, waiting = threading.Event(), threading.Event(), threading.Event()
    original_start, original_wait = rollout._start, rollout._condition.wait

    def start(*args, **kwargs):
        starting.set()
        assert release.wait(timeout=5)
        return original_start(*args, **kwargs)

    def wait(*args, **kwargs):
        waiting.set()
        return original_wait(*args, **kwargs)

    monkeypatch.setattr(rollout, '_start', start)
    monkeypatch.setattr(rollout._condition, 'wait', wait)
    with ThreadPoolExecutor(max_workers=2) as pool:
        admitted = pool.submit(rollout.admit, _trajectory()['messages'])
        assert starting.wait(timeout=5)
        drained = pool.submit(rollout.drain)
        try:
            assert waiting.wait(timeout=5)
            assert not drained.done()
        finally:
            release.set()
        handle = admitted.result(timeout=5)
        drained.result(timeout=5)
    assert rollout.is_idle
    assert rollout.stop(handle.submission_id) is True


def test_blackjack_tool_actions_flow_through_http_adapter_and_ledger():
    import importlib.util
    from pathlib import Path
    from twinkle_agentic.tools.tool_manager import ToolManager

    path = Path(__file__).parents[2] / 'cookbook/client/async_rl/streaming_multi_turn_grpo.py'
    spec = importlib.util.spec_from_file_location('blackjack_sdk_test', path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    # Find a deterministic hand for which hit is nonterminal, then stand ends it.
    for seed in range(100):
        probe = recipe.BlackjackEnv(seed)
        probe.reset()
        continues = not probe.step('play', {'action': 'hit'}).done
        probe.close()
        if continues:
            break
    else:
        raise AssertionError('no nonterminal hit hand found')
    env = recipe.BlackjackEnv(seed)
    initial = env.reset()
    tool = env.play_tool()
    manager = ToolManager([tool])

    class Template(_Template):
        def append_completion(self, state, tokens, **kwargs):
            next_state = splice_state(state, tokens, tokens, scored=True, sampled=True)
            calls = () if tokens == [30] else ({'id': f'play-{tokens[0]}', 'type': 'function',
                'function': {'name': 'play', 'arguments': {'action': 'hit' if tokens == [10] else 'stand'}}},)
            response = DecodedResponse(content='play', tool_calls=calls, reasoning_started=False)
            message = {'role': 'assistant', 'content': 'play', 'tool_calls': list(calls)}
            return CompletionEncoding(replace(next_state, messages=state.messages + (message,)), response)

    rollout = ClientMultiTurnRollout(_Sampler(((10, 'stop'), (20, 'stop'), (30, 'stop'))), Template(), max_turns=4)
    driver = type('Driver', (), {'abort': lambda self: None})()
    try:
        with rollout.streaming(driver):
            handle = rollout.admit([{'role': 'user', 'content': initial.observation}],
                                   tool_manager=manager, prompt_extras={'tools': manager.tool_infos()})
            _wait_for_handle(rollout, handle)
            sample = rollout.collect(handle)
            assert tool.done
            assert tool.episode_reward in (-1, 0, 1)
            assert sample.response_token_ids[:2] == [10, 20]
            assert len(sample.rollout_logprobs) == len(sample.response_token_ids)
            assert any(message['role'] == 'tool' for message in sample.messages)
        assert rollout.is_idle
    finally:
        env.close()


def test_streaming_scope_reports_abort_failure_without_replacing_body_error():
    class Driver:
        def abort(self):
            raise SystemExit('abort failed')

    rollout = ClientMultiTurnRollout(_Sampler(), _Template())
    with pytest.raises(BaseExceptionGroup) as grouped:
        with rollout.streaming(Driver()):
            pass
    assert any(isinstance(error, SystemExit) for error in grouped.value.exceptions)
    primary = ValueError('training failed')
    rollout = ClientMultiTurnRollout(_Sampler(), _Template())
    with pytest.raises(ValueError) as caught:
        with rollout.streaming(Driver()):
            raise primary
    assert caught.value is primary
    assert any('abort failed' in note for note in primary.__notes__)


def test_drain_uses_immutable_submission_identity_after_handle_tampering(monkeypatch):
    rollout = ClientMultiTurnRollout(_Sampler(((30, 'stop'),)), _Template())
    handle = rollout.admit(_trajectory()['messages'])
    _wait_for_handle(rollout, handle)
    entry = rollout._streaming[handle.submission_id]
    cancelled = []
    original = entry.call.execution.cancel

    def cancel():
        cancelled.append(True)
        original()

    monkeypatch.setattr(entry.call.execution, 'cancel', cancel)
    handle.submission_id = 'modified-by-caller'
    with pytest.raises(RuntimeError, match='unknown'):
        rollout.poll([handle])
    rollout.drain()
    first = len(cancelled)
    rollout.drain()
    assert len(cancelled) == first, 'already settled calls must not be drained again'
    assert rollout.is_idle
