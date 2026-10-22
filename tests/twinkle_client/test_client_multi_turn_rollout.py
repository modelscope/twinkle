# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import replace

import pytest

from twinkle.template.contract import (CompletionEncoding, DecodedResponse, MediaBundle, PromptEncoding, RolloutState,
                                       refresh, splice, splice_state)
from twinkle_client.rollout.multi_turn import ClientMultiTurnRollout
from twinkle.protocol.types.sampler import SampledSequenceModel, SampleResponseModel


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

    def sample(self, features, sampling_params=None):
        self.calls.append(features)
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
        def sample(features, sampling_params=None):
            sequence = SampledSequenceModel(
                stop_reason='stop', tokens=[10], logprobs=[[(10, -0.1)]], decoded='x')
            return [SampleResponseModel(sequences=[sequence])]

    with pytest.raises(RuntimeError, match='batch index 0.*trajectory 0.*new_input_feature'):
        ClientMultiTurnRollout(BrokenSampler(), _Template())([_trajectory()])


def test_http_multi_turn_requires_tool_manager_only_when_called():
    with pytest.raises(ValueError, match='trajectory 0.*tool_manager'):
        ClientMultiTurnRollout(_Sampler(), _Template(), max_turns=2)([_trajectory()])


def test_http_multi_turn_marks_length_stop_as_truncated():
    result = ClientMultiTurnRollout(_Sampler(((10, 'length'), )), _Template())([_trajectory()])[0]
    assert result['turns'] == 1
    assert result['stop_reason'] == 'length'
    assert result['truncated'] is True


def test_http_multi_turn_preserves_batch_output_order():
    trajectories = [dict(_trajectory(), sample_id=index) for index in range(4)]
    results = ClientMultiTurnRollout(_Sampler(((30, 'stop'), )), _Template())(trajectories)
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
    assert result['stop_reason'] == 'max_turns'
    assert result['truncated'] is True
    assert RecordingTools.calls == 0


def test_http_multi_turn_propagates_network_error_unchanged():
    class NetworkError(Exception):
        pass

    sentinel = NetworkError('connection reset')

    class FailingSampler:

        @staticmethod
        def sample(features, sampling_params=None):
            raise sentinel

    with pytest.raises(NetworkError) as error:
        ClientMultiTurnRollout(FailingSampler(), _Template())([_trajectory()])
    assert error.value is sentinel


def test_http_multi_turn_validates_public_arguments():
    with pytest.raises(ValueError, match='Template'):
        ClientMultiTurnRollout(object(), None)
    with pytest.raises(ValueError, match='max_turns'):
        ClientMultiTurnRollout(object(), _Template(), max_turns=0)
