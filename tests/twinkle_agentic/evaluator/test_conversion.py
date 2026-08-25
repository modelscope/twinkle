import pytest

from evalscope.api.messages import ChatMessageAssistant, ChatMessageTool, ChatMessageUser, ContentImage, ContentReasoning, ContentText
from evalscope.api.model import GenerateConfig, Model
from evalscope.api.tool import ToolInfo, ToolParams

from twinkle_agentic.evaluator._contracts import BackendContractError, UnsupportedCapabilityError
from twinkle_agentic.evaluator._evalscope_adapter import ProtocolModelAPI, SamplerModelAPI, to_twinkle_trajectory

from .conftest import RecordingAPI, RecordingSampler, ToolTemplate


def test_input_conversion_preserves_roles_reasoning_tools_and_does_not_mutate():
    assistant = ChatMessageAssistant(
        content=[ContentReasoning(reasoning='think'), ContentText(text='answer')],
        tool_calls=[{'id': 'call-1', 'function': {'name': 'lookup', 'arguments': {'q': 'x'}}}],
    )
    tool = ChatMessageTool(content='result', tool_call_id='call-1')
    source = [ChatMessageUser(content='question'), assistant, tool]
    tool_info = ToolInfo(name='lookup', description='find', parameters=ToolParams())
    trajectory = to_twinkle_trajectory(source, [tool_info])
    assert trajectory['messages'][1]['reasoning_content'] == 'think'
    assert trajectory['messages'][1]['tool_calls'][0]['function']['arguments'] == {'q': 'x'}
    assert trajectory['tools'][0]['function']['name'] == 'lookup'
    assert assistant.tool_calls[0].function.arguments == {'q': 'x'}


def test_multimodal_input_is_rejected():
    with pytest.raises(UnsupportedCapabilityError, match='Multimodal'):
        to_twinkle_trajectory([ChatMessageUser(content=[ContentImage(image='x')])], [])


def test_protocol_output_preserves_reasoning_multiple_choices_and_tools():
    api = RecordingAPI([
        {'role': 'assistant', 'content': 'one', 'reasoning_content': 'r', 'tool_calls': [
            {'function': {'name': 'f', 'arguments': '{"a": 1}'}}]},
        {'role': 'assistant', 'content': 'two', 'finish_reason': 'length'},
    ])
    output = Model(ProtocolModelAPI(api, 'fake', set()), GenerateConfig()).generate([ChatMessageUser(content='x')])
    assert len(output.choices) == 2
    assert output.choices[0].stop_reason == 'tool_calls'
    assert output.choices[0].message.tool_calls[0].id == 'call_0_0_0'
    assert output.choices[1].stop_reason == 'max_tokens'


def test_invalid_tool_arguments_fail():
    api = RecordingAPI({'role': 'assistant', 'tool_calls': [{'function': {'name': 'f', 'arguments': 'bad'}}]})
    with pytest.raises(BackendContractError, match='invalid JSON'):
        Model(ProtocolModelAPI(api, 'fake', set()), GenerateConfig()).generate([ChatMessageUser(content='x')])


def test_explicit_unsupported_config_fails_before_api_call():
    api = RecordingAPI()
    adapter = ProtocolModelAPI(api, 'fake', {'response_schema', 'temperature'})
    with pytest.raises(UnsupportedCapabilityError, match='response_schema'):
        Model(adapter, GenerateConfig(response_schema={'name': 'x', 'json_schema': {'type': 'object'}}, temperature=0)).generate(
            [ChatMessageUser(content='x')])
    assert not api.calls


def test_sampler_parses_tools_and_rejects_forcing():
    sampler = RecordingSampler()
    sampler.sample = lambda inputs, sampling_params, **kwargs: [
        {'sequences': [{'stop_reason': 'stop', 'tokens': [1], 'decoded': 'tool'}]} for _ in inputs]
    adapter = SamplerModelAPI(sampler, 's', ToolTemplate(), set(), batch_size=1, batch_wait_ms=0, sampler_kwargs={})
    try:
        output = Model(adapter, GenerateConfig()).generate([ChatMessageUser(content='x')], tools=[ToolInfo(name='lookup', description='x')])
        assert output.stop_reason == 'tool_calls'
        with pytest.raises(UnsupportedCapabilityError, match='tool_choice'):
            adapter.generate([ChatMessageUser(content='x')], [], 'any', GenerateConfig())
    finally:
        adapter.batcher.close()
