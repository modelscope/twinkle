# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for the pure helpers in twinkle.template.deepseek_v4_encoding."""
import json

import pytest

from twinkle.template import deepseek_v4_encoding as enc


class TestToJson:

    def test_plain_values_match_stdlib(self):
        assert enc.to_json({'a': 1, 'b': [True, None]}) == '{"a": 1, "b": [true, null]}'

    def test_non_ascii_is_kept_verbatim(self):
        # ensure_ascii=False: CJK text is not escaped to \\uXXXX.
        assert enc.to_json('中文') == '"中文"'
        assert enc.to_json({'city': '北京'}) == '{"city": "北京"}'


class TestOpenAiFormatShims:

    def test_tools_from_openai_format_keeps_function_section(self):
        tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {'a': 1}}}]
        assert enc.tools_from_openai_format(tools) == [{'name': 'f', 'parameters': {'a': 1}}]

    def test_tool_calls_round_trip_preserves_name_and_arguments(self):
        tool_calls = [{'name': 'get_weather', 'arguments': '{"city": "Beijing"}'}]
        openai = enc.tool_calls_to_openai_format(tool_calls)
        assert openai == [{
            'type': 'function',
            'function': {'name': 'get_weather', 'arguments': '{"city": "Beijing"}'},
        }]
        assert enc.tool_calls_from_openai_format(openai) == tool_calls


class TestEncodeArgumentsToDsml:

    def test_string_arguments_are_embedded_verbatim(self):
        encoded = enc.encode_arguments_to_dsml({'name': 'f', 'arguments': {'city': 'Beijing'}})
        assert encoded == '<｜DSML｜parameter name="city" string="true">Beijing</｜DSML｜parameter>'

    def test_non_string_arguments_are_json_encoded(self):
        arguments = {'n': 5, 'ok': True, 'nothing': None, 'arr': [1, 'a'], 'obj': {'k': 'v'}}
        encoded = enc.encode_arguments_to_dsml({'name': 'f', 'arguments': arguments})
        expected = '\n'.join(
            f'<｜DSML｜parameter name="{k}" string="false">{json.dumps(v, ensure_ascii=False)}</'
            f'｜DSML｜parameter>' for k, v in arguments.items())
        assert encoded == expected

    def test_string_arguments_json_payload_is_equivalent_to_dict(self):
        via_str = enc.encode_arguments_to_dsml({'name': 'f', 'arguments': json.dumps({'a': 1, 'b': 'x'})})
        via_dict = enc.encode_arguments_to_dsml({'name': 'f', 'arguments': {'a': 1, 'b': 'x'}})
        assert via_str == via_dict

    def test_empty_arguments_render_nothing(self):
        assert enc.encode_arguments_to_dsml({'name': 'f', 'arguments': {}}) == ''


class TestDecodeDsmlToArguments:

    def test_true_flag_quotes_and_false_flag_stays_raw(self):
        # string="true" values must come back quoted (so json.loads keeps them str);
        # string="false" values are already JSON and stay raw.
        result = enc.decode_dsml_to_arguments(
            'get_weather', {'city': ('Beijing', 'true'), 'days': ('3', 'false')})
        assert result['name'] == 'get_weather'
        assert json.loads(result['arguments']) == {'city': 'Beijing', 'days': 3}


def _render_tool_calls_block(tool_calls):
    """Build the exact <｜DSML｜tool_calls> block the renderer emits."""
    invokes = []
    for tool_call in tool_calls:
        arguments = enc.encode_arguments_to_dsml(tool_call)
        invokes.append(
            f'<{enc.dsml_token}invoke name="{tool_call["name"]}">\n{arguments}\n</{enc.dsml_token}invoke>')
    body = '\n'.join(invokes)
    return f'>\n{body}\n</{enc.dsml_token}{enc.tool_calls_block_name}>'


class TestToolCallRoundTrip:

    def test_round_trip_preserves_json_types(self):
        # Adversarial arguments: strings whose text looks like JSON scalars must stay
        # strings; real bools/ints/containers must keep their types.
        arguments = {
            'flag': 'false',
            'id': '123',
            'note': 'null',
            'ok': True,
            'n': 5,
            'arr': [1, 'a'],
            'obj': {'k': 'v'},
        }
        tool_call = {'name': 'probe', 'arguments': arguments}
        _, _, parsed = enc.parse_tool_calls(0, _render_tool_calls_block([tool_call]))
        assert len(parsed) == 1
        assert parsed[0]['name'] == 'probe'
        assert json.loads(parsed[0]['arguments']) == arguments

    def test_round_trip_survives_special_characters(self):
        arguments = {'text': 'a"b\nc<d>\t和'}
        tool_call = {'name': 'probe', 'arguments': arguments}
        _, _, parsed = enc.parse_tool_calls(0, _render_tool_calls_block([tool_call]))
        assert json.loads(parsed[0]['arguments']) == arguments

    def test_multiple_invokes_keep_order(self):
        tool_calls = [
            {'name': 'first', 'arguments': {'x': 1}},
            {'name': 'second', 'arguments': {'y': 'two'}},
        ]
        _, _, parsed = enc.parse_tool_calls(0, _render_tool_calls_block(tool_calls))
        assert [tc['name'] for tc in parsed] == ['first', 'second']
        assert json.loads(parsed[0]['arguments']) == {'x': 1}
        assert json.loads(parsed[1]['arguments']) == {'y': 'two'}

    def test_renderer_output_parses_back_through_completion_text(self):
        # End-to-end: render_message's assistant tool-call rendering must be decodable
        # by parse_message_from_completion_text back to the same arguments.
        messages = [
            {'role': 'user', 'content': 'weather?'},
            {'role': 'assistant', 'content': '', 'tool_calls': [
                {'function': {'name': 'get_weather', 'arguments': json.dumps({'city': 'Beijing', 'days': 3})}},
            ]},
        ]
        prompt = enc.render_message(1, messages, thinking_mode='chat')
        parsed = enc.parse_message_from_completion_text(prompt, thinking_mode='chat')
        assert parsed['role'] == 'assistant'
        assert [tc['function']['name'] for tc in parsed['tool_calls']] == ['get_weather']
        assert json.loads(parsed['tool_calls'][0]['function']['arguments']) == {'city': 'Beijing', 'days': 3}


class TestFindLastUserIndex:

    def test_counts_user_and_developer(self):
        messages = [
            {'role': 'system', 'content': 'x'},
            {'role': 'developer', 'content': 'd'},
            {'role': 'assistant', 'content': 'a'},
            {'role': 'user', 'content': 'u'},
        ]
        assert enc.find_last_user_index(messages) == 3
        messages[-1] = {'role': 'assistant', 'content': 'a2'}
        assert enc.find_last_user_index(messages) == 1

    def test_no_user_returns_minus_one(self):
        assert enc.find_last_user_index([{'role': 'assistant', 'content': 'a'}]) == -1


class TestRenderMessage:

    def test_system_message_appends_tools_and_response_format(self):
        tools = [{'type': 'function', 'function': {'name': 'f', 'parameters': {'a': 1}}}]
        response_format = {'type': 'json_object'}
        prompt = enc.render_message(
            0, [{'role': 'system', 'content': 'sys', 'tools': tools, 'response_format': response_format}],
            thinking_mode='chat')
        assert prompt.startswith('sys\n\n')
        assert f'<{enc.dsml_token}tool_calls>' in prompt
        assert json.dumps(tools[0]['function'], ensure_ascii=False) in prompt
        assert json.dumps(response_format, ensure_ascii=False) in prompt

    def test_developer_message_requires_content(self):
        with pytest.raises(AssertionError, match='Invalid message for role'):
            enc.render_message(0, [{'role': 'developer', 'content': ''}], thinking_mode='chat')

    def test_user_message_renders_special_tokens(self):
        prompt = enc.render_message(0, [{'role': 'user', 'content': 'hi'}], thinking_mode='chat')
        assert prompt == f'{enc.USER_SP_TOKEN}hi{enc.ASSISTANT_SP_TOKEN}{enc.thinking_end_token}'

    def test_user_content_blocks_render_tool_result_and_unsupported(self):
        messages = [{
            'role': 'user',
            'content_blocks': [
                {'type': 'text', 'text': 'q'},
                {'type': 'tool_result', 'content': [
                    {'type': 'text', 'text': 'ok'},
                    {'type': 'image_url', 'url': 'x'},
                ]},
                {'type': 'image_url', 'url': 'y'},
            ],
        }]
        prompt = enc.render_message(0, messages, thinking_mode='chat')
        assert '<tool_result>ok\n\n[Unsupported image_url]</tool_result>' in prompt
        assert '[Unsupported image_url]' in prompt
        assert prompt.startswith(enc.USER_SP_TOKEN + 'q')

    def test_tool_role_is_rejected_with_merge_hint(self):
        with pytest.raises(NotImplementedError, match='merge_tool_messages'):
            enc.render_message(0, [{'role': 'tool', 'content': 'r'}], thinking_mode='chat')

    def test_unknown_role_is_rejected(self):
        with pytest.raises(NotImplementedError, match='Unknown role'):
            enc.render_message(0, [{'role': 'bogus', 'content': 'x'}], thinking_mode='chat')

    def test_invalid_thinking_mode_is_rejected(self):
        with pytest.raises(AssertionError, match='Invalid thinking_mode'):
            enc.render_message(0, [{'role': 'user', 'content': 'x'}], thinking_mode='bogus')

    def test_thinking_dropped_before_last_user_and_kept_after(self):
        before_user = [{'role': 'assistant', 'content': 'a1', 'reasoning': 'R1'},
                       {'role': 'user', 'content': 'q'},
                       {'role': 'assistant', 'content': 'a2', 'reasoning': 'R2'}]
        dropped = enc.render_message(0, before_user, thinking_mode='thinking', drop_thinking=True)
        assert 'R1' not in dropped
        kept = enc.render_message(2, before_user, thinking_mode='thinking', drop_thinking=True)
        assert f'{enc.thinking_template.format(reasoning="R2")}{enc.thinking_end_token}' in kept

    def test_wo_eos_assistant_has_no_eos_suffix(self):
        messages = [{'role': 'user', 'content': 'q'}, {'role': 'assistant', 'content': 'a', 'wo_eos': True}]
        prompt = enc.render_message(1, messages, thinking_mode='chat')
        assert not prompt.endswith(enc.eos_token)

    def test_task_action_adds_assistant_and_thinking_tokens(self):
        messages = [{'role': 'assistant', 'content': 'a', 'task': 'action'}]
        prompt = enc.render_message(0, messages, thinking_mode='thinking')
        assert prompt.endswith(f'{enc.ASSISTANT_SP_TOKEN}{enc.thinking_start_token}'
                               f"{enc.DS_TASK_SP_TOKENS['action']}")

    def test_non_action_task_appends_only_task_token(self):
        messages = [{'role': 'assistant', 'content': 'a', 'task': 'query'}]
        prompt = enc.render_message(0, messages, thinking_mode='chat')
        assert prompt.endswith(enc.DS_TASK_SP_TOKENS['query'])
        assert enc.ASSISTANT_SP_TOKEN not in prompt

    def test_invalid_task_is_rejected(self):
        messages = [{'role': 'assistant', 'content': 'a', 'task': 'bogus'}]
        with pytest.raises(AssertionError, match='Invalid task'):
            enc.render_message(0, messages, thinking_mode='chat')


class TestMergeToolMessages:

    def test_tool_message_merges_into_preceding_user_blocks(self):
        # Canonical input shape: a plain-text user message followed by a tool result.
        messages = [
            {'role': 'user', 'content': 'q'},
            {'role': 'tool', 'tool_call_id': 't2', 'content': 'r'},
        ]
        merged = enc.merge_tool_messages(messages)
        assert len(merged) == 1
        assert merged[0]['role'] == 'user'
        assert merged[0]['content'] == 'q'
        blocks = merged[0]['content_blocks']
        assert blocks[0] == {'type': 'text', 'text': 'q'}
        assert blocks[1] == {'type': 'tool_result', 'tool_use_id': 't2', 'content': 'r'}

    def test_tool_message_without_preceding_user_creates_one(self):
        merged = enc.merge_tool_messages([{'role': 'tool', 'tool_call_id': 't', 'content': 'r'}])
        assert merged == [{'role': 'user', 'content_blocks': [
            {'type': 'tool_result', 'tool_use_id': 't', 'content': 'r'}]}]

    def test_consecutive_plain_users_merge_text_blocks(self):
        merged = enc.merge_tool_messages([{'role': 'user', 'content': 'a'}, {'role': 'user', 'content': 'b'}])
        assert len(merged) == 1
        assert merged[0]['content'] == 'a'
        assert [b['text'] for b in merged[0]['content_blocks']] == ['a', 'b']

    def test_user_after_task_user_starts_a_new_message(self):
        # A user carrying a task is a standalone unit: a later plain user must not be
        # appended into it (the merged[-1].task is None guard).
        merged = enc.merge_tool_messages([
            {'role': 'user', 'content': 'a', 'task': 'query'},
            {'role': 'user', 'content': 'b'},
        ])
        assert len(merged) == 2
        assert merged[0]['task'] == 'query'
        assert merged[1]['content'] == 'b'

    def test_user_with_task_starts_a_new_message(self):
        # A task-bearing user is a standalone quick-instruction unit: it must not be
        # appended into a preceding plain user, or its task key would be dropped.
        merged = enc.merge_tool_messages([
            {'role': 'user', 'content': 'a'},
            {'role': 'user', 'content': 'b', 'task': 'query'},
        ])
        assert len(merged) == 2
        assert merged[1]['task'] == 'query'
        assert merged[1]['content'] == 'b'

    def test_user_content_blocks_are_preserved_not_replaced(self):
        # A user already carrying content_blocks (a multimodal row, or this merge's
        # own output) keeps its blocks verbatim instead of losing them to a
        # synthesized text-only block.
        messages = [{'role': 'user', 'content': 'q', 'content_blocks': [
            {'type': 'text', 'text': 'q'},
            {'type': 'tool_result', 'tool_use_id': 't1', 'content': 'r'},
        ]}]
        merged = enc.merge_tool_messages(messages)
        assert merged[0]['content_blocks'] == messages[0]['content_blocks']

    def test_merge_is_idempotent(self):
        messages = [
            {'role': 'user', 'content': 'q'},
            {'role': 'tool', 'tool_call_id': 't', 'content': 'r'},
            {'role': 'user', 'content': 'follow-up'},
        ]
        once = enc.merge_tool_messages(messages)
        assert enc.merge_tool_messages(once) == once

    def test_input_messages_are_not_mutated(self):
        messages = [{'role': 'user', 'content': 'a'}, {'role': 'tool', 'tool_call_id': 't', 'content': 'r'}]
        snapshot = [dict(m) for m in messages]
        enc.merge_tool_messages(messages)
        assert messages == snapshot


class TestSortToolResultsByCallOrder:

    def test_tool_results_are_reordered_by_call_order(self):
        messages = [
            {'role': 'assistant', 'tool_calls': [
                {'id': 'a', 'function': {'name': 'f'}},
                {'id': 'b', 'function': {'name': 'g'}},
            ]},
            {'role': 'user', 'content_blocks': [
                {'type': 'tool_result', 'tool_use_id': 'b', 'content': 'B'},
                {'type': 'tool_result', 'tool_use_id': 'a', 'content': 'A'},
            ]},
        ]
        sorted_messages = enc.sort_tool_results_by_call_order(messages)
        blocks = sorted_messages[1]['content_blocks']
        assert [b['tool_use_id'] for b in blocks] == ['a', 'b']
        assert [b['content'] for b in blocks] == ['A', 'B']

    def test_unknown_tool_use_id_sorts_first(self):
        messages = [
            {'role': 'assistant', 'tool_calls': [
                {'id': 'z', 'function': {'name': 'f'}},
                {'id': 'a', 'function': {'name': 'g'}},
            ]},
            {'role': 'user', 'content_blocks': [
                {'type': 'tool_result', 'tool_use_id': 'a', 'content': 'A'},
                {'type': 'tool_result', 'tool_use_id': 'unknown', 'content': 'U'},
            ]},
        ]
        sorted_messages = enc.sort_tool_results_by_call_order(messages)
        assert [b['tool_use_id'] for b in sorted_messages[1]['content_blocks']] == ['unknown', 'a']

    def test_single_tool_result_and_plain_users_are_untouched(self):
        messages = [
            {'role': 'assistant', 'tool_calls': [{'id': 'a', 'function': {'name': 'f'}}]},
            {'role': 'user', 'content_blocks': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'A'}]},
            {'role': 'user', 'content': 'plain'},
        ]
        assert enc.sort_tool_results_by_call_order(messages) == messages


class TestDropThinkingMessages:

    def test_assistant_reasoning_dropped_before_last_user_kept_after(self):
        messages = [
            {'role': 'assistant', 'content': 'a1', 'reasoning': 'R1'},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a2', 'reasoning': 'R2'},
        ]
        dropped = enc._drop_thinking_messages(messages)
        assert 'reasoning' not in dropped[0]
        assert dropped[0]['content'] == 'a1'
        assert dropped[2]['reasoning'] == 'R2'

    def test_keep_roles_survive(self):
        messages = [
            {'role': 'system', 'content': 's'},
            {'role': 'tool', 'content': 't'},
            {'role': 'direct_search_results', 'content': 'd'},
        ]
        assert enc._drop_thinking_messages(messages) == messages

    def test_no_user_keeps_everything(self):
        messages = [{'role': 'assistant', 'content': 'a', 'reasoning': 'R'}]
        assert enc._drop_thinking_messages(messages) == messages


class TestReadUntilStop:

    def test_earliest_stop_wins(self):
        index, content, stop = enc._read_until_stop(0, 'abcXXdefYYghi', ['YY', 'XX'])
        assert (index, content, stop) == (5, 'abc', 'XX')

    def test_start_offset_is_respected(self):
        index, content, stop = enc._read_until_stop(5, 'abcXXdef', ['XX'])
        assert (index, content, stop) == (8, 'def', None)

    def test_no_match_consumes_the_tail(self):
        assert enc._read_until_stop(0, 'abc', ['ZZ']) == (3, 'abc', None)
