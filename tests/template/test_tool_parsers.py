# Copyright (c) ModelScope Contributors. All rights reserved.
"""Pure-Python tests for tool-call parsers (no model download).

Covers Hermes/Qwen, ReAct, Cline parse / clean / detect and registry
round-robin selection.
"""
import pytest

from twinkle.template.tools import ClineParser, HermesQwenParser, ReActParser, ToolCallRegistry

# ---------------------------------------------------------------------------
# HermesQwenParser
# ---------------------------------------------------------------------------


class TestHermesQwenParser:

    def setup_method(self):
        self.p = HermesQwenParser()

    def test_detect(self):
        assert self.p.detect('hi <tool_call>{"name":"f","arguments":{}}</tool_call>')
        assert not self.p.detect('plain text')
        assert not self.p.detect('')

    def test_matches_model(self):
        assert self.p.matches_model('qwen2.5-7b')
        assert self.p.matches_model('qwen3-32b')
        assert not self.p.matches_model('llama-3.1-8b')

    def test_parse_json_variant(self):
        text = '<tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
        out = self.p.parse(text)
        assert out == [{
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'arguments': {
                    'city': 'Paris'
                }
            },
        }]

    def test_parse_function_xml_variant(self):
        text = ('<tool_call><function=add>'
                '<parameter=a>1</parameter><parameter=b>2</parameter>'
                '</function></tool_call>')
        out = self.p.parse(text)
        assert len(out) == 1
        assert out[0]['function']['name'] == 'add'
        # JSON-decoding of param values: numbers come back as int.
        assert out[0]['function']['arguments'] == {'a': 1, 'b': 2}

    def test_parse_multiple_blocks(self):
        text = ('<tool_call>{"name":"f1","arguments":{}}</tool_call>'
                'between '
                '<tool_call>{"name":"f2","arguments":{"k":"v"}}</tool_call>')
        out = self.p.parse(text)
        assert [c['function']['name'] for c in out] == ['f1', 'f2']
        assert out[1]['function']['arguments'] == {'k': 'v'}

    def test_parse_unclosed_block_at_eof(self):
        # ``\Z`` fallback in _BLOCK_RE handles truncated trailing block.
        text = '<tool_call>{"name": "f", "arguments": {}}'
        out = self.p.parse(text)
        assert out and out[0]['function']['name'] == 'f'

    def test_parse_empty_returns_empty_list(self):
        assert self.p.parse('') == []
        assert self.p.parse('plain text without markers') == []

    def test_clean_strips_blocks(self):
        text = 'hello <tool_call>{"name":"f","arguments":{}}</tool_call> world'
        assert self.p.clean(text) == 'hello  world'

    def test_clean_unclosed_at_eof(self):
        text = 'hello <tool_call>{"name":"f"'
        assert self.p.clean(text) == 'hello'

    def test_clean_empty(self):
        assert self.p.clean('') == ''

    def test_markers_declared(self):
        assert self.p.open_marker == '<tool_call>'
        assert self.p.close_marker == '</tool_call>'


# ---------------------------------------------------------------------------
# ReActParser
# ---------------------------------------------------------------------------


class TestReActParser:

    def setup_method(self):
        self.p = ReActParser()

    def test_detect_action_line(self):
        assert self.p.detect('Thought: I need search.\nAction: search[python]')
        assert not self.p.detect('plain text without action keyword')
        assert not self.p.detect('')

    def test_no_block_marker(self):
        assert self.p.open_marker is None
        assert self.p.close_marker is None

    def test_does_not_match_qwen_model(self):
        assert not self.p.matches_model('qwen2.5')
        assert not self.p.matches_model('llama-3')

    def test_parse_single_action(self):
        text = 'Thought: search the web.\nAction: search[hello world]'
        out = self.p.parse(text)
        assert out == [{
            'type': 'function',
            'function': {
                'name': 'search',
                'arguments': {
                    'input': 'hello world'
                }
            },
        }]

    def test_parse_multiple_actions(self):
        text = ('Thought: a\nAction: tool_a[x]\n'
                'Observation: ok\n'
                'Thought: b\nAction: tool_b[y z]')
        out = self.p.parse(text)
        assert [c['function']['name'] for c in out] == ['tool_a', 'tool_b']
        assert out[1]['function']['arguments'] == {'input': 'y z'}

    def test_clean_removes_action_lines(self):
        text = 'Thought: hi\nAction: search[x]\nDone'
        cleaned = self.p.clean(text)
        assert 'Action: search' not in cleaned
        assert 'Thought: hi' in cleaned
        assert 'Done' in cleaned

    def test_parse_empty(self):
        assert self.p.parse('') == []


# ---------------------------------------------------------------------------
# ClineParser
# ---------------------------------------------------------------------------


class TestClineParser:

    def setup_method(self):
        self.p = ClineParser()

    def test_detect_simple_tool(self):
        assert self.p.detect('<read_file><path>foo.py</path></read_file>')

    def test_detect_ignores_html_like_tags(self):
        # ``think`` / ``code`` are denied — even with inner content they aren't
        # treated as tool calls.
        assert not self.p.detect('<think><inner>x</inner></think>')
        assert not self.p.detect('<code><line>x</line></code>')

    def test_detect_requires_inner_param(self):
        # No inner ``<key>VAL</key>`` → not a Cline call.
        assert not self.p.detect('<read_file>just text</read_file>')

    def test_detect_ignores_hermes_block(self):
        # Hermes already owns ``<tool_call>`` — Cline must skip it.
        assert not self.p.detect('<tool_call>{"name":"f","arguments":{}}</tool_call>')

    def test_no_marker(self):
        # Outer tag varies per call — no fixed marker.
        assert self.p.open_marker is None
        assert self.p.close_marker is None

    def test_does_not_match_any_model_by_default(self):
        # Cline is an app-level prompt protocol, not a model-family format.
        assert not self.p.matches_model('qwen2.5')
        assert not self.p.matches_model('claude-3')

    def test_parse_single_arg(self):
        text = '<read_file><path>src/foo.py</path></read_file>'
        out = self.p.parse(text)
        assert out == [{
            'type': 'function',
            'function': {
                'name': 'read_file',
                'arguments': {
                    'path': 'src/foo.py'
                }
            },
        }]

    def test_parse_multi_arg_with_whitespace(self):
        text = ('<execute_command>\n'
                '  <command>ls -la</command>\n'
                '  <requires_approval>false</requires_approval>\n'
                '</execute_command>')
        out = self.p.parse(text)
        fn = out[0]['function']
        assert fn['name'] == 'execute_command'
        assert fn['arguments'] == {'command': 'ls -la', 'requires_approval': 'false'}

    def test_parse_multiple_blocks(self):
        text = ('<read_file><path>a</path></read_file>'
                ' between '
                '<list_files><path>b</path><recursive>true</recursive></list_files>')
        out = self.p.parse(text)
        assert [c['function']['name'] for c in out] == ['read_file', 'list_files']
        assert out[1]['function']['arguments'] == {'path': 'b', 'recursive': 'true'}

    def test_parse_skips_hermes_block(self):
        text = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
        assert self.p.parse(text) == []

    def test_clean_strips_tool_blocks(self):
        text = 'before <read_file><path>x</path></read_file> after'
        assert self.p.clean(text) == 'before  after'

    def test_clean_preserves_non_tool_xml(self):
        text = '<think>reasoning</think> <read_file><path>x</path></read_file> tail'
        cleaned = self.p.clean(text)
        assert '<think>reasoning</think>' in cleaned
        assert '<read_file>' not in cleaned
        assert 'tail' in cleaned

    def test_clean_empty(self):
        assert self.p.clean('') == ''


# ---------------------------------------------------------------------------
# Registry round-robin
# ---------------------------------------------------------------------------


class TestRegistryRoundRobin:

    def test_first_match_wins_no_nested_reparse(self):
        # Hermes block must take ownership; ReAct/Cline shouldn't see it.
        text = '<tool_call>{"name":"f","arguments":{}}</tool_call>'
        parser = ToolCallRegistry.detect_first(text)
        assert parser is not None and parser.name == 'hermes_qwen'

    def test_cline_wins_for_xml_tools(self):
        text = '<read_file><path>x</path></read_file>'
        parser = ToolCallRegistry.detect_first(text)
        assert parser is not None and parser.name == 'cline'

    def test_react_wins_for_action_keyword(self):
        text = 'Thought: hi\nAction: search[x]'
        parser = ToolCallRegistry.detect_first(text)
        assert parser is not None and parser.name == 'react'

    def test_no_parser_for_plain_text(self):
        assert ToolCallRegistry.detect_first('just some plain text') is None
        assert ToolCallRegistry.detect_first('') is None

    def test_select_for_qwen_picks_hermes(self):
        parser = ToolCallRegistry.select_for_model('qwen2.5-7b')
        assert parser is not None and parser.name == 'hermes_qwen'

    def test_select_for_unknown_returns_none(self):
        assert ToolCallRegistry.select_for_model('llama-3.1-8b') is None
        assert ToolCallRegistry.select_for_model(None) is None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
