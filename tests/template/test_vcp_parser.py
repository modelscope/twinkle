# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for VCPParser and the ToolCallParser base-class defaults."""
import pytest

from twinkle.template.tools.base import ToolCallParser, ToolCallRegistry
from twinkle.template.tools.vcp import VCPParser


def _block(body: str) -> str:
    return f'<<<[TOOL_REQUEST]>>>{body}<<<[END_TOOL_REQUEST]>>>'


class TestVcpParser:

    def setup_method(self):
        self.parser = VCPParser()

    def test_detect_requires_open_marker(self):
        assert self.parser.detect(_block('tool_name: 「始」f「末」'))
        assert not self.parser.detect('plain text')
        assert not self.parser.detect('')
        assert not self.parser.detect(None)

    def test_parse_extracts_name_and_arguments(self):
        text = _block('tool_name: 「始」get_weather「末」\ncity: 「始」Beijing「末」days: 「始」3「末」')
        calls = self.parser.parse(text)
        assert calls == [{
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'arguments': {'city': 'Beijing', 'days': '3'},
            },
        }]

    def test_tool_name_is_stripped(self):
        calls = self.parser.parse(_block('tool_name: 「始」  padded  「末」'))
        assert calls[0]['function']['name'] == 'padded'

    def test_escape_variant_keeps_bare_close_markers_verbatim(self):
        # The ESCAPE pairing is strict: a bare 「末」 inside the value must not close
        # it, and inner bracket pairs survive as literal text.
        text = _block('tool_name: 「始」echo「末」payload: 「始ESCAPE」a「末」b「始」c「末」d「末ESCAPE」')
        calls = self.parser.parse(text)
        assert calls[0]['function']['arguments'] == {'payload': 'a「末」b「始」c「末」d'}

    def test_value_may_span_lines(self):
        text = _block('tool_name: 「始」f「末」note: 「始」line1\nline2「末」')
        calls = self.parser.parse(text)
        assert calls[0]['function']['arguments'] == {'note': 'line1\nline2'}

    def test_block_without_tool_name_reports_parse_error(self):
        # detect() is True but parse() is empty: parse_errors must explain why, or
        # the caller would silently drop the model's tool-call attempt.
        text = _block('city: 「始」Beijing「末」')
        assert self.parser.detect(text)
        assert self.parser.parse(text) == []
        errors = self.parser.parse_errors(text)
        assert len(errors) == 1
        assert 'tool_name' in errors[0]

    def test_multiple_blocks_produce_ordered_calls(self):
        text = (_block('tool_name: 「始」first「末」') + ' middle '
                + _block('tool_name: 「始」second「末」x: 「始」1「末」'))
        calls = self.parser.parse(text)
        assert [call['function']['name'] for call in calls] == ['first', 'second']
        assert calls[1]['function']['arguments'] == {'x': '1'}

    def test_unclosed_block_is_ignored_entirely(self):
        # A truncated stream leaves no matching close marker: the regex cannot match,
        # so the partial block yields neither a call nor an error.
        text = '<<<[TOOL_REQUEST]>>>tool_name: 「始」f「末」'
        assert self.parser.parse(text) == []
        assert self.parser.parse_errors(text) == []

    def test_clean_strips_blocks_and_trailing_whitespace(self):
        text = 'before ' + _block('tool_name: 「始」f「末」') + ' after  '
        assert self.parser.clean(text) == 'before  after'
        assert self.parser.clean('no blocks here') == 'no blocks here'
        assert self.parser.clean(None) == ''


class TestToolCallParserDefaults:

    class _Minimal(ToolCallParser):

        def detect(self, text):
            return True

        def parse(self, text):
            return []

        def clean(self, text):
            return text

    def test_base_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ToolCallParser()

    def test_default_parse_errors_is_empty(self):
        # A parser whose parse regex equals its detect regex cannot disagree with
        # detect, so the default reports no errors.
        parser = self._Minimal()
        assert parser.parse_errors('anything') == []

    def test_default_extract_tool_result_is_none(self):
        assert self._Minimal().extract_tool_result('anything') is None

    def test_vcp_is_registered(self):
        names = [parser.name for parser in ToolCallRegistry.parsers()]
        assert 'vcp' in names
