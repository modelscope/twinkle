# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for twinkle_agentic.tools: base Tool and ToolManager."""

import pytest

from twinkle.data_format import ToolCall
from twinkle.data_format.message import Tool as ToolInfo
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager, _extract_name


# ---------------------------------------------------------------------------
# Mock tools
# ---------------------------------------------------------------------------

class MockTool(Tool):

    def __init__(self, name='mock_tool'):
        self._name = name

    def __call__(self, tool_name: str, arguments: dict) -> str:
        return f'executed {tool_name} with {arguments}'

    def tool_info(self) -> ToolInfo:
        return {'function': {'name': self._name, 'description': 'A mock tool.'}}


class CalculatorTool(Tool):

    def __call__(self, tool_name: str, arguments: dict) -> str:
        op = arguments.get('operation', 'add')
        a = arguments.get('a', 0)
        b = arguments.get('b', 0)
        if op == 'add':
            return str(a + b)
        if op == 'subtract':
            return str(a - b)
        return 'unknown operation'

    def tool_info(self) -> ToolInfo:
        return {'function': {'name': 'calculator', 'description': 'Basic calculator.'}}


class FailingTool(Tool):

    def __call__(self, tool_name: str, arguments: dict) -> str:
        raise RuntimeError('intentional failure')

    def tool_info(self) -> ToolInfo:
        return {'function': {'name': 'failing_tool', 'description': 'Always fails.'}}


# ---------------------------------------------------------------------------
# _extract_name
# ---------------------------------------------------------------------------

class TestExtractName:

    def test_valid_dict(self):
        info = {'function': {'name': 'test_tool'}}
        assert _extract_name(info) == 'test_tool'

    def test_missing_function(self):
        assert _extract_name({'name': 'test_tool'}) is None

    def test_non_dict(self):
        assert _extract_name('not a dict') is None

    def test_empty_name(self):
        assert _extract_name({'function': {'name': ''}}) is None

    def test_none_name(self):
        assert _extract_name({'function': {'name': None}}) is None


# ---------------------------------------------------------------------------
# ToolManager
# ---------------------------------------------------------------------------

class TestToolManager:

    def test_empty_constructor(self):
        tm = ToolManager()
        assert tm.names() == []

    def test_from_dict(self):
        mock = MockTool()
        tm = ToolManager({'mock_tool': mock})
        assert tm.names() == ['mock_tool']

    def test_from_list(self):
        mock = MockTool()
        calc = CalculatorTool()
        tm = ToolManager([mock, calc])
        assert sorted(tm.names()) == ['calculator', 'mock_tool']

    def test_from_none(self):
        tm = ToolManager(None)
        assert tm.names() == []

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match='ToolManager expects dict'):
            ToolManager(42)

    def test_register(self):
        tm = ToolManager()
        mock = MockTool()
        tm.register(mock)
        assert 'mock_tool' in tm.names()

    def test_register_missing_name_raises(self):
        tm = ToolManager()
        with pytest.raises(ValueError, match='non-empty'):
            tm.register(object())  # object has no tool_info

    def test_unregister(self):
        mock = MockTool()
        tm = ToolManager({'mock_tool': mock})
        removed = tm.unregister('mock_tool')
        assert removed is mock
        assert 'mock_tool' not in tm.names()

    def test_unregister_missing(self):
        tm = ToolManager()
        assert tm.unregister('nonexistent') is None

    def test_copy(self):
        mock = MockTool()
        tm = ToolManager({'mock_tool': mock})
        copied = tm.copy()
        assert copied.names() == ['mock_tool']
        assert copied is not tm

    def test_tool_infos(self):
        mock = MockTool()
        tm = ToolManager({'mock_tool': mock})
        infos = tm.tool_infos()
        assert len(infos) == 1
        assert infos[0]['function']['name'] == 'mock_tool'

    def test_call_with_dict(self):
        calc = CalculatorTool()
        tm = ToolManager({'calculator': calc})
        result = tm({'function': {'name': 'calculator', 'arguments': {'a': 3, 'b': 4, 'operation': 'add'}}})
        assert result == '7'

    def test_call_with_tool_call(self):
        calc = CalculatorTool()
        tm = ToolManager({'calculator': calc})
        tc = ToolCall(**{'function': {'name': 'calculator', 'arguments': {'a': 10, 'b': 2, 'operation': 'subtract'}}})
        result = tm(tc)
        assert result == '8'

    def test_call_missing_tool(self):
        tm = ToolManager()
        result = tm({'function': {'name': 'missing', 'arguments': {}}})
        assert 'unknown tool' in result
        assert 'Available:' in result

    def test_call_missing_function(self):
        tm = ToolManager({'mock': MockTool()})
        result = tm({})
        assert 'missing "function"' in result

    def test_call_missing_name(self):
        tm = ToolManager({'mock': MockTool()})
        result = tm({'function': {}})
        assert 'missing "function.name"' in result

    def test_call_string_arguments(self):
        calc = CalculatorTool()
        tm = ToolManager({'calculator': calc})
        result = tm({'function': {'name': 'calculator', 'arguments': '{"a": 5, "b": 3, "operation": "add"}'}})
        assert result == '8'

    def test_call_invalid_json_string(self):
        calc = CalculatorTool()
        tm = ToolManager({'calculator': calc})
        result = tm({'function': {'name': 'calculator', 'arguments': 'not json'}})
        assert 'invalid JSON' in result

    def test_call_empty_json_string(self):
        calc = CalculatorTool()
        tm = ToolManager({'calculator': calc})
        result = tm({'function': {'name': 'calculator', 'arguments': '   '}})
        assert result == '0'  # default values

    def test_call_invalid_argument_type(self):
        tm = ToolManager({'mock': MockTool()})
        result = tm({'function': {'name': 'mock', 'arguments': 42}})
        assert 'must be a JSON string or object' in result

    def test_call_tool_exception(self):
        fail = FailingTool()
        tm = ToolManager({'failing_tool': fail})
        result = tm({'function': {'name': 'failing_tool', 'arguments': {}}})
        assert 'Error' in result
        assert 'intentional failure' in result

    def test_call_none_arguments(self):
        mock = MockTool()
        tm = ToolManager({'mock': mock})
        result = tm({'function': {'name': 'mock', 'arguments': None}})
        assert 'executed mock with {}' == result

    def test_call_tool_call_is_not_dict(self):
        tm = ToolManager()
        result = tm('not a dict')
        assert 'tool_call must be an object' in result

    def test_from_list_tool_without_tool_info(self):
        class BadTool(Tool):
            def __call__(self, *args, **kwargs):
                return ''
            def tool_info(self):
                return {}  # missing function.name
        with pytest.raises(ValueError, match='non-empty'):
            ToolManager([BadTool()])
