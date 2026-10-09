# Copyright (c) ModelScope Contributors. All rights reserved.
"""Focused unit contracts for the auto-agent tool dispatcher."""
from __future__ import annotations

import json
import pytest

from twinkle_client.auto.agent.tool_schemas import TOOL_SCHEMAS as DECLARED_TOOL_SCHEMAS
from twinkle_client.auto.agent.tools import TOOL_SCHEMAS, ToolExecutor


class _Connection:
    current_run_id: str | None = None

    def list_training_runs(self):
        return [{'run_id': 'run-1'}]


@pytest.mark.asyncio
async def test_execute_dispatches_and_serializes_result():
    result = json.loads(await ToolExecutor(_Connection()).execute('list_training_runs', {}))
    assert result == [{'run_id': 'run-1'}]


@pytest.mark.asyncio
async def test_execute_reports_unknown_tool_without_raising():
    result = json.loads(await ToolExecutor(_Connection()).execute('missing', {}))
    assert result == {'error': 'Unknown tool: missing'}


@pytest.mark.asyncio
async def test_execute_turns_handler_exception_into_error(monkeypatch):
    executor = ToolExecutor(_Connection())

    async def fail():
        raise RuntimeError('broken')

    monkeypatch.setattr(executor, '_tool_list_training_runs', fail)
    result = json.loads(await executor.execute('list_training_runs', {}))
    assert result == {'error': 'list_training_runs failed: broken'}


@pytest.mark.asyncio
async def test_search_dispatch_stays_behind_executor(monkeypatch):
    executor = ToolExecutor(_Connection())
    monkeypatch.setattr(executor, '_search_datasets_impl', lambda query, limit: [{'id': query, 'limit': limit}])
    result = await executor._tool_search_datasets('demo', limit=2)
    assert result == {
        'query': 'demo',
        'results': [{
            'id': 'demo',
            'limit': 2
        }],
    }


@pytest.mark.asyncio
async def test_server_health_helper_stays_behind_executor(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, 'urlopen', lambda *args, **kwargs: object())
    assert await ToolExecutor(_Connection())._check_server_health('http://server') is True


def test_tool_schema_names_are_unique_and_dispatchable():
    assert TOOL_SCHEMAS is DECLARED_TOOL_SCHEMAS
    names = [item['function']['name'] for item in TOOL_SCHEMAS]
    assert len(names) == len(set(names))
    assert all(hasattr(ToolExecutor, f'_tool_{name}') for name in names)
