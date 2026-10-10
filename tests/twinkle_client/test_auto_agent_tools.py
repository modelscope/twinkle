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


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
@pytest.mark.parametrize('samplers', [[], [{'model_id': 'Qwen/teacher', 'engine': 'vllm', 'tp': 2, 'dp': 2}]])
def test_generated_server_config_matches_launcher_schema(tmp_path, monkeypatch, backend, samplers):
    from pathlib import Path
    from twinkle.server.config import ServerConfig

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    path = ToolExecutor._generate_server_config('Qwen/student', train_gpus=2, backend=backend, samplers=samplers)
    config = ServerConfig.from_yaml(path)
    gateway = next(app for app in config.applications if app.import_path == 'server')
    assert gateway.deployments[0]['name'] == 'GatewayServer'
    assert gateway.args.supported_models == ['Qwen/student'] + [s['model_id'] for s in samplers]
    if samplers:
        sampler = next(app for app in config.applications if app.import_path == 'sampler')
        assert sampler.args.nproc_per_node == 4
        assert sampler.args.device_mesh['tp_size'] == 2
        assert sampler.args.device_mesh['dp_size'] == 2
        assert sampler.args.engine_args['tensor_parallel_size'] == 2


def test_removed_sampler_is_not_advertised_or_written(tmp_path, monkeypatch):
    from pathlib import Path

    start = next(tool['function'] for tool in TOOL_SCHEMAS if tool['function']['name'] == 'start_server')
    engines = start['parameters']['properties']['samplers']['items']['properties']['engine']['enum']
    assert 'torch' not in engines
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    with pytest.raises(ValueError, match='Unsupported sampler engine'):
        ToolExecutor._generate_server_config(
            'Qwen/student', train_gpus=1, samplers=[{'model_id': 'Qwen/teacher', 'engine': 'torch'}])
    assert not (tmp_path / '.cache' / 'twinkle' / 'server_config.yaml').exists()


@pytest.mark.parametrize('backend', ['torch', 'missing'])
def test_unsupported_training_backend_is_not_written(tmp_path, monkeypatch, backend):
    from pathlib import Path

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    with pytest.raises(ValueError, match='Unsupported training backend'):
        ToolExecutor._generate_server_config('Qwen/student', train_gpus=1, backend=backend)
    assert not (tmp_path / '.cache' / 'twinkle' / 'server_config.yaml').exists()
