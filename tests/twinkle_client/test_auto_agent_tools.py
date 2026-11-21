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
    application = next(app for app in config.applications if app.import_path == 'model')
    assert application.name == 'models-student'
    assert application.args.train_mode == 'lora'
    assert application.args.adapter_config.adapter_timeout == 600
    model = config.training_config.models[0]
    assert model.env_vars['TWINKLE_TRUST_REMOTE_CODE'] == '1'
    assert model.mode == 'lora'
    assert model.config.min_replicas == 0 and model.config.max_replicas == 1
    assert model.config.num_devices == 2 and model.config.idle_timeout_s == 600
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


def test_single_device_megatron_config_is_not_written(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    with pytest.raises(ValueError, match='at least 2 devices'):
        ToolExecutor._generate_server_config('Qwen/student', train_gpus=1, backend='megatron')
    assert not (tmp_path / '.cache' / 'twinkle' / 'server_config.yaml').exists()


@pytest.mark.parametrize('backend', ['torch', 'missing'])
def test_unsupported_training_backend_is_not_written(tmp_path, monkeypatch, backend):
    from pathlib import Path

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    with pytest.raises(ValueError, match='Unsupported training backend'):
        ToolExecutor._generate_server_config('Qwen/student', train_gpus=1, backend=backend)
    assert not (tmp_path / '.cache' / 'twinkle' / 'server_config.yaml').exists()


def test_detect_accelerator_cuda_via_nvidia_smi(monkeypatch):
    import subprocess
    from types import SimpleNamespace

    monkeypatch.setattr(
        subprocess,
        'run',
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout='0\n1\n2\n3\n', stderr=''),
    )
    assert ToolExecutor._detect_accelerator() == ('cuda', 4)
    assert ToolExecutor._detect_gpu_count() == 4


def test_detect_accelerator_npu_via_torch(monkeypatch):
    import subprocess
    import torch
    from types import SimpleNamespace

    def _no_nvidia(cmd, *args, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, 'run', _no_nvidia)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(
        torch,
        'npu',
        SimpleNamespace(is_available=lambda: True, device_count=lambda: 8),
        raising=False,
    )
    assert ToolExecutor._detect_accelerator() == ('npu', 8)
    assert ToolExecutor._detect_gpu_count() == 8


def test_detect_accelerator_npu_via_npu_smi_fallback(monkeypatch):
    import subprocess
    import torch
    from types import SimpleNamespace

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch, 'npu', None, raising=False)

    single_chip_out = (
        '\tTotal Count                    : 8\n\n'
        + '\n'.join(f'\tNPU ID                         : {i}\n\tChip Count                     : 1\n' for i in range(8))
    )
    dual_chip_out = (
        '\tTotal Count                    : 4\n\n'
        + '\n'.join(f'\tNPU ID                         : {i}\n\tChip Count                     : 2\n' for i in range(4))
    )

    for smi_stdout in (single_chip_out, dual_chip_out):
        def _run(cmd, *args, _out=smi_stdout, **kwargs):
            if cmd[0] == 'nvidia-smi':
                raise FileNotFoundError('nvidia-smi')
            if cmd[:3] == ['npu-smi', 'info', '-l']:
                return SimpleNamespace(returncode=0, stdout=_out, stderr='')
            raise AssertionError(f'Unexpected command: {cmd}')

        monkeypatch.setattr(subprocess, 'run', _run)
        assert ToolExecutor._detect_accelerator() == ('npu', 8)
        assert ToolExecutor._detect_gpu_count() == 8

    def _missing_all(cmd, *args, **kwargs):
        raise FileNotFoundError(cmd[0])

    monkeypatch.setattr(subprocess, 'run', _missing_all)
    assert ToolExecutor._detect_accelerator() == ('cuda', 0)
    assert ToolExecutor._detect_gpu_count() == 0


@pytest.mark.parametrize('backend', ['transformers', 'megatron'])
@pytest.mark.parametrize('samplers', [[], [{'model_id': 'Qwen/teacher', 'engine': 'vllm', 'tp': 2, 'dp': 2}]])
def test_generated_server_config_supports_npu_device_type(tmp_path, monkeypatch, backend, samplers):
    from pathlib import Path
    from twinkle.server.config import ServerConfig

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    path = ToolExecutor._generate_server_config(
        'Qwen/student',
        train_gpus=2,
        backend=backend,
        samplers=samplers,
        device_type='npu',
    )
    config = ServerConfig.from_yaml(path)
    model_app = next(app for app in config.applications if app.import_path == 'model')
    assert model_app.args.device_group['device_type'] == 'npu'
    assert model_app.args.device_mesh['device_type'] == 'npu'
    assert config.training_config.models[0].device_type == 'npu'
    proc_app = next(app for app in config.applications if app.import_path == 'processor')
    assert proc_app.args.device_group['device_type'] == 'CPU'
    if samplers:
        sampler_app = next(app for app in config.applications if app.import_path == 'sampler')
        assert sampler_app.args.device_group['device_type'] == 'npu'
        assert sampler_app.args.device_mesh['device_type'] == 'npu'


def test_start_ray_cluster_npu_and_cuda_args(monkeypatch, tmp_path):
    import subprocess
    import tempfile
    from types import SimpleNamespace

    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path))
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)

    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append((list(cmd), kwargs.get('env')))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(subprocess, 'run', _fake_run)

    assert ToolExecutor._start_ray_cluster(2, 2, device_type='npu') is None
    npu_starts = [(cmd, env) for cmd, env in calls if cmd[:2] == ['ray', 'start']]
    assert len(npu_starts) == 3

    head_cmd, head_env = npu_starts[0]
    assert '--head' in head_cmd
    assert not any(arg.startswith('--num-gpus') for arg in head_cmd)
    head_res = [arg.split('=', 1)[1] for arg in head_cmd if arg.startswith('--resources=')]
    assert len(head_res) == 1 and json.loads(head_res[0]) == {'NPU': 2}
    assert head_env is not None and head_env['ASCEND_RT_VISIBLE_DEVICES'] == '0,1'
    assert 'CUDA_VISIBLE_DEVICES' not in head_env

    worker_cmd, worker_env = npu_starts[1]
    assert not any(arg.startswith('--num-gpus') for arg in worker_cmd)
    worker_res = [arg.split('=', 1)[1] for arg in worker_cmd if arg.startswith('--resources=')]
    assert len(worker_res) == 1 and json.loads(worker_res[0]) == {'NPU': 2}
    assert worker_env is not None and worker_env['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'

    cpu_cmd, cpu_env = npu_starts[2]
    assert not any(arg.startswith('--num-gpus') for arg in cpu_cmd)
    assert not any(arg.startswith('--resources=') for arg in cpu_cmd)
    assert cpu_env is not None and cpu_env['ASCEND_RT_VISIBLE_DEVICES'] == ''

    calls.clear()
    assert ToolExecutor._start_ray_cluster(2, 2, device_type='cuda') is None
    cuda_starts = [(cmd, env) for cmd, env in calls if cmd[:2] == ['ray', 'start']]
    assert len(cuda_starts) == 3
    assert '--num-gpus=2' in cuda_starts[0][0]
    assert cuda_starts[0][1] is not None and cuda_starts[0][1]['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert '--num-gpus=2' in cuda_starts[1][0]
    assert cuda_starts[1][1] is not None and cuda_starts[1][1]['CUDA_VISIBLE_DEVICES'] == '2,3'
    assert '--num-gpus=0' in cuda_starts[2][0]
    assert cuda_starts[2][1] is not None and cuda_starts[2][1]['CUDA_VISIBLE_DEVICES'] == ''


@pytest.mark.asyncio
async def test_start_server_wires_detected_npu_end_to_end(monkeypatch):
    executor = ToolExecutor(_Connection())
    captured: dict[str, object] = {}

    async def _not_running(url):
        return False

    def _fake_config(model_id, train_gpus, port=8000, backend='transformers', samplers=None, device_type='cuda'):
        captured['config_device_type'] = device_type
        captured['config_train_gpus'] = train_gpus
        return '/tmp/fake_server_config.yaml'

    def _fake_ray(train_gpus, sampler_gpus, device_type='cuda'):
        captured['ray_device_type'] = device_type
        captured['ray_train_gpus'] = train_gpus
        return None

    monkeypatch.setattr(executor, '_check_server_health', _not_running)
    monkeypatch.setattr(ToolExecutor, '_detect_accelerator', staticmethod(lambda: ('npu', 4)))
    monkeypatch.setattr(ToolExecutor, '_generate_server_config', staticmethod(_fake_config))
    monkeypatch.setattr(ToolExecutor, '_start_ray_cluster', staticmethod(_fake_ray))
    monkeypatch.setattr(
        ToolExecutor,
        '_launch_server_process',
        staticmethod(lambda config_path: (object(), '/tmp/fake_server.log', None)),
    )
    monkeypatch.setattr(
        ToolExecutor,
        '_wait_server_ready',
        staticmethod(lambda **kwargs: {'status': 'started', 'server_url': kwargs['server_url']}),
    )

    result = await executor._tool_start_server('Qwen/Qwen2.5-0.5B', train_gpus=2)
    assert result['status'] == 'started'
    assert captured == {
        'config_device_type': 'npu',
        'config_train_gpus': 2,
        'ray_device_type': 'npu',
        'ray_train_gpus': 2,
    }
