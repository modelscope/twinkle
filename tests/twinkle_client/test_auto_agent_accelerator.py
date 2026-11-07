# Copyright (c) ModelScope Contributors. All rights reserved.
"""`start_server` must follow the accelerator it is actually running on.

The Auto server tool used to hard-code CUDA: it counted devices with
`nvidia-smi` only and wrote `device_type: cuda` into every generated device
group. On an Ascend NPU host that meant `start_server` reported "No GPUs
detected" without ever reading a config, and any config it did write asked for
CUDA devices.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from twinkle_client.auto.agent.tools import ToolExecutor


class _FakeTorchUtil:

    def __init__(self, gpu: bool = False, npu: bool = False):
        self._gpu = gpu
        self._npu = npu

    def is_gpu_available(self):
        return self._gpu

    def is_npu_available(self):
        return self._npu


class _FakeTorch:
    """Minimal `torch` stand-in: only the device counts are read."""

    class cuda:

        @staticmethod
        def device_count():
            return 2

    class npu:

        @staticmethod
        def device_count():
            return 8


@pytest.fixture
def fake_accelerator(monkeypatch):
    """Install a fake `twinkle.torch_util` + `torch` with a chosen accelerator."""

    def install(*, gpu: bool = False, npu: bool = False):
        import sys
        import types

        monkeypatch.setitem(sys.modules, 'torch', _FakeTorch)
        monkeypatch.setattr('twinkle.torch_util', _FakeTorchUtil(gpu=gpu, npu=npu), raising=True)
        return types

    return install


def test_detect_accelerators_prefers_npu_over_nvidia_smi(monkeypatch):
    """An NPU host has no nvidia-smi answer; it must still report its devices."""
    import sys
    import types

    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch)
    monkeypatch.setattr('twinkle.torch_util', _FakeTorchUtil(npu=True), raising=True)

    nvidia_smi_called = []
    fake_subprocess = types.ModuleType('subprocess')

    def run(cmd, **kwargs):
        nvidia_smi_called.append(cmd)
        raise FileNotFoundError('nvidia-smi')

    fake_subprocess.run = run
    monkeypatch.setitem(sys.modules, 'subprocess', fake_subprocess)

    assert ToolExecutor._detect_accelerators() == ('npu', 8)
    assert not nvidia_smi_called, 'nvidia-smi must not be consulted when torch sees NPUs'


def test_detect_accelerators_reports_cuda(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch)
    monkeypatch.setattr('twinkle.torch_util', _FakeTorchUtil(gpu=True), raising=True)
    assert ToolExecutor._detect_accelerators() == ('cuda', 2)


def test_detect_accelerators_falls_back_to_nvidia_smi(monkeypatch):
    """No accelerator visible to torch: keep the original nvidia-smi probe."""
    import sys

    monkeypatch.setitem(sys.modules, 'torch', _FakeTorch)
    monkeypatch.setattr('twinkle.torch_util', _FakeTorchUtil(), raising=True)
    monkeypatch.setattr(ToolExecutor, '_detect_gpu_count', staticmethod(lambda: 4))
    assert ToolExecutor._detect_accelerators() == ('cuda', 4)


def test_npu_config_carries_the_detected_device_type(tmp_path, monkeypatch):
    from twinkle.server.config import ServerConfig

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    path = ToolExecutor._generate_server_config(
        'Qwen/student',
        train_gpus=2,
        backend='transformers',
        samplers=[{
            'model_id': 'Qwen/teacher',
            'engine': 'vllm',
            'tp': 2,
            'dp': 1
        }],
        device_type='npu',
    )
    config = ServerConfig.from_yaml(path)
    model = next(app for app in config.applications if app.import_path == 'model')
    assert model.args.device_group['device_type'] == 'npu'
    assert model.args.device_mesh['device_type'] == 'npu'
    sampler = next(app for app in config.applications if app.import_path == 'sampler')
    assert sampler.args.device_group['device_type'] == 'npu'
    assert sampler.args.device_mesh['device_type'] == 'npu'


def test_config_device_type_defaults_to_cuda(tmp_path, monkeypatch):
    from twinkle.server.config import ServerConfig

    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    path = ToolExecutor._generate_server_config('Qwen/student', train_gpus=1)
    config = ServerConfig.from_yaml(path)
    model = next(app for app in config.applications if app.import_path == 'model')
    assert model.args.device_group['device_type'] == 'cuda'


def test_ray_cluster_registers_npu_custom_resource(monkeypatch, tmp_path):
    """Ray cannot discover NPUs; they must be published as a custom resource."""
    import sys
    import types

    commands = []

    def run(cmd, **kwargs):
        commands.append((list(cmd), dict(kwargs.get('env') or {})))
        return types.SimpleNamespace(returncode=0, stderr='')

    fake_subprocess = types.ModuleType('subprocess')
    fake_subprocess.run = run
    monkeypatch.setitem(sys.modules, 'subprocess', fake_subprocess)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)

    assert ToolExecutor._start_ray_cluster(2, 1, device_type='npu') is None
    assert commands[0][0] == ['ray', 'stop', '--force']  # stale cluster is cleared first
    head_cmd, head_env = commands[1]
    assert '--resources={"NPU": 2}' in head_cmd
    assert not any(arg.startswith('--num-gpus') for arg in head_cmd)
    assert head_env['ASCEND_RT_VISIBLE_DEVICES'] == '0,1'
    assert 'CUDA_VISIBLE_DEVICES' not in head_env

    sampler_cmd, sampler_env = commands[2]
    assert '--resources={"NPU": 1}' in sampler_cmd
    assert sampler_env['ASCEND_RT_VISIBLE_DEVICES'] == '2'


def test_ray_cluster_keeps_num_gpus_for_cuda(monkeypatch, tmp_path):
    import sys
    import types

    commands = []

    def run(cmd, **kwargs):
        commands.append((list(cmd), dict(kwargs.get('env') or {})))
        return types.SimpleNamespace(returncode=0, stderr='')

    fake_subprocess = types.ModuleType('subprocess')
    fake_subprocess.run = run
    monkeypatch.setitem(sys.modules, 'subprocess', fake_subprocess)
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)

    assert ToolExecutor._start_ray_cluster(2, 1, device_type='cuda') is None
    head_cmd, head_env = commands[1]
    assert '--num-gpus=2' in head_cmd
    assert head_env['CUDA_VISIBLE_DEVICES'] == '0,1'
    assert 'ASCEND_RT_VISIBLE_DEVICES' not in head_env
