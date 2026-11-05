# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import twinkle.utils.platforms.base as platforms_base
from twinkle.utils.platforms import metax
from twinkle.utils.platforms.base import Platform


def _only_mx_smi(name):
    return '/usr/bin/mx-smi' if name == 'mx-smi' else None


def test_is_metax_available(monkeypatch):
    monkeypatch.setattr(metax.shutil, 'which', _only_mx_smi)
    assert metax.is_metax_available() is True
    monkeypatch.setattr(metax.shutil, 'which', lambda name: None)
    assert metax.is_metax_available() is False


def test_get_platform_autodetects_metax(monkeypatch):
    monkeypatch.setattr(platforms_base.shutil, 'which', _only_mx_smi)
    assert Platform.get_platform() is metax.Metax


def test_get_platform_explicit_names():
    assert Platform.get_platform('MXC') is metax.Metax
    assert Platform.get_platform('METAX') is metax.Metax
    assert Platform.get_platform('mxc') is metax.Metax


def test_metax_inherits_gpu_behavior():
    from twinkle.utils.platforms.gpu import GPU

    assert issubclass(metax.Metax, GPU)
    assert metax.Metax.visible_device_env() == 'CUDA_VISIBLE_DEVICES'
    assert metax.Metax.device_prefix() == 'cuda'
    assert metax.Metax.device_backend() == 'nccl'


def test_ensure_vllm_metax_patch_noop_without_package():
    # vllm_metax is not installed on generic hosts: the call must not raise.
    metax.ensure_vllm_metax_patch()
    assert 'vllm_metax.patch' not in sys.modules


def test_apply_vllm_patch_dispatches_to_platform(monkeypatch):
    """The generic entry dispatches to the detected platform's hook (metax),
    and stays a no-op on platforms without one (gpu)."""
    calls = []

    def fake_hook():
        calls.append('metax')

    monkeypatch.setattr(metax.Metax, 'ensure_vllm_patch', staticmethod(fake_hook))
    monkeypatch.setattr(platforms_base.shutil, 'which', _only_mx_smi)
    Platform.apply_vllm_patch()
    assert calls == ['metax']

    from twinkle.utils.platforms.gpu import GPU

    monkeypatch.setattr(platforms_base.shutil, 'which', lambda name: None)
    Platform.apply_vllm_patch()
    assert calls == ['metax']  # GPU hook is the base no-op
    assert GPU.ensure_vllm_patch is Platform.ensure_vllm_patch


def test_ensure_vllm_metax_patch_applies_once(monkeypatch):
    fake_pkg = ModuleType('vllm_metax')
    fake_pkg.__spec__ = ModuleSpec('vllm_metax', None)
    monkeypatch.setitem(sys.modules, 'vllm_metax', fake_pkg)

    calls = []

    def fake_import_module(name):
        assert name == 'vllm_metax.patch'
        calls.append(name)
        module = ModuleType(name)
        sys.modules[name] = module
        return module

    monkeypatch.setattr(importlib, 'import_module', fake_import_module)

    metax.ensure_vllm_metax_patch()
    metax.ensure_vllm_metax_patch()
    assert len(calls) == 1
    assert 'vllm_metax.patch' in sys.modules
