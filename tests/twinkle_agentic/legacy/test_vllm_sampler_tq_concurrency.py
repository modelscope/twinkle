# Copyright (c) Model Scope Contributors. All rights reserved.
"""Archived: the TQ-class concurrency test, moved with the TransferQueue runtime it asserts on."""
import pytest

def test_server_concurrency_does_not_change_native_async_rl_defaults(monkeypatch):
    pytest.importorskip('transfer_queue')
    import twinkle.infra as infra
    from twinkle.infra._ray import RayHelper
    from twinkle.sampler.vllm_sampler.vllm_sampler import SAMPLER_MAX_CONCURRENCY
    from twinkle.server.sampler.backends.vllm_async import ServerVLLMSampler
    from twinkle.utils.device_mesh import DeviceMesh
    from twinkle_agentic.async_rl.legacy.vllm_sampler_tq import VLLMSamplerTQ

    captured = []
    def create(cls, *args, **kwargs):
        captured.append((cls.__name__, kwargs['max_concurrency']))
        return []
    monkeypatch.setattr(infra, '_mode', 'ray')
    monkeypatch.setattr(infra, '_device_group', [])
    monkeypatch.setattr(RayHelper, 'create_workers', create)
    monkeypatch.delenv('WORKER_NAME', raising=False)
    monkeypatch.delenv('CLUSTER_NAME', raising=False)
    monkeypatch.setenv('TWINKLE_TRUST_REMOTE_CODE', '1')
    mesh = object.__new__(DeviceMesh)
    for cls in (VLLMSamplerTQ, ServerVLLMSampler):
        cls(model_id='test-model', context_manager=None, device_mesh=mesh, remote_group='test-sampler')
    assert captured == [('VLLMSamplerTQ', None), ('ServerVLLMSampler', SAMPLER_MAX_CONCURRENCY)]
