# Copyright (c) ModelScope Contributors. All rights reserved.
"""CUDA and Ascend must use the same device-paired direct transport."""
import asyncio
import threading
import uuid
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin
from twinkle.utils import Platform
from twinkle.utils.framework import Torch


@pytest.mark.parametrize('prefix', ['cuda', 'npu'])
def test_direct_identity_uses_the_platform_device_without_cuda_probes(monkeypatch, prefix):
    monkeypatch.setattr(Platform, 'get_platform', lambda: NS(
        device_prefix=lambda: prefix, device_module=lambda: object()))
    monkeypatch.setattr(Torch, 'get_current_device', lambda: 3)
    identity = Mock(return_value='physical-device-13')
    monkeypatch.setattr(Platform, 'get_vllm_device_uuid', identity)
    monkeypatch.setattr(torch.cuda, 'is_available', Mock(side_effect=AssertionError('CUDA probe')))
    monkeypatch.setattr(torch.cuda, 'current_device', Mock(side_effect=AssertionError('CUDA probe')))

    assert CheckpointEngineMixin.direct_weight_sync_device.__wrapped__(NS()) == 'physical-device-13'
    identity.assert_called_once_with(3)


@pytest.mark.parametrize('prefix,module', [('cuda', None), ('npu', None), ('xpu', object())])
def test_direct_identity_keeps_unavailable_or_other_platforms_on_existing_transport(monkeypatch, prefix, module):
    monkeypatch.setattr(Platform, 'get_platform', lambda: NS(
        device_prefix=lambda: prefix, device_module=lambda: module))
    assert CheckpointEngineMixin.direct_weight_sync_device.__wrapped__(NS()) is None


@pytest.mark.parametrize('prefix', ['cuda', 'npu'])
def test_sender_binds_its_thread_to_the_actual_accelerator(monkeypatch, prefix):
    from twinkle.sampler.vllm_sampler import weight_transfer

    events = []
    device = NS(type=prefix, index=2)
    monkeypatch.setattr(torch, 'device', lambda kind, index: device if (kind, index) == (prefix, 2) else None)
    monkeypatch.setattr(Platform, 'device_prefix', lambda: prefix)
    monkeypatch.setattr(Torch, 'get_current_device', lambda: 2)
    monkeypatch.setattr(Torch, 'set_device', lambda value: events.append(('bind', value, threading.get_ident())))
    monkeypatch.setattr(torch.cuda, 'current_device', Mock(side_effect=AssertionError('CUDA probe')))

    class Sender:
        async def update_weights(self, weights, **kwargs):
            events.append(('send', kwargs, threading.get_ident()))
            assert list(weights) == [('weight', 'tensor')]

    monkeypatch.setattr(weight_transfer, 'VLLMWeightSender', Sender)
    actor = NS(direct_weight_sync_device=lambda: 'device-13',
               _get_weight_generator=lambda **kwargs: iter([('weight', 'tensor')]))
    CheckpointEngineMixin.send_weights_to_vllm(
        actor, [{'device_uuid': 'device-13', 'endpoint': 'ipc:///paired.sock'}], auth_key='key')

    assert [event[0] for event in events] == ['bind', 'send']
    assert events[0][1] is device
    assert events[1][1]['device'] is device
    assert events[1][1]['zmq_handle'] == 'ipc:///paired.sock'
    assert events[1][1]['auth_key'] == 'key'
    assert events[1][1]['direct_large_tensors'] is True
    assert events[0][2] == events[1][2] != threading.get_ident()


@pytest.mark.parametrize('device_ipc', [False, True])
def test_receiver_uses_the_platform_ipc_capability(monkeypatch, device_ipc):
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import TwinkleWorkerExtension

    monkeypatch.setattr(Platform, 'get_platform', lambda: NS(supports_device_ipc=lambda: device_ipc))
    receive = Mock()
    worker = NS(weight_sync_metadata=lambda: {'device_uuid': 'npu13'}, update_weights_from_ipc=receive)
    TwinkleWorkerExtension.update_weights_direct(
        worker, [{'device_uuid': 'npu6', 'endpoint': 'wrong'}, {'device_uuid': 'npu13', 'endpoint': 'paired'}],
        auth_key='key', base_sync_done=True, peft_config={'r': 8})
    receive.assert_called_once_with(
        peft_config={'r': 8}, zmq_handle='paired', auth_key='key',
        base_sync_done=True, independent=True, use_shm=not device_ipc)


def test_direct_shm_fallback_transfers_multiple_buckets_and_publications(monkeypatch):
    """Exercise the NPU fallback wire protocol with real ZMQ/SHM on a CPU runner."""
    from twinkle.sampler.vllm_sampler import vllm_worker_extension as extension
    from twinkle.sampler.vllm_sampler.weight_transfer import VLLMWeightSender

    monkeypatch.setenv('TWINKLE_VLLM_IPC_TIMEOUT_S', '5')
    monkeypatch.setattr(Platform, 'get_platform', lambda: NS(supports_device_ipc=lambda: False))
    monkeypatch.setattr(Platform, 'get_vllm_device_uuid', lambda *args: 'physical-device')
    monkeypatch.setattr(Torch, 'synchronize', lambda *args: None)
    monkeypatch.setattr(Torch, 'ipc_collect', lambda: None)
    monkeypatch.setattr('twinkle.patch.vllm_weight_loader.patch_lora_weight_loader', lambda: None)
    # Darwin limits POSIX SHM names to 31 bytes, below the production UUID length.
    tag = uuid.uuid4().hex[:8]
    monkeypatch.setattr(uuid, 'uuid4', lambda: NS(hex=tag))

    class Receiver(extension.TwinkleWorkerExtension):
        def _load_weights(self, weights, **kwargs):
            self.loaded.update((name, tensor.clone()) for name, tensor in weights)

    worker = object.__new__(Receiver)
    worker.device = torch.device('cpu')
    worker.rank = worker.local_rank = 0
    worker.model_runner = NS(parallel_config=NS(tensor_parallel_size=2))
    sender = VLLMWeightSender()
    for step in range(2):
        endpoint = f'ipc:///tmp/twinkle-direct-test-{tag}-{step}.sock'
        target = {'device_uuid': 'physical-device', 'endpoint': endpoint}
        worker._weight_sync_target = target
        worker.loaded = {}
        errors = []
        weights = {'large': torch.full((300_000,), step + 1, dtype=torch.float32),
                   'small': torch.arange(7) + step}

        def receive():
            try:
                worker.update_weights_direct([target], auth_key='key')
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=receive, daemon=True)
        thread.start()
        try:
            asyncio.run(sender.update_weights(
                weights, bucket_size_mb=1, zmq_handle=endpoint, auth_key='key',
                device=torch.device('cpu'), direct_large_tensors=True))
        finally:
            thread.join(timeout=10)
        assert not thread.is_alive(), 'Direct receiver failed to complete'
        assert not errors, errors
        assert worker.loaded.keys() == weights.keys()
        for name, expected in weights.items():
            assert torch.equal(worker.loaded[name], expected)
