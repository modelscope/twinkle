"""CPU-only coverage for the device-neutral parts of the IPC checkpoint engine."""

from unittest.mock import Mock

import pytest
import torch

import twinkle.checkpoint_engine.ipc_checkpoint_engine as ipc_module
from twinkle.checkpoint_engine import IPCCheckpointEngine


def test_endpoint_uses_platform_uuid_without_touching_cuda(monkeypatch):
    """Endpoint construction must not query the current CUDA device."""
    monkeypatch.setattr(torch.cuda, 'current_device', Mock(side_effect=AssertionError('CUDA initialized')))
    monkeypatch.setattr(torch.cuda, 'get_device_properties', Mock(side_effect=AssertionError('CUDA initialized')))
    monkeypatch.setattr(ipc_module.Torch, 'get_current_device', lambda: 0)
    monkeypatch.setattr(
        ipc_module.Platform,
        'get_vllm_device_uuid',
        staticmethod(lambda device_id=0, platform=None: f'uuid-{device_id}'),
    )

    assert IPCCheckpointEngine.endpoint() == 'ipc:///tmp/twinkle-colocate-uuid-0.sock'


def test_map_uses_reducer_callable_and_receiver_device(monkeypatch):
    calls = []

    def rebuild(*args):
        calls.append(args)
        return 'mapped'

    monkeypatch.setattr(ipc_module.Platform, 'device_prefix', staticmethod(lambda: 'cuda'))
    monkeypatch.setattr(ipc_module.Torch, 'get_current_device', lambda: 3)
    sender_args = [None, None, None, None, None, None, 17]

    engine = IPCCheckpointEngine()
    assert engine._map((rebuild, sender_args)) == 'mapped'
    assert calls[0][6] == 3
    assert sender_args[6] == 17


def test_vllm_worker_uses_reducer_callable_and_receiver_device():
    from twinkle.sampler.vllm_sampler.vllm_worker_extension import _rebuild_ipc

    calls = []

    def rebuild(*args):
        calls.append(args)
        return 'mapped'

    sender_args = [None, None, None, None, None, None, 17]

    assert _rebuild_ipc((rebuild, sender_args), device_id=3) == 'mapped'
    assert calls[0][6] == 3
    assert sender_args[6] == 17


def test_buffer_and_flush_use_framework_device_and_sync_helpers(monkeypatch):
    class CPU:
        @staticmethod
        def device_prefix():
            return 'cpu'

    monkeypatch.setattr(ipc_module.Platform, 'get_platform', staticmethod(lambda platform=None: CPU))
    monkeypatch.setattr(ipc_module.Torch, 'get_current_device', lambda: 0)

    from torch.multiprocessing import reductions

    monkeypatch.setattr(reductions, 'reduce_tensor', lambda tensor: (None, [None]))
    engine = IPCCheckpointEngine(bucket_size=8)
    engine._ensure_buffer(8)
    assert engine.send_buf.device.type == 'cpu'

    class Socket:
        def send(self, payload):
            self.payload = payload

        def recv(self):
            return b'ack'

    engine.socket = Socket()
    synchronize = Mock()
    monkeypatch.setattr(ipc_module.Torch, 'synchronize', synchronize)
    engine._flush([], is_last=True)
    synchronize.assert_called_once_with()


def test_npu_without_device_ipc_uses_shared_memory_and_closes_it(monkeypatch):
    class NPU:
        @staticmethod
        def device_prefix():
            return 'npu'

        @staticmethod
        def is_ipc_supported():
            return False

    monkeypatch.setattr(ipc_module.Platform, 'get_platform', staticmethod(lambda platform=None: NPU))
    monkeypatch.setattr(ipc_module.Torch, 'get_current_device', lambda: 0)

    sender = IPCCheckpointEngine(bucket_size=8)
    sender._ensure_buffer(8)
    handle = sender._handle
    assert set(handle) == {'name', 'size'}
    assert sender.send_buf.device.type == 'cpu'

    receiver = IPCCheckpointEngine()
    mapped = receiver._map(handle)
    assert mapped.device.type == 'cpu'
    assert mapped.numel() == 8

    sender.finalize()
    del mapped
    receiver.finalize()

    from multiprocessing import shared_memory

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=handle['name'])


def test_shared_memory_growth_keeps_the_previous_receiver_mapping_alive(monkeypatch):
    class NPU:
        @staticmethod
        def device_prefix():
            return 'npu'

        @staticmethod
        def is_ipc_supported():
            return False

    monkeypatch.setattr(ipc_module.Platform, 'get_platform', staticmethod(lambda platform=None: NPU))

    sender = IPCCheckpointEngine(bucket_size=8)
    receiver = IPCCheckpointEngine()
    sender._ensure_buffer(8)
    first = receiver._map(sender._handle)
    first[0] = 7

    sender._ensure_buffer(16)
    second = receiver._map(sender._handle)

    assert first[0].item() == 7
    assert second.numel() == 16

    del first, second
    sender.finalize()
    receiver.finalize()
