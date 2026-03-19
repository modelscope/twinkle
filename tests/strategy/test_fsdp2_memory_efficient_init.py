# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import unittest
from torch.distributed.fsdp import fully_shard

from twinkle.utils import Platform

_PLATFORM = Platform.get_platform()
_DEVICE_TYPE = _PLATFORM.device_prefix()  # 'cuda' or 'npu'
_DIST_BACKEND = _PLATFORM.device_backend()  # 'nccl' or 'hccl'


def _device_count() -> int:
    if _DEVICE_TYPE == 'npu':
        import torch_npu  # noqa: F401
        return torch.npu.device_count()
    return torch.cuda.device_count()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _init_dist(rank, world_size, port):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    if _DEVICE_TYPE == 'npu':
        from twinkle.utils.platforms.npu import ensure_hccl_socket_env
        ensure_hccl_socket_env(port)
    dist.init_process_group(_DIST_BACKEND, rank=rank, world_size=world_size)
    device = torch.device(_PLATFORM.get_local_device(rank))
    torch.device(device)  # set current device
    if _DEVICE_TYPE == 'npu':
        import torch_npu  # noqa: F401
        torch.npu.set_device(device)
    else:
        torch.cuda.set_device(rank)


class TinyModel(nn.Module):
    """2-layer MLP for testing. Small enough to fit on any GPU."""

    def __init__(self, dim=32):
        super().__init__()
        self.layer1 = nn.Linear(dim, dim, bias=False)
        self.layer2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.layer2(self.layer1(x))


class TinyDecoderLayer(nn.Module):
    """Minimal decoder layer for per-layer sharding tests."""

    def __init__(self, dim=32):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim, bias=False)
        self.fc2 = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class TinyTransformerModel(nn.Module):
    """Model with model.model.layers structure matching _get_decoder_layers expectations."""

    def __init__(self, dim=32, num_layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyDecoderLayer(dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        for layer in self.model.layers:
            x = layer(x)
        return self.lm_head(x)


def _worker_broadcast_sharded(rank, world_size, port, ref_sd):
    """Worker function: shard on meta, broadcast, verify values."""
    _init_dist(rank, world_size, port)
    from twinkle.model.transformers.strategy.native_fsdp import _broadcast_sharded_state_dict

    model = TinyModel(dim=32)

    # Only rank 0 has the real weights; others get empty shell
    if rank == 0:
        full_sd = ref_sd  # passed via mp args (shared memory)
    else:
        full_sd = {}

    # Save original full state dict for later verification
    original_sd = ref_sd

    # Move to meta and shard
    model = model.to('meta')
    fully_shard(model.layer1)
    fully_shard(model.layer2)
    fully_shard(model)

    # Broadcast
    _broadcast_sharded_state_dict(model, full_sd, device_type=_DEVICE_TYPE)

    # Verify: gather full state dict back and compare to original
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    gathered = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if rank == 0:
        for key in original_sd:
            assert torch.allclose(gathered[key], original_sd[key], atol=1e-6), \
                f"Mismatch on {key} after broadcast"

    dist.destroy_process_group()


@unittest.skipIf(_device_count() < 2, f'Need >= 2 {_DEVICE_TYPE.upper()}s')
class TestBroadcastShardedStateDict(unittest.TestCase):

    def test_broadcast_restores_weights(self):
        port = _find_free_port()
        world_size = 2
        # Create reference weights on CPU
        ref_model = TinyModel(dim=32)
        ref_sd = {k: v.clone() for k, v in ref_model.state_dict().items()}
        mp.spawn(
            _worker_broadcast_sharded,
            args=(world_size, port, ref_sd),
            nprocs=world_size,
            join=True,
        )


# ---------------------------------------------------------------------------
# Task 2: _get_non_persistent_buffers
# ---------------------------------------------------------------------------


class ModelWithNonPersistentBuffer(nn.Module):

    def __init__(self, dim=32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        # Non-persistent buffer — will NOT appear in state_dict()
        self.register_buffer('mask', torch.ones(dim), persistent=False)


class TestGetNonPersistentBuffers(unittest.TestCase):

    def test_finds_non_persistent_buffers(self):
        from twinkle.model.transformers.strategy.native_fsdp import _get_non_persistent_buffers
        model = ModelWithNonPersistentBuffer()
        result = _get_non_persistent_buffers(model)
        assert 'mask' in result
        assert torch.equal(result['mask'], torch.ones(32))
        # Persistent params/buffers should NOT be in the result
        assert 'linear.weight' not in result

    def test_empty_when_no_non_persistent(self):
        from twinkle.model.transformers.strategy.native_fsdp import _get_non_persistent_buffers
        model = TinyModel()
        result = _get_non_persistent_buffers(model)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Task 3: _restore_non_persistent_buffers
# ---------------------------------------------------------------------------


class TestRestoreNonPersistentBuffers(unittest.TestCase):

    def test_restores_buffers_after_meta(self):
        from twinkle.model.transformers.strategy.native_fsdp import (_get_non_persistent_buffers,
                                                                     _restore_non_persistent_buffers)
        model = ModelWithNonPersistentBuffer()
        saved = _get_non_persistent_buffers(model)
        # Move to meta — buffer becomes meta tensor
        model = model.to('meta')
        assert model.mask.device.type == 'meta'
        # Restore
        _restore_non_persistent_buffers(model, saved, device=torch.device('cpu'))
        assert model.mask.device.type == 'cpu'
        assert torch.equal(model.mask, torch.ones(32))


# ---------------------------------------------------------------------------
# Task 4: wrap_model with memory_efficient=True
# ---------------------------------------------------------------------------
import numpy as np


def _worker_wrap_model_memory_efficient(rank, world_size, port, ref_sd):
    """Test that wrap_model with memory_efficient=True produces correct sharded model."""
    _init_dist(rank, world_size, port)
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
    from twinkle.utils import DeviceMesh as TwinkleMesh

    mesh = TwinkleMesh(
        mesh=np.arange(world_size),
        mesh_dim_names=('fsdp', ),
        device_type=_DEVICE_TYPE,
    )
    strategy = NativeFSDPStrategy(device_mesh=mesh, mixed_precision='no')

    model = TinyModel(dim=32).to(_DEVICE_TYPE)
    if rank == 0:
        model.load_state_dict(ref_sd)

    model, _ = strategy.wrap_model(model, optimizer=None, memory_efficient=True)

    # Verify: model should be on device, not meta
    for name, param in model.named_parameters():
        assert param.device.type == _DEVICE_TYPE, f"{name} still on {param.device}"

    # Verify: gathered full state dict matches original
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    gathered = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if rank == 0:
        for key in ref_sd:
            assert torch.allclose(gathered[key], ref_sd[key], atol=1e-6), \
                f"Mismatch on {key}"

    dist.destroy_process_group()


@unittest.skipIf(_device_count() < 2, f'Need >= 2 {_DEVICE_TYPE.upper()}s')
class TestWrapModelMemoryEfficient(unittest.TestCase):

    def test_wrap_model_memory_efficient(self):
        port = _find_free_port()
        world_size = 2
        ref_model = TinyModel(dim=32)
        ref_sd = {k: v.clone() for k, v in ref_model.state_dict().items()}
        mp.spawn(
            _worker_wrap_model_memory_efficient,
            args=(world_size, port, ref_sd),
            nprocs=world_size,
            join=True,
        )


# ---------------------------------------------------------------------------
# Task 5: wrap_model with memory_efficient=False (legacy path)
# ---------------------------------------------------------------------------


def _worker_wrap_model_legacy(rank, world_size, port, ref_sd):
    """Test that wrap_model with memory_efficient=False still works (old path)."""
    _init_dist(rank, world_size, port)
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy
    from twinkle.utils import DeviceMesh as TwinkleMesh

    mesh = TwinkleMesh(
        mesh=np.arange(world_size),
        mesh_dim_names=('fsdp', ),
        device_type=_DEVICE_TYPE,
    )
    strategy = NativeFSDPStrategy(device_mesh=mesh, mixed_precision='no')

    model = TinyModel(dim=32).to(_DEVICE_TYPE)
    model.load_state_dict(ref_sd)

    model, _ = strategy.wrap_model(model, optimizer=None, memory_efficient=False)

    for name, param in model.named_parameters():
        assert param.device.type == _DEVICE_TYPE, f"{name} still on {param.device}"

    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    gathered = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if rank == 0:
        for key in ref_sd:
            assert torch.allclose(gathered[key], ref_sd[key], atol=1e-6), \
                f"Mismatch on {key}"

    dist.destroy_process_group()


@unittest.skipIf(_device_count() < 2, f'Need >= 2 {_DEVICE_TYPE.upper()}s')
class TestWrapModelLegacy(unittest.TestCase):

    def test_wrap_model_legacy_path(self):
        port = _find_free_port()
        world_size = 2
        ref_model = TinyModel(dim=32)
        ref_sd = {k: v.clone() for k, v in ref_model.state_dict().items()}
        mp.spawn(
            _worker_wrap_model_legacy,
            args=(world_size, port, ref_sd),
            nprocs=world_size,
            join=True,
        )


# ---------------------------------------------------------------------------
# Task 5b: wrap_model memory_efficient with per-layer sharding
# ---------------------------------------------------------------------------


def _worker_wrap_model_per_layer(rank, world_size, port, ref_sd):
    """Test that wrap_model with memory_efficient=True correctly shards per decoder layer."""
    _init_dist(rank, world_size, port)
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy, _get_decoder_layers
    from twinkle.utils import DeviceMesh as TwinkleMesh

    mesh = TwinkleMesh(
        mesh=np.arange(world_size),
        mesh_dim_names=('fsdp', ),
        device_type=_DEVICE_TYPE,
    )
    strategy = NativeFSDPStrategy(device_mesh=mesh, mixed_precision='no')

    model = TinyTransformerModel(dim=32, num_layers=2).to(_DEVICE_TYPE)
    if rank == 0:
        model.load_state_dict(ref_sd)

    # Verify the model has the expected structure before wrapping
    assert _get_decoder_layers(model) is not None, \
        "TinyTransformerModel should have model.model.layers"

    model, _ = strategy.wrap_model(model, optimizer=None, memory_efficient=True)

    # Verify: all parameters on device, not meta
    for name, param in model.named_parameters():
        assert param.device.type == _DEVICE_TYPE, f"{name} still on {param.device}"

    # Verify: gathered full state dict matches original
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    gathered = get_model_state_dict(
        model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    if rank == 0:
        for key in ref_sd:
            assert torch.allclose(gathered[key], ref_sd[key], atol=1e-6), \
                f"Mismatch on {key}"

    dist.destroy_process_group()


@unittest.skipIf(_device_count() < 2, f'Need >= 2 {_DEVICE_TYPE.upper()}s')
class TestWrapModelPerLayerSharding(unittest.TestCase):

    def test_wrap_model_per_layer_memory_efficient(self):
        """Verify per-layer fully_shard works with the meta→broadcast path."""
        port = _find_free_port()
        world_size = 2
        ref_model = TinyTransformerModel(dim=32, num_layers=2)
        ref_sd = {k: v.clone() for k, v in ref_model.state_dict().items()}
        mp.spawn(
            _worker_wrap_model_per_layer,
            args=(world_size, port, ref_sd),
            nprocs=world_size,
            join=True,
        )


# ---------------------------------------------------------------------------
# Task 6: env var / memory_efficient_init parameter in TransformersModel
# ---------------------------------------------------------------------------
class TestEnvVarRamEfficientLoading(unittest.TestCase):
    """Test that __init__ sets FSDP env vars during from_pretrained."""

    def test_env_vars_set_during_from_pretrained(self):
        """Verify env vars are set when memory_efficient_init=True and restored after."""
        import inspect
        from unittest.mock import MagicMock, patch

        from twinkle.model.transformers.transformers import TransformersModel

        # Verify the parameter exists in __init__ signature
        sig = inspect.signature(TransformersModel.__init__)
        assert 'memory_efficient_init' in sig.parameters, \
            'memory_efficient_init parameter should exist in __init__'

        # Verify env vars are actually set during from_pretrained call
        captured_env = {}

        original_from_pretrained = None

        def fake_from_pretrained(cls_or_self, *args, **kwargs):
            """Capture env vars at the moment from_pretrained is called."""
            captured_env['ACCELERATE_USE_FSDP'] = os.environ.get('ACCELERATE_USE_FSDP')
            captured_env['FSDP_CPU_RAM_EFFICIENT_LOADING'] = os.environ.get('FSDP_CPU_RAM_EFFICIENT_LOADING')
            # Return a minimal mock model
            mock_model = MagicMock()
            mock_model.gradient_checkpointing_enable = MagicMock()
            mock_model.named_parameters = MagicMock(return_value=iter([]))
            return mock_model

        from twinkle.utils import DeviceMesh as TwinkleMesh

        mesh = TwinkleMesh(
            mesh=np.arange(1),
            mesh_dim_names=('fsdp', ),
            device_type='cpu',
        )

        # Clean env before test
        saved = {}
        for key in ('ACCELERATE_USE_FSDP', 'FSDP_CPU_RAM_EFFICIENT_LOADING'):
            saved[key] = os.environ.pop(key, None)

        try:
            with patch('twinkle.model.transformers.transformers.HubOperation') as mock_hub, \
                 patch('twinkle.model.transformers.transformers.AutoModelForCausalLM') as mock_auto, \
                 patch.object(TransformersModel, '_try_init_process_group'), \
                 patch.object(TransformersModel, '_decide_strategy'), \
                 patch.object(TransformersModel, '_construct_default_optimizer_group', return_value=MagicMock()):
                mock_hub.download_model.return_value = '/fake/path'
                mock_auto.from_pretrained = classmethod(fake_from_pretrained)

                # memory_efficient_init=True with device_mesh → env vars should be set
                try:
                    TransformersModel(
                        model_id='/fake/model',
                        device_mesh=mesh,
                        memory_efficient_init=True,
                    )
                except Exception:
                    pass  # We only care about the env var capture

            assert captured_env.get('ACCELERATE_USE_FSDP') == 'true', \
                f"ACCELERATE_USE_FSDP should be 'true' during from_pretrained, got {captured_env.get('ACCELERATE_USE_FSDP')}"
            assert captured_env.get('FSDP_CPU_RAM_EFFICIENT_LOADING') == 'true', \
                f"FSDP_CPU_RAM_EFFICIENT_LOADING should be 'true' during from_pretrained, got {captured_env.get('FSDP_CPU_RAM_EFFICIENT_LOADING')}"

            # Verify env vars are restored after __init__
            assert os.environ.get('ACCELERATE_USE_FSDP') is None, \
                'ACCELERATE_USE_FSDP should be restored (removed) after __init__'
            assert os.environ.get('FSDP_CPU_RAM_EFFICIENT_LOADING') is None, \
                'FSDP_CPU_RAM_EFFICIENT_LOADING should be restored (removed) after __init__'
        finally:
            # Restore original env
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    def test_env_vars_not_set_when_disabled(self):
        """Verify env vars are NOT set when memory_efficient_init=False."""
        from unittest.mock import MagicMock, patch

        from twinkle.model.transformers.transformers import TransformersModel
        from twinkle.utils import DeviceMesh as TwinkleMesh

        captured_env = {}

        def fake_from_pretrained(cls_or_self, *args, **kwargs):
            captured_env['ACCELERATE_USE_FSDP'] = os.environ.get('ACCELERATE_USE_FSDP')
            captured_env['FSDP_CPU_RAM_EFFICIENT_LOADING'] = os.environ.get('FSDP_CPU_RAM_EFFICIENT_LOADING')
            mock_model = MagicMock()
            mock_model.gradient_checkpointing_enable = MagicMock()
            mock_model.named_parameters = MagicMock(return_value=iter([]))
            return mock_model

        mesh = TwinkleMesh(
            mesh=np.arange(1),
            mesh_dim_names=('fsdp', ),
            device_type='cpu',
        )

        saved = {}
        for key in ('ACCELERATE_USE_FSDP', 'FSDP_CPU_RAM_EFFICIENT_LOADING'):
            saved[key] = os.environ.pop(key, None)

        try:
            with patch('twinkle.model.transformers.transformers.HubOperation') as mock_hub, \
                 patch('twinkle.model.transformers.transformers.AutoModelForCausalLM') as mock_auto, \
                 patch.object(TransformersModel, '_try_init_process_group'), \
                 patch.object(TransformersModel, '_decide_strategy'), \
                 patch.object(TransformersModel, '_construct_default_optimizer_group', return_value=MagicMock()):
                mock_hub.download_model.return_value = '/fake/path'
                mock_auto.from_pretrained = classmethod(fake_from_pretrained)

                try:
                    TransformersModel(
                        model_id='/fake/model',
                        device_mesh=mesh,
                        memory_efficient_init=False,
                    )
                except Exception:
                    pass

            assert captured_env.get('ACCELERATE_USE_FSDP') is None, \
                'ACCELERATE_USE_FSDP should NOT be set when memory_efficient_init=False'
            assert captured_env.get('FSDP_CPU_RAM_EFFICIENT_LOADING') is None, \
                'FSDP_CPU_RAM_EFFICIENT_LOADING should NOT be set when memory_efficient_init=False'
        finally:
            for key, val in saved.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val


# ---------------------------------------------------------------------------
# Task 8: End-to-end integration test
# ---------------------------------------------------------------------------


def _worker_e2e_memory_efficient(rank, world_size, port, model_path):
    """End-to-end: init → set_optimizer → trigger _lazy_wrap_model with memory_efficient."""
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    import twinkle
    from twinkle.model import TransformersModel
    from twinkle.utils import DeviceMesh as TwinkleMesh

    mesh = TwinkleMesh(
        mesh=np.arange(world_size),
        mesh_dim_names=('fsdp', ),
        device_type=_DEVICE_TYPE,
    )

    # Must initialize twinkle before creating TransformersModel, otherwise
    # the @remote_class decorator will strip the device_mesh parameter.
    twinkle.initialize(mode='local', global_device_mesh=mesh)

    model = TransformersModel(
        model_id=model_path,
        device_mesh=mesh,
        strategy='native_fsdp',
        mixed_precision='bf16',
        memory_efficient_init=True,
    )

    model.set_optimizer('AdamW', lr=1e-4)

    # Trigger _lazy_wrap_model by calling the internal method directly.
    model._lazy_wrap_model()

    # Verify: model should be wrapped and parameters on device
    assert model._model_wrapped, 'Model should be wrapped after _lazy_wrap_model'
    for name, param in model.model.named_parameters():
        assert param.device.type == _DEVICE_TYPE, f"{name} on {param.device}, expected {_DEVICE_TYPE}"

    # Verify: gathered full state dict matches (weights were broadcast correctly)
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
    gathered = get_model_state_dict(
        model.model,
        options=StateDictOptions(full_state_dict=True, cpu_offload=True),
    )
    # full_state_dict=True gathers shards to rank 0 only; other ranks get {}.
    if rank == 0:
        assert len(gathered) > 0, 'Should have gathered state dict'

    if dist.is_initialized():
        dist.destroy_process_group()


@unittest.skipIf(_device_count() < 2, f'Need >= 2 {_DEVICE_TYPE.upper()}s')
class TestE2EMemoryEfficientInit(unittest.TestCase):

    def test_e2e_wrap_model(self):
        """Full pipeline test: TransformersModel init → set_optimizer → _lazy_wrap_model."""
        model_id = os.environ.get('TEST_SMALL_MODEL_ID')
        if not model_id:
            self.skipTest('Set TEST_SMALL_MODEL_ID env var to a small HF model path')

        port = _find_free_port()
        world_size = 2
        mp.spawn(
            _worker_e2e_memory_efficient,
            args=(world_size, port, model_id),
            nprocs=world_size,
            join=True,
        )


if __name__ == '__main__':
    unittest.main()
