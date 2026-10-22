# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trainer side: diff, placement and gather primitives, plus the FSDP2 delta backend and its lifecycle."""
from __future__ import annotations

import asyncio
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh as TorchMesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

from twinkle import DeviceMesh, Platform
from twinkle.checkpoint_engine.delta.trainer.backends import FSDP2DeltaBackend
from twinkle.checkpoint_engine.delta.trainer.backends.fsdp import validate_fsdp2
from twinkle.checkpoint_engine.delta.transport.nccl_engine import DeltaShardedCheckpointEngine, _GatherQueue
from twinkle.checkpoint_engine.delta.trainer.export import shard_delta_indices
from twinkle.checkpoint_engine.delta.trainer.mixin import DeltaTrainerMixin
from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_delta_indices_matches_bytewise_diff(dtype):
    torch.manual_seed(0)
    # A "shard" of some parameter, whose flat start in the full param is `offset`.
    shard = torch.randn(1000, dtype=dtype)
    new = shard.clone()
    changed = torch.tensor([3, 17, 500, 999], dtype=torch.int64)
    new[changed] = new[changed] + 0.5
    offset = 4096  # this shard begins at flat position 4096 within the full param

    gidx, gval = shard_delta_indices(new, shard, offset)

    # positions are (offset + local changed index), bytewise-exact values
    assert torch.equal(gidx.sort().values, (changed + offset).sort().values)
    order = torch.argsort(gidx)
    got_pos = (gidx[order] - offset).to(torch.int64)
    assert torch.equal(
        gval[order].view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
        new[got_pos].view(torch.int16 if dtype == torch.bfloat16 else torch.int32),
    )


def test_shard_delta_indices_no_change_is_empty():
    shard = torch.randn(256, dtype=torch.bfloat16)
    gidx, gval = shard_delta_indices(shard.clone(), shard, offset=0)
    assert gidx.numel() == 0
    assert gval.numel() == 0


def test_derive_dtensor_placement_unsharded():
    # A non-DTensor (replicated / unsharded) param: offset 0, no gather group,
    # and outside a process group rank 0 is assumed -> contributes.
    from twinkle.checkpoint_engine.delta.trainer.spec import ShardSpec, derive_dtensor_placement

    t = torch.randn(64, 8, dtype=torch.bfloat16)
    spec = ShardSpec.from_param(t)
    assert spec.mesh is None and spec.full_shape == (64, 8)
    offset, contributes, group = derive_dtensor_placement(spec)
    assert offset == 0 and contributes is True and group is None


def test_gather_slot_entries_sub_rounds_world1():
    """max_round_bytes splits the slot list into deterministic sub-rounds; the
    reassembled output must equal the single-round result (world=1 mechanics)."""
    import os

    import torch.distributed as dist

    from twinkle.checkpoint_engine.delta.transport.gather import gather_slot_entries_to_rank0

    owns_pg = not dist.is_initialized()
    if owns_pg:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29512")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        torch.manual_seed(7)
        k = 9
        counts = torch.tensor([5, 0, 3, 7, 1, 0, 4, 2, 6], dtype=torch.int64)
        n = int(counts.sum())
        idx = torch.randint(0, 1000, (n,), dtype=torch.int32)
        val = torch.randn(n, dtype=torch.bfloat16)

        ref = gather_slot_entries_to_rank0(idx, val, counts)
        per_elem = idx.element_size() + val.element_size()
        for budget_elems in (1, 4, 8):
            got = gather_slot_entries_to_rank0(idx, val, counts, max_round_bytes=budget_elems * per_elem)
            assert len(got) == k
            for (ri, rv), (gi, gv) in zip(ref, got, strict=True):
                assert torch.equal(ri, gi)
                assert torch.equal(rv, gv)
    finally:
        # leaving the default pg alive leaks into whatever test runs next in the
        # same session (repo convention: init in the test -> destroy in finally)
        if owns_pg:
            dist.destroy_process_group()


def test_prime_then_hf_delta_export_roundtrip():
    """The backend-side default delta strategy: prime_delta_snapshots pins the
    shards; a later pass through hf_delta_export yields a final-HF-coordinate
    entry with exactly the changed elements and refreshes the snapshot (a
    second delta pass yields a zero-count entry)."""
    from twinkle.checkpoint_engine.delta.trainer.spec import ShardSpec
    from twinkle.checkpoint_engine.delta.trainer.export import _hf_entry_identity, hf_delta_export, prime_delta_snapshots

    w = torch.arange(12, dtype=torch.float32)
    spec = ShardSpec(full_shape=(12,))
    snaps: dict = {}

    prime_delta_snapshots(iter([("w", w.clone(), spec)]), snaps, pin=False)
    assert "w" in snaps and snaps["w"].numel() == 12

    w2 = w.clone()
    w2[3] = -1.0
    w2[7] = 42.0

    ((slots, dtype_str, counts, hf_idx, hf_val, pg),) = list(
        hf_delta_export(iter([("w", w2, spec)]), snaps, _hf_entry_identity)
    )
    assert slots == [("w", (12,))] and dtype_str == "float32" and pg is None
    assert counts.tolist() == [2]
    assert hf_idx.dtype == torch.int32 and hf_idx.tolist() == [3, 7]
    assert hf_val.tolist() == [-1.0, 42.0]

    ((_, _, counts2, hf_idx2, _, _),) = list(
        hf_delta_export(iter([("w", w2.clone(), spec)]), snaps, _hf_entry_identity)
    )
    assert counts2.tolist() == [0] and hf_idx2.numel() == 0


def test_hf_delta_export_requires_seed():
    """A delta export without a prior prime must fail loud, not diff against
    garbage."""
    import pytest

    from twinkle.checkpoint_engine.delta.trainer.spec import ShardSpec
    from twinkle.checkpoint_engine.delta.trainer.export import _hf_entry_identity, hf_delta_export

    def raw():
        yield "w", torch.zeros(4), ShardSpec(full_shape=(4,))

    with pytest.raises(AssertionError, match="seed snapshot"):
        list(hf_delta_export(raw(), {}, _hf_entry_identity))


class FSDPDeltaTrainer(DeltaTrainerMixin, FSDP2DeltaBackend):
    """A model host: lifecycle mixin plus the FSDP2 export backend."""


def make_delta_trainer(owner):
    trainer = FSDPDeltaTrainer()
    trainer.__dict__.update(vars(owner))
    return trainer


@pytest.fixture
def fsdp_owner(monkeypatch):
    platform = type('GPU', (), {})
    monkeypatch.setattr(Platform, 'get_platform', staticmethod(lambda: platform))
    model = SimpleNamespace(config=SimpleNamespace(), state_dict=lambda: {})
    strategy = Mock(spec=NativeFSDPStrategy)
    strategy.unwrap_model.return_value = model
    return SimpleNamespace(strategy=strategy, model=model)


def test_default_two_card_fsdp_passes_parallelism_check(fsdp_owner):
    fsdp_owner.device_mesh = DeviceMesh.from_sizes(world_size=2, fsdp_size=2, device_type='cuda')
    with pytest.raises(ValueError, match='wrapped FSDP2 DTensor model'):
        validate_fsdp2(fsdp_owner)


def test_single_rank_trainer_accepts_unsharded_model(fsdp_owner):
    fsdp_owner.device_mesh = DeviceMesh.from_sizes(world_size=1, fsdp_size=1, device_type='cuda')
    fsdp_owner.model.state_dict = lambda: {'w': torch.ones(4, dtype=torch.bfloat16)}
    validate_fsdp2(fsdp_owner)


def test_single_rank_trainer_uses_rank_zero_without_process_group(monkeypatch):
    from twinkle.checkpoint_engine.delta.trainer import mixin as delta_trainer
    monkeypatch.setattr(delta_trainer.dist, 'is_initialized', lambda: False)
    assert FSDPDeltaTrainer._delta_rank() == 0
    trainer = FSDPDeltaTrainer()
    trainer.validate_delta_backend = Mock()
    trainer._validate_delta_ranks()
    assert trainer._delta_metadata_group is None


@pytest.mark.parametrize('size', [None, 0, 1])
def test_disabled_parallel_dimensions_pass_parallelism_check(fsdp_owner, size):
    fsdp_owner.device_mesh = SimpleNamespace(tp_world_size=size, pp_world_size=size, ep_world_size=size)
    with pytest.raises(ValueError, match='wrapped FSDP2 DTensor model'):
        validate_fsdp2(fsdp_owner)


@pytest.mark.parametrize('dimension', ['tp_world_size', 'pp_world_size', 'ep_world_size'])
def test_enabled_training_parallelism_is_rejected(fsdp_owner, dimension):
    fsdp_owner.device_mesh = SimpleNamespace(tp_world_size=0, pp_world_size=0, ep_world_size=0)
    setattr(fsdp_owner.device_mesh, dimension, 2)
    with pytest.raises(ValueError, match='training TP=PP=EP=1'):
        validate_fsdp2(fsdp_owner)


def test_full_seed_stages_dtensor_before_full_tensor(monkeypatch):
    calls = []

    class HostDTensor:
        def to(self, device, non_blocking):
            calls.append(('stage', device, non_blocking))
            return self

        def full_tensor(self):
            calls.append(('full',))
            return torch.ones(2)

    monkeypatch.setattr(sys.modules['torch.distributed.tensor'], 'DTensor', HostDTensor)
    exporter = make_delta_trainer(SimpleNamespace(_delta_device='cuda'))
    exporter._delta_state_dict = lambda: {'w': HostDTensor()}
    weights, _ = exporter.get_per_tensor_param()
    list(weights)
    assert calls == [('stage', 'cuda', True), ('full',)]


def test_full_sync_baseline_also_stages_cpu_offloaded_dtensor(monkeypatch):
    from twinkle.model.transformers.transformers import TransformersModel
    from twinkle.utils.framework import Torch
    from torch.distributed import tensor as tensor_module
    calls = []

    class HostDTensor:
        def to(self, device, non_blocking):
            calls.append(('stage', device, non_blocking))
            return self

        def full_tensor(self):
            calls.append(('full',))
            return torch.ones(2)

    monkeypatch.setattr(tensor_module, 'DTensor', HostDTensor)
    monkeypatch.setattr(Torch, 'get_device', lambda rank: 'cuda')
    model = SimpleNamespace(state_dict=lambda: {'w': HostDTensor()})
    owner = SimpleNamespace(model=model, strategy=SimpleNamespace(unwrap_model=lambda model: model))
    list(TransformersModel._get_weight_generator(owner, adapter_name='', merge_and_sync=True))
    assert calls == [('stage', 'cuda', True), ('full',)]


def test_hf_key_conversion_matches_seed_and_steady_export():
    weights = {'internal.weight': torch.zeros(2)}
    model = SimpleNamespace(state_dict=lambda: weights, _checkpoint_conversion_mapping={'checkpoint': 'internal'})
    owner = SimpleNamespace(_delta_device='cpu', model=model, strategy=SimpleNamespace(unwrap_model=lambda m: m))
    exporter = make_delta_trainer(owner)
    full, _ = exporter.get_per_tensor_param()
    assert [name for name, _ in full] == ['checkpoint.weight']
    exporter.prime_delta_snapshots()
    weights['internal.weight'][1] = 1
    delta, _ = exporter.get_per_tensor_param_delta_shard()
    entries = list(delta)
    assert entries[0][0] == [('checkpoint.weight', (2,))]
    assert entries[0][3].tolist() == [1]


def test_full_sync_baseline_uses_same_hf_key_conversion():
    from twinkle.model.transformers.transformers import TransformersModel
    model = SimpleNamespace(state_dict=lambda: {'internal.weight': torch.ones(1)},
                            _checkpoint_conversion_mapping={'checkpoint': 'internal'})
    owner = SimpleNamespace(model=model, strategy=SimpleNamespace(unwrap_model=lambda m: m))
    weights = TransformersModel._get_weight_generator(owner, adapter_name='', merge_and_sync=True)
    assert [name for name, _ in weights] == ['checkpoint.weight']


def test_snapshot_pinning_can_be_disabled(monkeypatch):
    from twinkle.checkpoint_engine.delta.trainer.backends import base as backend_base
    prime = Mock()
    monkeypatch.setattr(backend_base, 'prime_delta_snapshots', prime)
    exporter = make_delta_trainer(SimpleNamespace(_delta_device='cuda', _delta_pin_snapshots=False))
    exporter.get_per_tensor_param_shard = lambda: (iter([]), None)
    exporter.prime_delta_snapshots()
    assert prime.call_args.kwargs == {'pin': False}


def test_seed_uses_whole_parameter_and_steady_uses_verl_schema():
    weights = {'w': torch.tensor([0., 1., 2., 3.]), 'v': torch.ones(3)}
    model = SimpleNamespace(state_dict=lambda: weights)
    owner = SimpleNamespace(_delta_device='cpu', model=model, strategy=SimpleNamespace(unwrap_model=lambda m: m))
    exporter = make_delta_trainer(owner)
    engine = DeltaShardedCheckpointEngine(bucket_size=6, batch_gather=32)
    engine.is_master, engine.rank = True, 0
    packets = []
    engine._publish_values_flush = lambda params, values, **flags: packets.append(('dense', params, values.clone(), flags))
    engine._publish_flush = lambda flush, **flags: packets.append(('indices', flush, flags))
    engine._publish_terminal = lambda first: packets.append(('terminal', first))
    asyncio.run(engine.send_weights(exporter))
    assert [packet[2].numel() for packet in packets] == [4, 3]
    packets.clear()
    weights['w'][0] = -0.
    asyncio.run(engine.send_weights(exporter))
    flush = packets[0][1]
    assert (flush.params[0].name, flush.positions_cpu.view(torch.int32).tolist(),
            flush.values_gpu.view(torch.int16).tolist(), packets[0][2]['is_last']) == ('w', [0], [-32768], True)
    packets.clear()
    asyncio.run(engine.send_weights(exporter))
    assert packets == [('terminal', False)]


def test_eval_and_same_layout_restore_keep_snapshot_and_sparse_stream():
    model = torch.nn.Linear(3, 2, bias=False).to(torch.bfloat16)
    trainer = make_delta_trainer(SimpleNamespace(
        _delta_device='cpu', model=model, strategy=SimpleNamespace(unwrap_model=lambda m: m)))
    engine = DeltaShardedCheckpointEngine(bucket_size=64 << 20)
    engine.is_master, engine.rank = True, 0
    dense = Mock()
    sparse = Mock()
    terminal = Mock()
    engine._publish_values_flush = dense
    engine._publish_flush = sparse
    engine._publish_terminal = terminal
    asyncio.run(engine.send_weights(trainer))
    snapshot = trainer._delta_shard_snap['weight']
    baseline = snapshot.clone()
    model.eval()
    asyncio.run(engine.send_weights(trainer))
    terminal.assert_called_once_with(False)
    model.train()
    model.load_state_dict({'weight': torch.full_like(model.weight, 2)})
    asyncio.run(engine.send_weights(trainer))
    assert trainer._delta_shard_snap['weight'] is snapshot
    assert dense.call_count == 1
    assert sparse.call_count == 1
    flush = sparse.call_args.args[0]
    restored = baseline.clone()
    restored[flush.positions_cpu.view(torch.int32).long()] = flush.values_gpu
    assert torch.equal(restored.view(torch.int16), model.weight.detach().flatten().view(torch.int16))


def test_trainer_prepare_reuses_engine_between_workflow_rounds():
    from twinkle.checkpoint_engine.delta import DeltaSyncConfig

    trainer = FSDPDeltaTrainer()
    trainer._lazy_wrap_model = Mock()
    engine = object()
    master_metadata = object()

    def initialize(config):
        trainer._delta_engine = engine
        trainer._delta_master_metadata = master_metadata

    trainer._initialize_delta_engine = Mock(side_effect=initialize)
    config = DeltaSyncConfig()
    first = trainer.prepare_delta_sync(config)
    second = trainer.prepare_delta_sync(config)
    assert first == second == dict(device='cuda', master_metadata=master_metadata)
    trainer._initialize_delta_engine.assert_called_once_with(config)


def test_trainer_validation_failure_releases_metadata_group(monkeypatch):
    from twinkle.checkpoint_engine.delta import DeltaSyncConfig
    from twinkle.checkpoint_engine.delta.trainer import mixin as delta_trainer

    trainer = FSDPDeltaTrainer()
    group = object()
    monkeypatch.setattr(delta_trainer.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(delta_trainer.dist, 'new_group', Mock(return_value=group))
    destroy = Mock()
    monkeypatch.setattr(delta_trainer.dist, 'destroy_process_group', destroy)
    trainer._validate_delta_ranks = Mock(side_effect=ValueError('unsupported layout'))
    with pytest.raises(ValueError, match='unsupported layout'):
        trainer._initialize_delta_engine(DeltaSyncConfig())
    destroy.assert_called_once_with(group)
    assert trainer._delta_metadata_group is None
    assert trainer._delta_engine is None


def exercise_fsdp_export(rank, rendezvous):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        mesh = TorchMesh('cpu', [0, 1])
        replicated_mesh = TorchMesh('cpu', [[0, 1]])
        layouts = [(mesh, [Shard(0)]), (mesh, [Shard(1)]),
                   (replicated_mesh, [Shard(0), Replicate()])]
        for parameter_mesh, placements in layouts:
            baseline = torch.arange(15, dtype=torch.bfloat16).reshape(5, 3)
            target = baseline.clone()
            target.view(torch.int16)[0, 0] ^= -32768
            target[4, 2] = 42
            param = distribute_tensor(baseline, parameter_mesh, placements)
            state = {'w': param}
            model = SimpleNamespace(state_dict=lambda: state)
            owner = SimpleNamespace(_delta_device='cpu', model=model,
                                    strategy=SimpleNamespace(unwrap_model=lambda model: model))
            exporter = FSDPDeltaTrainer()
            exporter.__dict__.update(vars(owner))
            exporter.prime_delta_snapshots()
            changed = distribute_tensor(target, parameter_mesh, placements)
            param.to_local().copy_(changed.to_local())
            restored = baseline.clone().flatten()

            def apply(name, dtype, shape, size, indices, values):
                if indices is not None:
                    restored[indices.long()] = values

            queue = _GatherQueue(32, 30, rank == 0, apply)
            original = DTensor.full_tensor
            DTensor.full_tensor = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError('steady export must not all-gather full tensors'))
            try:
                entries, _ = exporter.get_per_tensor_param_delta_shard()
                for slots, dtype, counts, indices, values, group in entries:
                    queue.put(group, slots, dtype, counts, indices, values)
                queue.flush_all()
            finally:
                DTensor.full_tensor = original
            if rank == 0:
                assert torch.equal(restored.view(torch.int16), target.flatten().view(torch.int16))
            entries, _ = exporter.get_per_tensor_param_delta_shard()
            assert all(not entry[2].sum().item() for entry in entries)
        # FSDP can produce an empty shard when there are more ranks than rows.
        param = distribute_tensor(torch.ones(1, 3), mesh, [Shard(0)])
        state = {'w': param}
        exporter = FSDPDeltaTrainer()
        exporter.__dict__.update(vars(owner))
        exporter.prime_delta_snapshots()
        param.to_local().add_(1)
        entries, _ = exporter.get_per_tensor_param_delta_shard()
        entry = next(entries)
        assert entry[2].tolist() == [3 if rank == 0 else 0]
    finally:
        dist.destroy_process_group()


def test_fsdp_export_preserves_bytes_with_uneven_column_and_replica_shards(tmp_path):
    mp.spawn(exercise_fsdp_export, args=(f'file://{tmp_path}/rendezvous',), nprocs=2, join=True)
