# Copyright (c) ModelScope Contributors. All rights reserved.
"""Device-agnostic XOR-tree delta wire checksum (CPU only)."""
import importlib.util
import json
import sys
from dataclasses import dataclass
from functools import reduce
from operator import xor
from pathlib import Path
from types import ModuleType

import pytest
import torch

from twinkle.checkpoint_engine.delta.protocol import wire
from twinkle.checkpoint_engine.delta.protocol.wire import checksum

MASK = (1 << 64) - 1


def ref_xor(t):
    return reduce(xor, t.contiguous().view(torch.uint8).flatten().tolist(), 0)


def rand_bytes(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (n,), dtype=torch.uint8, generator=g)


def test_empty_is_zero():
    empty = torch.empty(0, dtype=torch.uint8)
    assert wire._xor_tree(empty) == 0
    assert checksum(empty, empty) == 0
    assert checksum(empty, torch.empty(0, dtype=torch.bfloat16)) == 0


def test_deterministic_and_dtype_view_invariant():
    values = torch.randn(33).to(torch.bfloat16)
    pos = rand_bytes(12)
    a = checksum(pos, values)
    assert a == checksum(pos, values)
    assert a == checksum(pos, values.view(torch.uint8))
    assert a == checksum(pos, values.view(torch.int16))
    assert a == checksum(pos.clone(), values.clone())


@pytest.mark.parametrize('n', [1, 2, 3, 7, 8, 9, 1023])
def test_matches_reference_for_lengths(n):
    pos = rand_bytes(n, seed=n)
    vals = rand_bytes(n, seed=n + 100)
    assert wire._xor_tree(pos) == ref_xor(pos)
    assert checksum(pos, vals) == (ref_xor(pos) ^ (ref_xor(vals) << 1)) & MASK
    odd = rand_bytes(n * 2 + 2, seed=n).view(torch.bfloat16)  # non-8-aligned byte size for most n
    assert wire._xor_tree(odd) == ref_xor(odd)


def test_single_bit_flip_changes_value():
    pos = rand_bytes(13, seed=1)
    vals = rand_bytes(40, seed=2)
    base = checksum(pos, vals)
    for i in range(vals.numel()):
        for bit in range(8):
            flipped = vals.clone()
            flipped[i] ^= 1 << bit
            assert checksum(pos, flipped) != base
    for i in range(pos.numel()):
        for bit in range(8):
            flipped = pos.clone()
            flipped[i] ^= 1 << bit
            assert checksum(flipped, vals) != base


def test_input_not_mutated():
    for t in (rand_bytes(9), rand_bytes(16), torch.randn(5).to(torch.bfloat16)):
        before = t.clone()
        checksum(t, t)
        assert torch.equal(t.view(torch.uint8), before.view(torch.uint8))


def test_non_contiguous_input():
    base = rand_bytes(64).view(8, 8)
    view = base.t()
    assert wire._xor_tree(view) == ref_xor(view.contiguous())


# ---------- end-to-end decode ---------------------------------------------


@pytest.fixture
def adapter(monkeypatch):
    @dataclass
    class CheckpointWeightPatch:
        name: str
        shape: tuple
        dtype: torch.dtype
        values: torch.Tensor
        indices: torch.Tensor | None = None

    class Base:
        def __init__(self, *args, **kwargs):
            pass

    modules = {
        'vllm': {'__version__': 'test'},
        'vllm.config.weight_transfer': {'WeightTransferConfig': type('WeightTransferConfig', (), {})},
        'vllm.distributed.weight_transfer.base': {
            'WeightTransferEngine': Base,
            'WeightTransferInitInfo': type('WeightTransferInitInfo', (), {}),
            'WeightTransferUpdateInfo': type('WeightTransferUpdateInfo', (), {}),
        },
        'vllm.model_executor.model_loader.checkpoint_weight_patch': {
            'CheckpointWeightPatch': CheckpointWeightPatch,
            'load_checkpoint_weight_patches': lambda *a, **k: None,
        },
        'vllm.model_executor.layers.vocab_parallel_embedding': {
            'VocabParallelEmbedding': type('VocabParallelEmbedding', (), {}),
        },
        'vllm.model_executor.model_loader.reload': {
            'initialize_layerwise_reload': lambda *a, **k: None,
            'finalize_layerwise_reload': lambda *a, **k: None,
        },
    }
    for name, fields in modules.items():
        module = ModuleType(name)
        module.__dict__.update(fields)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).parents[2] / 'src/twinkle/sampler/vllm_sampler/delta/transfer_engine.py'
    spec = importlib.util.spec_from_file_location('_twinkle_delta_checksum_adapter_test', path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
        module.decode_delta_payload  # noqa: B018
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f'vLLM adapter unavailable: {exc!r}')
    return module


def build_flush(encoding, corrupt=False):
    values = torch.tensor([-0., 2.], dtype=torch.bfloat16)
    positions = torch.tensor([0, 3], dtype=torch.int32).view(torch.uint8)
    if encoding == 'dense':
        positions = torch.empty(0, dtype=torch.uint8)
    params = [dict(name='w', dtype='bfloat16', shape=[4 if encoding == 'indices' else 2],
                   pos_start=0, pos_end=positions.numel(), pos_width=4, val_start=0, val_end=2)]
    spec = dict(encoding=encoding, params=params, checksum=checksum(positions, values))
    spec_tensor = torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)
    if corrupt:
        values[0] = 42
    return [('__delta_spec__', spec_tensor), ('__positions__', positions), ('__values__', values)]


@pytest.mark.parametrize('encoding', ['indices', 'dense'])
def test_decode_accepts_valid_payload(adapter, encoding):
    kind, patches = adapter.decode_delta_payload(build_flush(encoding))
    assert kind == encoding and patches[0].name == 'w'


@pytest.mark.parametrize('encoding', ['indices', 'dense'])
def test_decode_rejects_corrupt_payload(adapter, encoding):
    with pytest.raises(RuntimeError, match='checksum'):
        adapter.decode_delta_payload(build_flush(encoding, corrupt=True))


def test_decode_rejects_tampered_checksum_field(adapter):
    flush = build_flush('indices')
    spec = json.loads(bytes(flush[0][1].numpy().tobytes()).decode())
    spec['checksum'] ^= 1
    flush[0] = ('__delta_spec__', torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8))
    with pytest.raises(RuntimeError, match='checksum'):
        adapter.decode_delta_payload(flush)
