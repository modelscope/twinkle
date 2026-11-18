# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from twinkle.sampler.sglang_sampler import sglang_engine


@dataclass
class _UpdateRequest:
    serialized_named_tensors: list
    load_format: str | None = None
    flush_cache: bool = True


def _engine(monkeypatch, sessions):
    names = ['sglang', 'sglang.srt', 'sglang.srt.managers', 'sglang.srt.managers.io_struct',
             'sglang.srt.utils', 'sglang.srt.utils.patch_torch']
    modules = {name: ModuleType(name) for name in names}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if '.' in name:
            parent, child = name.rsplit('.', 1)
            setattr(modules[parent], child, module)
    io_struct = modules['sglang.srt.managers.io_struct']
    io_struct.UpdateWeightsFromTensorReqInput = _UpdateRequest
    modules['sglang.srt.utils'].MultiprocessingSerializer = SimpleNamespace(serialize=lambda named: named)
    modules['sglang.srt.utils.patch_torch'].monkey_patch_torch_reductions = lambda: None
    # Keep the weight-update API tests independent of accelerator availability.
    monkeypatch.setattr(sglang_engine, 'torch', SimpleNamespace(
        device=lambda *args: torch.device('cpu'), cuda=SimpleNamespace(current_device=lambda: 0)))

    events = []
    manager = SimpleNamespace(
        update_weights_from_tensor=AsyncMock(side_effect=lambda *args: events.append('update') or (True, 'ok')),
        flush_cache=AsyncMock(return_value=True))
    if sessions:
        io_struct.BeginWeightUpdateReqInput = type('BeginWeightUpdateReqInput', (), {})
        io_struct.EndWeightUpdateReqInput = type('EndWeightUpdateReqInput', (), {})
        manager.begin_weight_update = AsyncMock(side_effect=lambda *args: events.append('begin') or (True, 'ok'))
        manager.end_weight_update = AsyncMock(side_effect=lambda *args: events.append('end') or (True, 'ok'))
    engine = object.__new__(sglang_engine.SGLangEngine)
    engine.engine = SimpleNamespace(tokenizer_manager=manager)
    engine.tensor_parallel_size = 2
    return engine, manager, events


@pytest.mark.parametrize('sessions', [False, True], ids=['legacy', 'sessions'])
def test_tensor_update_supports_both_sglang_apis(monkeypatch, sessions):
    engine, manager, events = _engine(monkeypatch, sessions)
    weight = torch.arange(4, dtype=torch.float32)
    asyncio.run(engine._flush_bucket([('weight', weight)], flush_cache=True, load_format='custom.loader'))
    assert events == (['begin', 'update', 'end'] if sessions else ['update'])
    request, context = manager.update_weights_from_tensor.call_args.args
    assert context is None
    assert request.flush_cache is True
    assert request.load_format == 'custom.loader'
    assert len(request.serialized_named_tensors) == 2
    for payload in request.serialized_named_tensors:
        assert payload[0][0] == 'weight'
        torch.testing.assert_close(payload[0][1], weight)


@pytest.mark.parametrize('sessions', [False, True], ids=['legacy', 'sessions'])
def test_rejected_tensor_update_is_reported(monkeypatch, sessions):
    engine, manager, _ = _engine(monkeypatch, sessions)
    manager.update_weights_from_tensor.side_effect = None
    manager.update_weights_from_tensor.return_value = (False, 'bad shape')
    with pytest.raises(RuntimeError, match='bad shape'):
        asyncio.run(engine._flush_bucket([('weight', torch.ones(1))], flush_cache=False))
    if sessions:
        manager.end_weight_update.assert_awaited_once()


def test_session_is_closed_when_loading_raises(monkeypatch):
    engine, manager, _ = _engine(monkeypatch, True)
    manager.update_weights_from_tensor.side_effect = ValueError('load failed')
    with pytest.raises(ValueError, match='load failed'):
        asyncio.run(engine._flush_bucket([('weight', torch.ones(1))], flush_cache=False))
    manager.end_weight_update.assert_awaited_once()


def test_rejected_session_does_not_load_weights(monkeypatch):
    engine, manager, _ = _engine(monkeypatch, True)
    manager.begin_weight_update.side_effect = None
    manager.begin_weight_update.return_value = (False, 'cannot begin')
    with pytest.raises(RuntimeError, match='cannot begin'):
        asyncio.run(engine._flush_bucket([('weight', torch.ones(1))], flush_cache=False))
    manager.update_weights_from_tensor.assert_not_awaited()
    manager.end_weight_update.assert_not_awaited()


def test_rejected_session_close_is_reported(monkeypatch):
    engine, manager, _ = _engine(monkeypatch, True)
    manager.end_weight_update.side_effect = None
    manager.end_weight_update.return_value = (False, 'cannot end')
    with pytest.raises(RuntimeError, match='cannot end'):
        asyncio.run(engine._flush_bucket([('weight', torch.ones(1))], flush_cache=False))


@pytest.mark.parametrize('sessions', [False, True], ids=['legacy', 'sessions'])
@pytest.mark.parametrize('short_tail', [False, True])
def test_bucketed_update_flushes_cache_once(monkeypatch, sessions, short_tail):
    engine, manager, _ = _engine(monkeypatch, sessions)
    weights = {'first': torch.ones(1 << 18), 'second': torch.zeros(3 if short_tail else 1 << 18)}
    asyncio.run(engine.update_weights(weights, bucket_size_mb=1))
    requests = [call.args[0] for call in manager.update_weights_from_tensor.await_args_list]
    assert [request.flush_cache for request in requests] == [False, short_tail]
    assert manager.flush_cache.await_count == (0 if short_tail else 1)

