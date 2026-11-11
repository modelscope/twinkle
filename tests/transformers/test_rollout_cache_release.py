"""Full Python GC remains periodic and mandatory when the rollout reservation is at risk."""
from unittest.mock import Mock

import pytest
import torch

from twinkle.model import offload
from twinkle.utils.framework import Torch


@pytest.mark.parametrize('free,full_expected', [(90, False), (20, True)])
def test_gc_pressure_uses_rollout_reservation(monkeypatch, free, full_expected):
    collect, empty = Mock(), Mock()
    monkeypatch.setattr('gc.collect', collect)
    monkeypatch.setattr(Torch, 'empty_cache', empty)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (free, 100))
    monkeypatch.setattr(offload, '_CACHE_RELEASE_COUNT', 1)
    offload.release_caching_allocator(min_free_fraction=.55)
    assert collect.call_args_list[0].args == (0,)
    assert (() in [call.args for call in collect.call_args_list]) is full_expected
    assert empty.call_count == 1 + int(full_expected)


def test_periodic_full_collection_even_with_free_memory(monkeypatch):
    collect = Mock()
    monkeypatch.setattr('gc.collect', collect)
    monkeypatch.setattr(Torch, 'empty_cache', Mock())
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (90, 100))
    monkeypatch.setattr(offload, '_CACHE_RELEASE_COUNT', 16)
    offload.release_caching_allocator(min_free_fraction=.55)
    assert collect.call_args.args == ()
