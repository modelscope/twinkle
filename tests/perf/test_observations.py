"""Atomic observations carry units and stay independent of recipe policy."""

from types import SimpleNamespace

import torch

from twinkle import Platform
from twinkle.perf.stats import memory_stats


def test_memory_returns_raw_bytes_without_implicit_sync_or_reset(monkeypatch):
    import psutil

    actions = []
    monkeypatch.setattr(Platform, "device_prefix", staticmethod(lambda: "cuda"))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(used=123456))
    monkeypatch.setattr(torch, "cuda", SimpleNamespace(
        is_available=lambda: True,
        synchronize=lambda: actions.append("sync"),
        max_memory_allocated=lambda: 8192,
        max_memory_reserved=lambda: 16384,
    ))
    assert memory_stats() == {
        "cpu_memory_used_bytes": 123456,
        "max_memory_allocated_bytes": 8192,
        "max_memory_reserved_bytes": 16384,
    }
    assert actions == []
    assert memory_stats(synchronize=True)["max_memory_allocated_bytes"] == 8192
    assert actions == ["sync"]


def test_unavailable_device_counters_are_absent(monkeypatch):
    monkeypatch.setattr(Platform, "device_prefix", staticmethod(lambda: "cuda"))
    monkeypatch.setattr(torch, "cuda", SimpleNamespace(is_available=lambda: False))
    assert set(memory_stats()) == {"cpu_memory_used_bytes"}
