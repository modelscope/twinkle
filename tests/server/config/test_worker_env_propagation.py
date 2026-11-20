# Copyright (c) ModelScope Contributors. All rights reserved.
"""Regression guard for backend-worker env propagation (F9-④ / defect88).

Server-side perf instruments (checkpoint_manager save_training/save_sampler disk-write latency)
gate on ``TWINKLE_PERF`` in the BACKEND worker's ``os.environ``. The key must therefore travel
through ``SHARED_WORKER_ENV_KEYS`` -> ``build_propagated_env_vars`` -> the Ray runtime_env
``env_vars`` that seeds every worker; otherwise the instruments stay silent even when the client
sets ``TWINKLE_PERF=1`` (the F9-④ disk-fraction instrument emitted nothing before this fix).
The worker setup hook (defect88) rides the same path and is guarded here against regression too.
"""
from __future__ import annotations

import pytest

from twinkle.server.config.env_propagation import (
    SHARED_WORKER_ENV_KEYS,
    build_propagated_env_vars,
    build_shared_env_vars,
)


def test_perf_and_hook_keys_are_shared():
    assert 'TWINKLE_PERF' in SHARED_WORKER_ENV_KEYS
    assert 'TWINKLE_WORKER_SETUP_HOOK' in SHARED_WORKER_ENV_KEYS


def test_shared_env_propagates_perf_when_set(monkeypatch):
    monkeypatch.setenv('TWINKLE_PERF', '1')
    monkeypatch.setenv('TWINKLE_WORKER_SETUP_HOOK', 'some.plugin:register')
    shared = build_shared_env_vars()
    assert shared.get('TWINKLE_PERF') == '1'
    assert shared.get('TWINKLE_WORKER_SETUP_HOOK') == 'some.plugin:register'


def test_shared_env_omits_perf_when_unset(monkeypatch):
    monkeypatch.delenv('TWINKLE_PERF', raising=False)
    assert 'TWINKLE_PERF' not in build_shared_env_vars()


def test_propagated_env_includes_perf(monkeypatch):
    monkeypatch.setenv('TWINKLE_PERF', '1')
    assert build_propagated_env_vars().get('TWINKLE_PERF') == '1'


def test_ray_runtime_env_seeds_workers_with_perf(monkeypatch):
    from twinkle.server.launcher.server_launcher import build_ray_runtime_env

    monkeypatch.setenv('TWINKLE_PERF', '1')
    runtime_env = build_ray_runtime_env()
    assert runtime_env.get('env_vars', {}).get('TWINKLE_PERF') == '1'
