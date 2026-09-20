# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared Ray runtime, per-test isolation, and evidence boundaries.

``RayActorBackend`` is a forwarding wrapper around a detached Ray actor;
instantiating one without an initialized Ray runtime raises
:class:`RuntimeError`. The vast majority of server tests just construct
``RayActorBackend()`` somewhere in their fixtures, so we boot a small Ray
cluster once per test session here instead of opting in module by module.

Two ``RayActorBackend()`` instances created in the same test session share one
canonical actor (``twinkle_state_actor``) by design — that is the whole point
of the actor wrapper. To keep tests independent we clear that actor's store
before each test function. Tests that pin a non-default ``key_prefix`` get
their own actor; this fixture intentionally leaves those alone.

Evidence boundary (spec T8.4 / R9#10): every mock-model backend method accepts
``**kwargs`` without argument validation, and the mock enters no real collective.
A mock-backed test therefore proves neither request/argument validation nor NCCL
behavior (asymmetric failure, collective mis-pairing, ReduceScatter, etc.). It may
prove only backend dispatch, task-queue behavior, timeout/admission mechanisms,
and event-loop responsiveness. Validation and NCCL claims require the GPU-gated
``test_nccl_safe_*_e2e.py`` tests against a real server. The contract suite covers
all five apps; Tinker compatibility follows the pinned 0.16.1 SDK wire values.
"""
from __future__ import annotations

import pytest


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    """Boot a session-scoped Ray runtime so RayActorBackend can attach actors."""
    ray = pytest.importorskip('ray')
    already_running = ray.is_initialized()
    if not already_running:
        ray.init(
            ignore_reinit_error=True,
            num_cpus=2,
            log_to_driver=False,
            namespace='twinkle_test',
        )
    yield
    if not already_running and ray.is_initialized():
        ray.shutdown()


@pytest.fixture(autouse=True)
def _reset_canonical_state_actor():
    """Clear the canonical state actor's store before each test function.

    Hypothesis property tests reuse the function scope across examples and
    so should call the actor's explicit ``flush_all()`` test hook themselves.
    """
    import ray

    if not ray.is_initialized():
        yield
        return
    try:
        actor = ray.get_actor('twinkle_state_actor')
    except ValueError:
        actor = None
    if actor is not None:
        try:
            ray.get(actor.flush_all.remote())
        except Exception:
            pass
    yield
