"""Abort must reach the rollout actor while lazy generate calls are still pending."""

import threading
from types import SimpleNamespace

import pytest
import ray

from twinkle.infra import dispatch
from twinkle.infra._ray import RayHelper
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout


class PendingGeneration(BaseException):
    """Stop at the real barrier instead of waiting on a fake Ray task."""


class ReachedTransport(BaseException):
    """Stop after dispatch reaches transport; no Ray cluster is needed."""


@pytest.fixture
def pending_rollout(monkeypatch):
    ref = ray.ObjectRef.from_random()
    worker = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "rollout-worker"))
    proxy = SimpleNamespace(
        _actors=[worker],
        _max_concurrency=1024,
        device_mesh=None,
        _continous_work_state={
            "lock": threading.RLock(),
            "owners": {ref: ("rollout-worker", 0)},
            "inflight": {"generate": [ref]},
        },
    )
    events = []

    def wait_for_generation(refs, **kwargs):
        assert refs == [ref]
        events.append("wait_for_generation")
        raise PendingGeneration

    def send(method, workers_and_args, **kwargs):
        assert len(workers_and_args) == 1
        events.append(method)
        raise ReachedTransport

    monkeypatch.setattr(dispatch, "_infra", lambda: SimpleNamespace(_mode="ray", _lazy_collect=False))
    monkeypatch.setattr(dispatch, "_capture_caller", lambda: None)
    monkeypatch.setattr(ray, "get", wait_for_generation)
    monkeypatch.setattr(ray, "wait", lambda *args, **kwargs: ([], [ref]))
    monkeypatch.setattr(RayHelper, "execute_all_async", send)
    return proxy, events


def test_abort_dispatch_does_not_wait_for_pending_generate(pending_rollout):
    proxy, events = pending_rollout
    with pytest.raises(ReachedTransport):
        MultiTurnRollout.abort_all_inflight(proxy)
    assert events == ["abort_all_inflight"]


def test_ordinary_close_still_waits_for_pending_generate(pending_rollout):
    proxy, events = pending_rollout
    with pytest.raises(PendingGeneration):
        MultiTurnRollout.close(proxy)
    assert events == ["wait_for_generation"]


def test_abort_worker_body_preserves_sampler_result():
    result = {"aborted": 7}
    proxy = SimpleNamespace(sampler=SimpleNamespace(abort_all_inflight=lambda: result))
    assert MultiTurnRollout.abort_all_inflight.__wrapped__(proxy) is result
