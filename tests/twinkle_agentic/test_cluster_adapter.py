# Copyright (c) ModelScope Contributors. All rights reserved.
"""Adaptive scheduling must survive the Ray actor constructor boundary."""

import pickle
import subprocess
import sys
import textwrap

import pytest

from twinkle_agentic.async_rl.cluster_adapter import ClusterAdapter


class _Model:
    def __init__(self, events):
        self.events = events
        self.offloaded = False

    def offload_to_cpu(self):
        self.events.append("offload")
        self.offloaded = True

    def reload_to_gpu(self):
        self.events.append("reload")
        self.offloaded = False


class _Sampler:
    def __init__(self, events):
        self.events = events

    def pressure(self):
        return 1.0

    def add_workers(self, count, *, warmup, ranks):
        self.events.append(("grow", count, ranks))
        warmup(self)

    def remove_workers(self, *, ranks):
        self.events.append(("retire", ranks))


def _warm(sampler):
    sampler.events.append("warm")


def _adapter():
    events = []
    return ClusterAdapter(
        {"sampler": _Sampler(events)},
        model=_Model(events),
        lend_ranks=[0, 1],
        spare_ranks=[4, 5],
        warmups={"sampler": _warm},
        widths={"sampler": 2},
        window=2,
        cooldown=3,
    )


@pytest.fixture(params=["pickle", "ray.cloudpickle"])
def serializer(request):
    return pickle if request.param == "pickle" else pytest.importorskip(request.param)


def test_roundtrip_preserves_controller_state_and_recreates_lock(serializer):
    original = _adapter()
    original._history["sampler"].append(0.75)
    original._cooldown = 2
    original._engaged = True
    original._switch_overheads = [1.0, 2.0]

    restored = serializer.loads(serializer.dumps(original))

    for name in (
        "lend_ranks",
        "widths",
        "window",
        "ratio_up",
        "ratio_down",
        "cooldown",
        "lend_idle_ratio",
        "_spare",
        "_history",
        "_engaged",
        "_cooldown",
        "_lent",
        "_switch_overheads",
        "_lend_duration",
    ):
        assert getattr(restored, name) == getattr(original, name)
    assert restored._history["sampler"].maxlen == 2
    assert restored.roles["sampler"].events is restored.model.events
    assert restored._lock is not original._lock
    with original._lock:
        assert restored._lock.acquire(blocking=False)
        restored._lock.release()
    assert not original._lock.locked()


def test_roundtrip_can_lend_warm_and_reclaim(serializer):
    restored = serializer.loads(serializer.dumps(_adapter()))

    restored.lend(0.8, waited=11.0)
    assert restored.model.events == ["offload", ("grow", 1, [0, 1]), "warm"]
    assert restored._lent == "sampler"

    restored.reclaim()
    assert restored.model.events[-2:] == [("retire", [0, 1]), "reload"]
    assert restored._lent is None
    assert len(restored._switch_overheads) == 1


def test_adapter_passes_through_ray_actor_constructor():
    pytest.importorskip("ray")
    # Isolate the Ray cluster and Twinkle process globals from the rest of the suite.
    code = textwrap.dedent("""
        from types import SimpleNamespace
        import uuid
        import ray
        from twinkle_agentic.async_rl.cluster_adapter import ClusterAdapter

        @ray.remote
        class Receiver:
            def __init__(self, adapter):
                from twinkle_agentic.async_rl.streaming_driver import StreamingDriver
                self.driver = StreamingDriver([], SimpleNamespace(), batch_size=8,
                                              num_generations=4, cluster_adapter=adapter)

            def inspect(self):
                adapter = self.driver.cluster_adapter
                with adapter._lock:
                    return adapter.lend_ranks, adapter.widths, self.driver.batch_size

            def close(self):
                self.driver.close()

        ray.init(address='local', num_cpus=1, include_dashboard=False, log_to_driver=False,
                 _temp_dir='/tmp/twca-' + uuid.uuid4().hex[:6])
        try:
            adapter = ClusterAdapter({'sampler': SimpleNamespace()}, model=SimpleNamespace(),
                                     lend_ranks=[0, 1], widths={'sampler': 2})
            receiver = Receiver.remote(adapter)
            assert ray.get(receiver.inspect.remote(), timeout=45) == ((0, 1), {'sampler': 2}, 8)
            ray.get(receiver.close.remote(), timeout=15)
            ray.kill(receiver, no_restart=True)
        finally:
            ray.shutdown()
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
