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


class _Checkpoint:
    def __init__(self, events, model):
        self.events = events
        self.model = model
        self.merged = False

    def prepare_for_lending(self):
        assert not self.model.offloaded
        self.events.append("merge")
        self.merged = True

    def restore_after_lending(self):
        assert not self.model.offloaded
        self.events.append("unmerge")
        self.merged = False

    def warm(self, sampler):
        assert sampler.events is self.events
        assert self.merged and self.model.offloaded
        self.events.append("warm")


def _adapter():
    events = []
    model = _Model(events)
    checkpoint = _Checkpoint(events, model)
    return ClusterAdapter(
        {"sampler": _Sampler(events)},
        model=model,
        lend_ranks=[0, 1],
        spare_ranks=[4, 5],
        warmups={"sampler": checkpoint.warm},
        before_lend={"sampler": checkpoint.prepare_for_lending},
        after_reclaim={"sampler": checkpoint.restore_after_lending},
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
    assert restored.model.events == ["merge", "offload", ("grow", 1, [0, 1]), "warm"]
    assert restored._lent == "sampler"

    restored.reclaim()
    assert restored.model.events[-3:] == [("retire", [0, 1]), "reload", "unmerge"]
    assert restored._lent is None
    assert len(restored._switch_overheads) == 1


@pytest.mark.parametrize("failure", ["prepare", "offload", "grow", "warm"])
def test_failed_lend_restores_training_state(monkeypatch, failure):
    adapter = _adapter()
    checkpoint = adapter.before_lend["sampler"].__self__

    def fail():
        raise RuntimeError(f"{failure} failed")

    if failure == "prepare":

        def partial_prepare():
            checkpoint.prepare_for_lending()
            fail()

        adapter.before_lend["sampler"] = partial_prepare
    elif failure == "offload":
        original = adapter.model.offload_to_cpu

        def partial_offload():
            original()
            fail()

        monkeypatch.setattr(adapter.model, "offload_to_cpu", partial_offload)
    elif failure == "grow":
        monkeypatch.setattr(adapter.roles["sampler"], "add_workers", lambda *args, **kwargs: fail())
    else:
        adapter.warmups["sampler"] = lambda sub: fail()

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        adapter.lend(0.8, waited=11.0)
    assert adapter.model.events[-2:] == ["reload", "unmerge"]
    assert not adapter.model.offloaded and not checkpoint.merged
    assert adapter._lent is None and adapter._pending_restore is None


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_lend_oom_still_restores_prepared_lora(monkeypatch, cleanup_fails):
    import torch

    adapter = _adapter()
    checkpoint = adapter.before_lend["sampler"].__self__

    def fail_warm(sub):
        checkpoint.warm(sub)
        raise torch.OutOfMemoryError("CUDA out of memory")

    adapter.warmups["sampler"] = fail_warm
    if cleanup_fails:
        def fail_cleanup(**kwargs):
            raise RuntimeError("actor already dead")

        monkeypatch.setattr(adapter.roles["sampler"], "remove_workers", fail_cleanup)

    adapter.lend(0.8, waited=11.0)

    assert adapter.model.events[-2:] == ["reload", "unmerge"]
    assert not adapter.model.offloaded and not checkpoint.merged
    assert adapter._lent is None and adapter._pending_restore is None


def test_failed_restore_is_retried_before_training_can_resume():
    adapter = _adapter()
    checkpoint = adapter.before_lend["sampler"].__self__

    def fail_restore():
        raise RuntimeError("unmerge failed")

    def fail_warm(sub):
        raise RuntimeError("warm failed")

    adapter.after_reclaim["sampler"] = fail_restore
    adapter.warmups["sampler"] = fail_warm
    with pytest.raises(RuntimeError, match="unmerge failed"):
        adapter.lend(0.8, waited=11.0)
    assert adapter._pending_restore == "sampler"
    with pytest.raises(RuntimeError, match="unmerge failed"):
        adapter.reclaim()
    adapter.after_reclaim["sampler"] = checkpoint.restore_after_lending
    adapter.reclaim()
    assert adapter._pending_restore is None and not checkpoint.merged


@pytest.mark.parametrize("failure", ["reload", "unmerge"])
def test_reclaim_keeps_failed_restore_pending(monkeypatch, failure):
    adapter = _adapter()
    checkpoint = adapter.before_lend["sampler"].__self__
    adapter.lend(0.8, waited=11.0)
    reload = adapter.model.reload_to_gpu

    def fail():
        raise RuntimeError(f"{failure} failed")

    if failure == "reload":
        monkeypatch.setattr(adapter.model, "reload_to_gpu", fail)
    else:
        adapter.after_reclaim["sampler"] = fail
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        adapter.reclaim()
    assert adapter._pending_restore == "sampler"
    assert checkpoint.merged
    assert adapter.model.offloaded == (failure == "reload")

    monkeypatch.setattr(adapter.model, "reload_to_gpu", reload)
    adapter.after_reclaim["sampler"] = checkpoint.restore_after_lending
    adapter.reclaim()
    assert not checkpoint.merged and not adapter.model.offloaded
    assert adapter._pending_restore is None
    assert adapter.model.events.count(("retire", [0, 1])) == 1


def test_lending_without_callbacks_retains_the_original_cycle():
    adapter = _adapter()
    adapter.before_lend.clear()
    adapter.after_reclaim.clear()
    adapter.warmups["sampler"] = lambda sub: adapter.model.events.append("warm")
    adapter.lend(0.8, waited=11.0)
    adapter.reclaim()
    assert adapter.model.events == ["offload", ("grow", 1, [0, 1]), "warm", ("retire", [0, 1]), "reload"]


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
