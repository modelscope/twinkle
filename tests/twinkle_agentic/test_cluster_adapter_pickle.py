# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect114 permanent gate: ClusterAdapter must survive ray argument serialization.

StreamingDriver is a @remote_class actor; in ray mode the cluster_adapter argument is pickled into
the actor. Every operand the adapter holds is a ray proxy (the model, the sampler role, and
``warmups={'sampler': manager.warm}`` -- CheckpointEngineManager is itself a @remote_class actor, so
its bound method pickles as handle+name). The ONE non-serializable attribute is the process-local
``_lock = threading.Lock()`` guarding adjust/lend/reclaim. The fix drops it in __getstate__ and
rebuilds it in __setstate__ (a fresh lock is the correct semantics: the actor copy's concurrent
adjust/lend/reclaim still need guarding, but a lock cannot cross a process boundary).

PLAN (case | input | expected | failure it kills):
1 round_trip | adapter with picklable ray-handle stand-ins + a real bound-method warmup |
  pickle.dumps/loads succeeds; _lock is a fresh acquirable lock; lend_ranks/widths/window/warmups
  preserved | the exact defect114 TypeError: cannot pickle '_thread.lock'
2 adjust_guard | deserialized clone.adjust() | runs under the rebuilt lock, _history advances |
  a dropped lock that leaves adjust unprotected / broken
3 lend_reclaim | clone.lend(1.0) then clone.reclaim() | model offload/reload called, role
  add/remove_workers called, warmup invoked, _switch_overheads recorded | a rebuilt lock that
  deadlocks or a warmup bound method that did not survive the round trip
4 reverse-verify (procedure) | remove __getstate__/__setstate__ | case 1 RED TypeError | a gate
  that passes without the fix proves nothing

The stand-ins replace only the ray handles (real proxies in the D3 e2e); ClusterAdapter itself and
its adjust/lend/reclaim/pickle path are the real product code. The real ray actor creation is
signed off by the D3 adaptive e2e rerun, not here.
"""
import pickle
import threading

from twinkle_agentic.async_rl.cluster_adapter import ClusterAdapter


class _Role:
    """Picklable stand-in for a ray sampler handle."""

    def __init__(self):
        self.added = []
        self.removed = []

    def pressure(self):
        return 0.5

    def add_workers(self, count, warmup=None, ranks=None):
        self.added.append((count, tuple(ranks or ())))
        if warmup is not None:
            warmup(self)
        return list(ranks or ())

    def remove_workers(self, k=0, ranks=None):
        # Mirrors worker_mixin.remove_workers(k=0, ranks=None) -> released ranks: reclaim calls it
        # with ranks only (k defaults 0), _move calls it with a positional count only.
        released = list(ranks) if ranks is not None else list(range(k))
        self.removed.append((k, tuple(released)))
        return released


class _Model:
    """Picklable stand-in for the ray trainer model handle."""

    def __init__(self):
        self.offloads = 0
        self.reloads = 0

    def offload_to_cpu(self):
        self.offloads += 1

    def reload_to_gpu(self):
        self.reloads += 1


class _Warm:
    """Picklable object whose bound method stands in for the ray manager's warm."""

    def __init__(self):
        self.calls = []

    def warm(self, sub):
        self.calls.append(sub)


def _adapter():
    return ClusterAdapter(
        {'sampler': _Role()},
        model=_Model(),
        lend_ranks=[0, 1],
        spare_ranks=[2, 3],
        warmups={'sampler': _Warm().warm},
        widths={'sampler': 2},
        window=4,
        cooldown=8)


def test_cluster_adapter_survives_ray_serialization():
    adapter = _adapter()
    # case 1: round trip (pre-fix this raises TypeError: cannot pickle '_thread.lock')
    clone = pickle.loads(pickle.dumps(adapter))
    assert isinstance(clone._lock, type(threading.Lock()))
    assert clone._lock.acquire(blocking=False)
    clone._lock.release()
    assert clone.lend_ranks == (0, 1)
    assert clone.widths == {'sampler': 2}
    assert clone.window == 4
    assert set(clone.warmups) == {'sampler'}

    # case 2: adjust runs under the rebuilt lock
    clone.adjust()
    assert len(clone._history['sampler']) == 1

    # case 3: lend/reclaim exercise the rebuilt lock + the survived warmup bound method
    clone.lend(train_idle_ratio=1.0)
    assert clone._lent == 'sampler'
    assert clone.model.offloads == 1
    assert clone.roles['sampler'].added
    clone.reclaim()
    assert clone._lent is None
    assert clone.model.reloads >= 1
    assert clone.roles['sampler'].removed == [(0, (0, 1))]
    assert len(clone._switch_overheads) == 1
