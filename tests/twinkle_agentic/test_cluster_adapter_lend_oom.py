# Copyright (c) ModelScope Contributors. All rights reserved.
"""defect117 permanent gate: lend must graceful-skip on CUDA OOM, not crash the trainer.

In disaggregated mode the shared_sampler is resident on the trainer ranks; lend offloads only the
trainer, so a grown engine's warm reserve (util x total) plus the resident sampler can exceed free
GPU memory. v13 (util0.9) crashed: add_workers -> warm -> torch.OutOfMemoryError propagated out of
lend and killed the run. The fix detects OOM through the ray-wrapped cause chain, retires the
half-built replica, reloads the trainer, warns, and returns (``_lent`` stays None so training
continues); non-OOM errors still raise (original semantics).

PLAN (case | input | expected | failure it kills):
1 oom_typed_skip | add_workers raises torch.OutOfMemoryError | lend returns; _lent None;
  remove_workers(lend_targets) called; reload_to_gpu called; no raise | the exact defect117 crash
2 oom_ray_wrapped_skip | add_workers raises RuntimeError whose __cause__ is OutOfMemoryError |
  skip (cause chain walked) | ray-wrapped OOM missed by a top-level-only check
3 oom_str_only_skip | add_workers raises RuntimeError('CUDA out of memory ...') no typed cause |
  skip via message | string-only OOM (ray flattens the cause) missed
4 non_oom_raises | add_workers raises ValueError('bad ranks') | reload_to_gpu called THEN
  ValueError raised; _lent None; remove_workers NOT called | non-OOM swallowed (masks real bugs)
5 headroom_ok_grow | add_workers succeeds | _lent='sampler'; added recorded; no reload/remove |
  normal grow broken by the fix
6 cleanup_fail_safe | add_workers OOM AND remove_workers raises | reload_to_gpu still called;
  lend returns (no crash) | a failed cleanup crashing the trainer
7 reverse-verify (procedure) | revert the OOM branch to plain reload+raise | case 1 RED (raises) |
  a gate that passes without the fix proves nothing

Stand-ins replace only the ray handles; ClusterAdapter.lend and the OOM detector are real product
code. The real ray e2e (util0.9 skip-no-crash vs util0.7 grow) is signed off by the D3 rerun.
"""
import pytest
import torch

from twinkle_agentic.async_rl.cluster_adapter import ClusterAdapter


class _Role:
    def __init__(self, add_exc=None, remove_exc=None):
        self.added = []
        self.removed = []
        self._add_exc = add_exc
        self._remove_exc = remove_exc

    def pressure(self):
        return 0.5

    def add_workers(self, count, warmup=None, ranks=None):
        if self._add_exc is not None:
            raise self._add_exc
        self.added.append((count, tuple(ranks or ())))
        if warmup is not None:
            warmup(self)
        return list(ranks or ())

    def remove_workers(self, k=0, ranks=None):
        if self._remove_exc is not None:
            raise self._remove_exc
        released = list(ranks) if ranks is not None else list(range(k))
        self.removed.append((k, tuple(released)))
        return released


class _Model:
    def __init__(self):
        self.offloads = 0
        self.reloads = 0

    def offload_to_cpu(self):
        self.offloads += 1

    def reload_to_gpu(self):
        self.reloads += 1


def _adapter(role):
    return ClusterAdapter(
        {'sampler': role},
        model=_Model(),
        lend_ranks=[0, 1],
        spare_ranks=[2, 3],
        warmups={'sampler': lambda sub: None},
        widths={'sampler': 2},
        window=4,
        cooldown=8)


def _oom():
    return torch.OutOfMemoryError('CUDA out of memory. Tried to allocate 3.00 GiB.')


def test_lend_oom_typed_skips_without_crash():
    role = _Role(add_exc=_oom())
    adapter = _adapter(role)
    adapter.lend(train_idle_ratio=1.0)  # pre-fix: raises OutOfMemoryError (defect117)
    assert adapter._lent is None
    assert role.removed == [(0, (0, 1))]  # half-built replica retired on the attempted ranks
    assert adapter.model.reloads == 1
    assert adapter.model.offloads == 1


def test_lend_oom_ray_wrapped_cause_skips():
    wrapped = RuntimeError('ray::SamplingDriver.add_workers() failed')
    wrapped.__cause__ = _oom()
    role = _Role(add_exc=wrapped)
    adapter = _adapter(role)
    adapter.lend(train_idle_ratio=1.0)
    assert adapter._lent is None
    assert adapter.model.reloads == 1
    assert role.removed == [(0, (0, 1))]


def test_lend_oom_string_only_skips():
    role = _Role(add_exc=RuntimeError('CUDA out of memory. Tried to allocate 3.00 GiB.'))
    adapter = _adapter(role)
    adapter.lend(train_idle_ratio=1.0)
    assert adapter._lent is None
    assert adapter.model.reloads == 1


def test_lend_non_oom_still_raises():
    role = _Role(add_exc=ValueError('bad ranks'))
    adapter = _adapter(role)
    with pytest.raises(ValueError, match='bad ranks'):
        adapter.lend(train_idle_ratio=1.0)
    assert adapter._lent is None
    assert adapter.model.reloads == 1  # original semantics: reload then raise
    assert role.removed == []  # no OOM cleanup for a non-OOM error


def test_lend_headroom_ok_grows():
    role = _Role()
    adapter = _adapter(role)
    adapter.lend(train_idle_ratio=1.0)
    assert adapter._lent == 'sampler'
    assert role.added == [(1, (0, 1))]
    assert adapter.model.reloads == 0
    assert role.removed == []


def test_lend_oom_cleanup_failure_still_recovers():
    role = _Role(add_exc=_oom(), remove_exc=RuntimeError('actor already dead'))
    adapter = _adapter(role)
    adapter.lend(train_idle_ratio=1.0)  # cleanup raises but must not crash the trainer
    assert adapter._lent is None
    assert adapter.model.reloads == 1