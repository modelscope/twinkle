# Copyright (c) ModelScope Contributors. All rights reserved.
"""Weight-publication strategies for the async-RL trainer.

When a training partition finishes, the trainer must make the just-trained
policy visible to the sampler before the next rollout.  How that happens is
model- and deployment-specific, so ``TrainerWorker`` receives it as an injected
``save_adapter`` callable instead of hard-coding one mechanism.  The strategies
here are the concrete, named forms of that callable:

* :class:`AdapterSnapshotSync` -- save a LoRA adapter to disk and let the sampler
  load that snapshot.  Every version is a distinct path, so several can stay
  resident at once and ``max_staleness`` may exceed 1.
* :class:`InPlaceWeightSync` -- overwrite the sampler's single live copy of the
  weights (twinkle's ``CheckpointEngineManager``).  The transport is a deployment
  choice, not a separate strategy: ``mode='colocate'`` pushes over CUDA IPC on a
  shared GPU, ``mode='standalone'`` pushes over NCCL to a disaggregated sampler
  (verl's checkpoint-engine approach).  How much staleness this allows follows
  from the deployment, not from there being one copy: a colocated sampler must
  sit idle while it is overwritten, so at most one batch can be in flight
  (``max_staleness <= 1``); a disaggregated sampler keeps generating on its own
  GPUs between sparse syncs, so queued samples may be several versions old
  (``max_staleness > 1``), exactly verl's fully-async regime.

A strategy is callable with the same ``(admission) -> handle`` signature
``TrainerWorker`` already expects of ``save_adapter``, so it drops into that seam
unchanged.  The handle is the version-pinning token: a disk path for
:class:`AdapterSnapshotSync`, ``None`` for :class:`InPlaceWeightSync` (its single
live copy needs no pinning).  Both are plain objects holding one callback, so
they cross the Ray actor boundary exactly like the ``partial`` callables the
pipeline injects today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from .types import PartitionAdmission


class WeightSyncStrategy(ABC):
    """Publishes newly-trained weights and returns their version-pinning handle."""

    @abstractmethod
    def publish(self, admission: PartitionAdmission) -> str | None:
        """Make the trained policy visible to the sampler and return its handle.

        The handle is a disk path when each version is a separate snapshot, or
        ``None`` when the sampler's single live copy is overwritten in place.
        """

    def __call__(self, admission: PartitionAdmission) -> str | None:
        return self.publish(admission)


class AdapterSnapshotSync(WeightSyncStrategy):
    """Save each trained version as a LoRA adapter snapshot on disk.

    ``save_fn`` is the existing disk-save seam (``pipeline._save_adapter``): it
    writes the adapter and returns its non-empty checkpoint path, which the
    control plane then pins by reference count.  Because each version lives at
    its own path, several can be resident at once, so this strategy does not
    bound ``max_staleness``.
    """

    def __init__(self, save_fn: Callable[[PartitionAdmission], str]):
        if not callable(save_fn):
            raise TypeError(f'AdapterSnapshotSync needs a callable save_fn, got {type(save_fn).__name__}')
        self._save_fn = save_fn

    def publish(self, admission: PartitionAdmission) -> str:
        return self._save_fn(admission)


class InPlaceWeightSync(WeightSyncStrategy):
    """Overwrite the sampler's single live copy of the weights; keep no snapshot.

    ``sync_fn`` is the in-place weight-sync callback -- in dev, the rollout
    engine's ``sync_weights``, backed by twinkle's ``CheckpointEngineManager``
    (CUDA IPC when colocated, NCCL when the sampler is disaggregated).  It leaves
    the sampler generating from the just-trained policy, so ``publish`` returns
    ``None``: there is no per-version path to pin.

    ``abort_fn`` is the abort-on-publish hook that makes overwriting a single live
    copy sound for a deep buffer.  A disaggregated sampler is still generating when
    the trainer publishes, so overwriting its weights mid-decode would produce the
    rest of that generation's tokens under half-updated weights, whose recorded
    logprobs then match no consistent policy version and importance sampling cannot
    correct.  ``publish`` therefore calls ``abort_fn`` BEFORE ``sync_fn``: it
    interrupts every in-flight generation (twinkle's ``abort_all_inflight``), each
    of which returns the tokens it produced so far with ``stop_reason='abort'`` and
    a resume point, and the partial-rollout loop then re-submits it from that point
    ON THE NEW WEIGHTS.  No generation decodes across the overwrite, so each
    segment's logprobs match a single consistent version (the pre-abort segment the
    old one, the resumed segment the new one) and the off-policy correction stays
    valid.  ``abort_fn`` is optional and a no-op when nothing is in flight -- a
    strictly synchronous sampler is idle at the sync point (nothing admitted ahead),
    so a strategy built without it is still correct there; any streaming driver that
    admits a lookahead window has a generation in flight when it publishes, which is
    why ``dev.config.validate._check_streaming_publication`` refuses in_place under
    the streaming regimes unless partial rollout supplies the abort.

    The staleness this permits is a property of the deployment, not of keeping one
    copy.  Colocated, the sampler shares the trainer's GPUs and must be idle to be
    overwritten, so at most one batch is in flight against an older policy
    (``max_staleness <= 1``).  Disaggregated, the sampler generates on its own GPUs
    while the trainer advances, and a sparse sync cadence lets queued samples age
    several versions (``max_staleness > 1``) -- verl's fully-async regime.
    ``dev.config.validate._check_async_mode`` bounds ``max_staleness`` by the
    deployment in use.
    """

    def __init__(self, sync_fn: Callable[[], None], abort_fn: Callable[[], Any] | None = None):
        if not callable(sync_fn):
            raise TypeError(f'InPlaceWeightSync needs a callable sync_fn, got {type(sync_fn).__name__}')
        if abort_fn is not None and not callable(abort_fn):
            raise TypeError(f'InPlaceWeightSync abort_fn must be callable or None, got {type(abort_fn).__name__}')
        self._sync_fn = sync_fn
        self._abort_fn = abort_fn

    def publish(self, admission: PartitionAdmission) -> None:
        # Abort every in-flight generation BEFORE the weights are overwritten, so none decodes across the
        # update: each returns its partial tokens + resume point and re-runs on the new weights (see the
        # class docstring). A no-op when nothing is in flight.
        if self._abort_fn is not None:
            self._abort_fn()
        self._sync_fn()
        return None


def build_weight_sync_strategy(
    name: str,
    *,
    save_fn: Callable[[PartitionAdmission], str] | None = None,
    sync_fn: Callable[[], None] | None = None,
    abort_fn: Callable[[], Any] | None = None,
) -> WeightSyncStrategy:
    """Build the weight-sync strategy selected by ``name``.

    ``save_fn`` is required for ``adapter_snapshot`` and ``sync_fn`` for
    ``in_place``.  ``abort_fn`` is the optional abort-on-publish hook, meaningful
    only for ``in_place`` (the sampler's ``abort_all_inflight``): it interrupts
    every in-flight generation before the weights are overwritten so each resumes
    on the new weights instead of decoding across the update.  There is no separate
    ``nccl`` strategy: NCCL is the transport ``in_place`` uses when the sampler is
    disaggregated, chosen by the deployment (``CheckpointEngineManager`` mode), not
    by this name.
    """
    if name == 'adapter_snapshot':
        if save_fn is None:
            raise ValueError("weight_sync_strategy='adapter_snapshot' requires a save_fn")
        return AdapterSnapshotSync(save_fn)
    if name == 'in_place':
        if sync_fn is None:
            raise ValueError("weight_sync_strategy='in_place' requires a sync_fn")
        return InPlaceWeightSync(sync_fn, abort_fn=abort_fn)
    if name == 'nccl':
        raise ValueError(
            "weight_sync_strategy='nccl' is not a separate strategy: NCCL is how 'in_place' transports "
            "weights to a disaggregated sampler (CheckpointEngineManager mode='standalone'). Use 'in_place' "
            "and select colocate (IPC) vs standalone (NCCL) via the deployment setting.")
    raise ValueError(f'unknown weight_sync_strategy: {name!r}')
