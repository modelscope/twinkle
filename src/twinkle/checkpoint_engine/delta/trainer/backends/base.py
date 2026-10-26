# Copyright (c) ModelScope Contributors. All rights reserved.
"""Trainer-backend contract for sharded delta synchronization.

The delta engine only needs a training object that can export (1) the full HF-keyed
weights for the seed, (2) its local parameter shards with placement metadata, and
(3) final HF-coordinate sparse deltas. A backend implements the first two plus a
config check and, when its local coordinates are not already HF coordinates,
overrides :meth:`DeltaTrainerBackend._hf_delta_entry`. Snapshots and diffing are
shared and need no backend code.
"""
from __future__ import annotations

from collections.abc import Iterator

import torch

from ..export import _hf_entry_identity, hf_delta_export, prime_delta_snapshots
from ..spec import ShardSpec

TensorWeights = Iterator[tuple[str, torch.Tensor]]
TensorShards = Iterator[tuple[str, torch.Tensor, ShardSpec]]
HFDeltaEntry = tuple[list[tuple[str, tuple[int, ...]]], str, torch.Tensor,
                     torch.Tensor, torch.Tensor, object]


class DeltaTrainerBackend:
    """Export hooks a training backend provides to the delta engine.

    Required: :meth:`validate_delta_backend`, :meth:`get_per_tensor_param`,
    :meth:`get_per_tensor_param_shard`. Optional: :meth:`_hf_delta_entry`.
    Provided: :meth:`prime_delta_snapshots`, :meth:`get_per_tensor_param_delta_shard`.
    """

    _delta_device: str = 'cuda'
    # Pinned snapshots speed up the diff read-back but compete with other pinned pools
    # on the node; a backend with a tight host-memory profile can default this to False.
    _delta_pin_snapshots: bool = True

    def validate_delta_backend(self) -> None:
        """Raise ``ValueError`` when this rank's configuration cannot be delta-synced."""
        raise NotImplementedError

    def get_per_tensor_param(self) -> tuple[TensorWeights, None]:
        """Full HF-keyed export for the seed. Collective: every rank must iterate it."""
        raise NotImplementedError

    def get_per_tensor_param_shard(self) -> tuple[TensorShards, None]:
        """Yield ``(name, local flat shard, ShardSpec)``, in the same order on every rank."""
        raise NotImplementedError

    def _hf_delta_entry(self, name, spec, place, lidx, lval):
        """Translate one shard-local delta into a final HF-coordinate entry.

        The default is the identity profile: the parameter is its own single HF slot."""
        return _hf_entry_identity(name, spec, place, lidx, lval)

    def prime_delta_snapshots(self) -> None:
        """Snapshot the current shards as the diff base. Runs right after the seed sync."""
        self._delta_shard_snap = getattr(self, '_delta_shard_snap', {})
        shards, _ = self.get_per_tensor_param_shard()
        pin = self._delta_device == 'cuda' and self._delta_pin_snapshots
        prime_delta_snapshots(shards, self._delta_shard_snap, pin=pin)

    def get_per_tensor_param_delta_shard(self) -> tuple[Iterator[HFDeltaEntry], None]:
        """Steady export: HF-coordinate deltas since the previous export. Needs a prior seed."""
        self._delta_shard_snap = getattr(self, '_delta_shard_snap', {})
        shards, _ = self.get_per_tensor_param_shard()
        return hf_delta_export(shards, self._delta_shard_snap, self._hf_delta_entry), None
