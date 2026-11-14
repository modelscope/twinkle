# Copyright (c) ModelScope Contributors. All rights reserved.
"""Collect-only delta pipeline: the trainer's gathered flushes are RETURNED, not broadcast.

Used when the rollout side is an SGLang engine. SGLang has no external NCCL group, no ZMQ
manifest channel and no receiver actors -- its own ``update_weights_from_tensor`` already
serializes tensors into every TP worker process as CUDA IPC handles. So this engine keeps only
the transport-agnostic half of :class:`DeltaSyncMixin` (per-shard diff on every trainer rank,
sparse gather to rank 0, bucketing) and overrides the three publish hooks to collect each flush
as the ``(named_tensors, is_last)`` item the rollout adapter consumes:

* ``__delta_spec__``  -- uint8 tensor holding the JSON manifest (encoding/params/checksum)
* ``__positions__``  -- uint8 blob of packed int32 positions (absent for dense flushes)
* ``__values__``     -- the flush's values

The manager then hands the collected stream to the sampler actor, which forwards it through
sglang with ``load_format`` pointing at :data:`..rollout.sglang_loader.LOADER_FQN`.

Collected flushes are pinned to host memory: the trainer keeps at most two GPU buckets alive
(the bucket pipeline releases each one as soon as it is published), and a full seed -- a whole
model's worth of flushes collected at once -- must not sit in trainer GPU memory while it waits
for the manager to pick the stream up. Host peak during the seed is therefore one model copy,
the same order as a full sync's offload.
"""
from __future__ import annotations

import json
import torch

from .pipeline import DeltaSyncMixin


def _spec_tensor(spec: dict, device: str) -> torch.Tensor:
    """One flush's JSON manifest as a uint8 tensor, on ``device`` (the wire shape)."""
    payload = json.dumps(spec).encode()
    return torch.frombuffer(bytearray(payload), dtype=torch.uint8).to(device)


class DeltaCollectEngine(DeltaSyncMixin):
    """Gather-to-rank-0 delta engine whose flush stream is returned to the caller.

    The trainer constructs it with the same two knobs as the wire engines
    (``bucket_size`` / ``batch_gather``) and sets ``is_master`` afterwards, exactly as with the
    NCCL/HCCL engines; ``send_weights`` then deposits the flush stream in :meth:`take_flushes`.
    """

    def __init__(self, bucket_size: int = 64 << 20, rollout_dtype: torch.dtype | None = None,
                 **kwargs) -> None:
        # DeltaSyncMixin takes encoding/batch_gather and forwards anything else to
        # object.__init__, so accept exactly the trainer's two keys here.
        batch_gather = kwargs.pop('batch_gather', 32)
        if kwargs:
            raise TypeError(f'DeltaCollectEngine does not read {sorted(kwargs)}; construct it with '
                            'bucket_size / batch_gather as the wire engines are.')
        super().__init__(batch_gather=batch_gather)
        self.bucket_size = int(bucket_size)
        self.rollout_dtype = torch.bfloat16 if rollout_dtype is None else rollout_dtype
        self.rank = 0
        self.is_master = False
        self._collected: list[tuple[list[tuple[str, torch.Tensor]], bool]] = []

    # ---- lifecycle (no rendezvous, no process group, no sockets) ----
    def prepare(self):
        return None

    def close(self) -> None:
        self.take_flushes()

    def take_flushes(self) -> list[tuple[list[tuple[str, torch.Tensor]], bool]]:
        """The collected ``(named_tensors, is_last)`` stream of the last round; consumed once."""
        stream, self._collected = self._collected, []
        return stream

    # ---- transport hooks ----
    @property
    def _wire_device(self):
        return 'cuda'

    def _ensure_wire(self) -> None:
        """No transport to initialize; kept for pipeline symmetry."""

    def _release_after_round(self, phase: str) -> None:
        """Nothing is held transport-side; the collected stream is the caller's to free."""

    # ---- publish overrides: collect instead of broadcasting ----
    def _publish_flush(self, flush, first: bool, is_last: bool) -> None:
        named = [
            ('__delta_spec__', _spec_tensor(
                {'encoding': self.encoding, 'params': [vars(p) for p in flush.params],
                 'checksum': int(flush.checksum)}, 'cpu')),
            ('__positions__', flush.positions_cpu.to('cpu')),
            ('__values__', flush.values_gpu.to('cpu')),
        ]
        self._collected.append((named, bool(is_last)))

    def _publish_values_flush(self, params, values, is_last: bool) -> None:
        # Same manifest the wire engines publish for a values-only (seed) flush: the dense
        # checksum over an empty positions blob, so the receiver verifies exactly the same way.
        from ..protocol.wire import checksum as _checksum

        empty_pos = torch.empty(0, dtype=torch.uint8)
        named = [
            ('__delta_spec__', _spec_tensor(
                {'encoding': 'dense', 'verify': False, 'is_last': is_last,
                 'params': [vars(p) for p in params],
                 'checksum': int(_checksum(empty_pos, values))}, 'cpu')),
            ('__values__', values.to('cpu')),
        ]
        self._collected.append((named, bool(is_last)))

    def _publish_terminal(self, first: bool) -> None:
        """Zero flushes this round: an empty stream tells the receiver nothing changed."""
