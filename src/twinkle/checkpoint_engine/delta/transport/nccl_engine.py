# Copyright (c) ModelScope Contributors. All rights reserved.
"""NCCL transport for the delta pipeline (the pipeline itself lives in :mod:`.pipeline`)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import torch

from ...nccl_checkpoint_engine import MasterMetadata, NCCLCheckpointEngine, _pg_broadcast
from twinkle.utils.network import find_free_port
# Re-exported for callers that import the pipeline internals from this module.
from .pipeline import (  # noqa: F401
    DeltaSyncMixin,
    _bucket_sliced,
    _FlushBucket,
    _FlushPiece,
    _GatherQueue,
    _prodshape,
    _ValuesPiece,
)

logger = logging.getLogger(__name__)


class DeltaShardedCheckpointEngine(DeltaSyncMixin, NCCLCheckpointEngine):
    """Sparse delta weight sync over NCCL, diffed on each rank's local shard.

    The transport-agnostic pipeline (``send_weights`` state machine, bucketing,
    gather batching, flush assembly) comes from :class:`DeltaSyncMixin`; this class
    is the NCCL transport: it reuses NCCLCheckpointEngine's group/zmq machinery,
    implements the four transport hooks on top of ``_pg_broadcast`` and holds the
    rollout-side ``receive_weights``.
    """

    wire_format = "delta_flush"

    def prepare(self) -> MasterMetadata | None:
        if not self._prepared:
            if self.is_master:
                self._start_zmq_server()
                self._nccl_store_host = self.ip
                self._nccl_store_port = find_free_port()
            self._prepared = True
        if self.is_master:
            return MasterMetadata(self.ip, self.listen_port, self._nccl_store_host, self._nccl_store_port)
        return None

    def close(self):
        self.rebuild_group = True
        self.finalize()

    def _broadcast(self, tensor):
        if tensor.numel():
            _pg_broadcast(self._pg, tensor)

    # ---- transport hooks ----
    @property
    def _wire_device(self):
        return "cuda"

    def _broadcast_flush(self, pos_u8, val_u8) -> None:
        if pos_u8 is not None:
            self._broadcast(pos_u8)
        self._broadcast(val_u8)

    def _release_after_round(self, phase: str) -> None:
        """NCCL keeps no per-round scratch memory."""

    # ---- rollout worker side ----
    def receive_weights(self, global_steps: int | None = None) -> Iterator[tuple[list[tuple[str, torch.Tensor]], bool]]:
        """Yield delta flushes for the rollout adapter to consume.

        Each ``(named_tensors, is_last)`` item contains a JSON manifest, optional
        int32 position bytes, and patch values. The generator keeps at most one
        received GPU flush live at a time.
        """
        assert self.rank > 0, "Rank 0 should not receive weights."
        applied = 0
        while True:
            self.socket.recv_string()
            meta = self.socket.recv_pyobj()
            if meta.get("terminal_empty"):
                break

            dense = meta.get("encoding") == "dense"
            val_dtype = getattr(torch, meta["val_dtype"])
            elem = torch.empty(0, dtype=val_dtype).element_size()
            val_u8 = torch.empty(meta["val_numel"] * elem, dtype=torch.uint8, device=self._wire_device)
            if dense:
                pos = None
                self._broadcast(val_u8)
            else:
                pos = torch.empty(meta["pos_numel"], dtype=torch.uint8, device=self._wire_device)
                self._broadcast(pos)
                self._broadcast(val_u8)
            val = val_u8.view(val_dtype)
            spec_bytes = json.dumps(meta["spec"]).encode()
            spec_t = torch.frombuffer(bytearray(spec_bytes), dtype=torch.uint8).to(self._wire_device)
            named = [("__delta_spec__", spec_t), ("__values__", val)]
            if pos is not None:
                named.insert(1, ("__positions__", pos))
            is_last = bool(meta["is_last"])
            yield named, is_last
            applied += 1
            del pos, val_u8, val, spec_t
            if is_last:
                break
        logger.info("delta recv v=%s flushes=%d (yielded to server adapter)", global_steps, applied)
