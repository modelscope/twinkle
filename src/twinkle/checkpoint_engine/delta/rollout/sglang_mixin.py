# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout-side delta lifecycle for SGLang samplers.

Unlike the vLLM lane (:mod:`.mixin`), there are no receiver actors and no external NCCL group:
SGLang's own ``update_weights_from_tensor`` serializes each flush into every TP worker process
as CUDA IPC handles, and the custom weight loader (:mod:`.sglang_loader`) decodes and
masked-applies it inside the worker. The sampler actor only drives lifecycle and forwards the
flush stream the manager hands it (see :class:`~..manager.CheckpointEngineManager`): no
``delta_sampler_metadata``, no transport endpoint, no sidecars.
"""

from __future__ import annotations
from typing import Any

from twinkle import remote_function
from ..config import DeltaSyncConfig, DeltaWorkerCapabilities
from .sglang_loader import LOADER_FQN


class SGLangDeltaSamplerMixin:
    """SGLang sampler-side delta lifecycle RPCs (in-process receive, no receiver actors)."""

    _delta_config: DeltaSyncConfig
    _delta_job_failed: bool = False

    @remote_function(dispatch="all", lazy_collect=False)
    def validate_delta_sampler(self, config: DeltaSyncConfig) -> DeltaWorkerCapabilities:
        self._delta_config = config
        if self._delta_job_failed:
            raise RuntimeError("delta sampler failed; restart the entire training/rollout job before reseeding")
        if config.free_cache_engine and not self.engine._memory_saver_enabled:
            raise ValueError(
                "delta KV cache release requires enable_memory_saver=True (the "
                "swift CLI sets it with --weight_sync_backend delta_sharded) or "
                "free_cache_engine=False"
            )
        loaders = self._sglang_engine_kwargs.get("custom_weight_loader")
        if LOADER_FQN not in (loaders or []):
            raise ValueError(
                "delta_sharded on sglang requires the delta loader registered in the "
                "engine's server args: pass custom_weight_loader with "
                f"{LOADER_FQN!r} (the swift CLI sets it with --weight_sync_backend "
                "delta_sharded)."
            )
        return {"tp": self.engine.tensor_parallel_size, "device": "cuda"}

    @remote_function(dispatch="all", lazy_collect=False)
    def begin_delta_sync(self) -> None:
        # SGLang serializes ``update_weights_from_tensor`` against generation inside the engine,
        # but NOT ``release_memory_occupation`` (what ``sleep(level=1)`` calls): the scheduler
        # asserts a fully idle server there and the worker dies if any request is still running
        # or queued. So drain generation first through the same gate the full-sync lane uses --
        # ``pause_generation`` blocks new ``sample`` calls and waits for the running ones -- and
        # only then free the KV region. ``publish_delta_sampler`` (or ``fail_delta_sampler`` on
        # the error path) reopens the gate. SGLang's sleep level 1 keeps the weights resident, so
        # unlike vLLM there is nothing to wake back before the flush stream arrives.
        self.pause_generation()

        async def begin():
            if self._delta_config.free_cache_engine:
                await self.engine.sleep(level=1)

        self._run_in_loop(begin())

    @remote_function(dispatch="all", lazy_collect=False)
    def publish_delta_sampler(self) -> None:
        async def publish():
            if self._delta_config.free_cache_engine:
                await self.engine.wake_up(tags=["kv_cache"])
            await self.engine.reset_prefix_cache()

        self._run_in_loop(publish())
        self.resume_generation()

    @remote_function(dispatch="all", lazy_collect=False)
    def fail_delta_sampler(self) -> None:
        # The weights may be half updated: refuse every later delta round. ``begin_delta_sync``
        # drained generation before the update, so nothing of ours is in flight here; reopen the
        # gate (idempotent) so teardown is not stuck behind a paused sampler. The manager raises
        # out of the failed sync and the job stops.
        self._delta_job_failed = True
        self.resume_generation()

    @remote_function(dispatch="all", lazy_collect=True, offload_blocking=True)
    def receive_delta_flushes(self, flushes: list[tuple[list[tuple[str, Any]], bool]]) -> dict[str, bool]:
        """Apply the trainer's collected flush stream (see :mod:`..transport.collect_engine`).

        An empty stream is a zero-change round: the weights are exactly what the rollout already
        holds, so nothing is forwarded and the prefix cache stays valid.
        """
        if self._delta_job_failed:
            raise RuntimeError("delta sampler failed; restart the entire training/rollout job before reseeding")
        changed = bool(flushes)

        # Env-gated transport fingerprint at the trainer->sampler RPC boundary: pair each received
        # flush's weight payload against the collect engine's published bytes by (name, occurrence).
        import os

        from twinkle.checkpoint_engine.weight_keys import record_delta_flush_fingerprint

        seq = getattr(self, "_delta_fp_recv_seq", 0)
        for named, _is_last in flushes:
            record_delta_flush_fingerprint("recv", seq, named, os.getpid())
            seq += 1
        self._delta_fp_recv_seq = seq

        async def receive():
            await self.engine.update_weights_delta(flushes)
            if changed:
                await self.engine.reset_prefix_cache()

        self._run_in_loop(receive())
        return {"changed": changed}

    @remote_function(dispatch="all", lazy_collect=False)
    def close_delta_sampler(self) -> None:
        # No transport to release (no receiver actors, no process group, no sockets); the next
        # full seed rebuilds the session state by itself.
        pass
