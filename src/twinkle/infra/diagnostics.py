# Copyright (c) ModelScope Contributors. All rights reserved.

from __future__ import annotations

import time
import threading
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional


import sys

def _diag_emit(msg: str, *args) -> None:
    """Internal diagnostic logger. Supports printf-style formatting."""
    try:
        if args:
            msg = msg % args
    except Exception:
        # fallback: join reprs
        msg = str(msg) + " " + " ".join(repr(a) for a in args)
    try:
        print(msg, flush=True)
    except Exception:
        pass
LOGGER = logging.getLogger("twinkle.infra.diagnostics")


@dataclass
class DiagnosticsConfig:
    """
    Framework-level diagnostics for Ray execution.

    enabled:
      - False by default to avoid behavior change for existing users.

    preflight:
      - Checks if Ray sees GPU/CPU and whether enough resources are available.
      - Optional minimal GPU probe task to disambiguate "resource starvation" vs "logic".

    watchdog:
      - Periodically logs Ray cluster/available resources.
      - Emits warnings when starvation persists (useful for "hang" diagnosis).
    """
    enabled: bool = False

    # preflight
    need_gpu: float = 1.0
    need_cpu: float = 1.0
    gpu_probe: bool = True
    probe_timeout_s: int = 90

    # watchdog
    watchdog: bool = True
    watchdog_interval_s: int = 30
    warn_after_s: int = 60
    hard_fail_after_s: int = 300
    # NOTE: hard-fail is LOG-only by default to avoid os._exit side effects.


def _ray_snapshot() -> Dict[str, Any]:
    import ray
    return {
        "cluster_resources": ray.cluster_resources(),
        "available_resources": ray.available_resources(),
    }


def _is_starvation(snap: Dict[str, Any], *, need_gpu: float, need_cpu: float) -> bool:
    cr = snap.get("cluster_resources") or {}
    ar = snap.get("available_resources") or {}
    cluster_gpu = float(cr.get("GPU", 0.0))
    avail_gpu = float(ar.get("GPU", 0.0))
    avail_cpu = float(ar.get("CPU", 0.0))

    # If Ray doesn't see GPU at all, any GPU workload will hang.
    if cluster_gpu <= 0.0:
        return True
    if avail_gpu < need_gpu:
        return True
    if avail_cpu < need_cpu:
        return True
    return False


def _gpu_probe(timeout_s: int) -> Dict[str, Any]:
    import ray

    @ray.remote(num_gpus=1)
    def _probe():
        import os
        import torch
        out = {
            "pid": os.getpid(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
        }
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            out["cuda_name0"] = torch.cuda.get_device_name(0)
        return out

    ref = _probe.remote()
    start = time.time()
    while True:
        ready, _ = ray.wait([ref], timeout=1.0)
        if ready:
            return ray.get(ready[0])
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"GPU probe timed out after {timeout_s}s (resource starvation likely)."
            )


def preflight_or_raise(cfg: Optional[DiagnosticsConfig]) -> None:
    if not cfg or not cfg.enabled:
        return

    snap = _ray_snapshot()
    _diag_emit("[preflight] cluster=%s", snap["cluster_resources"])
    _diag_emit("[preflight] available=%s", snap["available_resources"])

    if _is_starvation(snap, need_gpu=cfg.need_gpu, need_cpu=cfg.need_cpu):
        raise RuntimeError(
            "Twinkle diagnostics preflight failed: resource starvation or Ray cannot see GPU. "
            f"need(GPU>={cfg.need_gpu}, CPU>={cfg.need_cpu}) "
            f"cluster={snap['cluster_resources']} available={snap['available_resources']}. "
            "This indicates infra/resource/placement issues, not algorithm logic."
        )

    if cfg.gpu_probe:
        info = _gpu_probe(timeout_s=cfg.probe_timeout_s)
        LOGGER.info("[preflight] gpu_probe_ok=%s", info)


class Watchdog:
    def __init__(self, cfg: Optional[DiagnosticsConfig]):
        self.cfg = cfg
        self._t: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._start_ts = time.time()

    def start(self) -> None:
        if not self.cfg or not self.cfg.enabled or not self.cfg.watchdog:
            return
        if self._t is not None:
            return
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        LOGGER.info("[watchdog] started interval=%ss", self.cfg.watchdog_interval_s)

    def stop(self) -> None:
        if self._t is None:
            return
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            elapsed = int(time.time() - self._start_ts)
            try:
                snap = _ray_snapshot()
                if _is_starvation(snap, need_gpu=self.cfg.need_gpu, need_cpu=self.cfg.need_cpu):
                    if elapsed >= self.cfg.warn_after_s:
                        LOGGER.warning(
                            "[watchdog] starvation elapsed=%ss cluster=%s available=%s",
                            elapsed, snap["cluster_resources"], snap["available_resources"],
                        )
                    if elapsed >= self.cfg.hard_fail_after_s:
                        LOGGER.error(
                            "[watchdog] hard_fail reached (%ss). Starvation is not algorithm logic. "
                            "Recommend terminate job / free GPUs / restart Ray.",
                            elapsed,
                        )
            except Exception as e:
                LOGGER.warning("[watchdog] snapshot failed: %s: %s", type(e).__name__, e)
            self._stop.wait(self.cfg.watchdog_interval_s)
