# Copyright (c) ModelScope Contributors. All rights reserved.
"""RL iteration metrics with explicit token, time and allocation scopes.

This module is deliberately independent of torch/Ray. Counts are unpadded sequence
lengths, once per trajectory in an RL batch, NOT once per optimizer update/replay.
Timing is supplied by the orchestration layer; a formula cannot fix a mismatched
measurement boundary. Callers supply explicit iteration and phase boundaries.
"""

from __future__ import annotations
import math
import threading
import time
import uuid
from pathlib import Path

import json


def compute_rl_performance_metrics(
    prompt_lengths, response_lengths, timing_raw, n_gpus, *, prompt_width=None, response_width=None
):
    """Throughput, lengths and per-token timing with explicit denominators.

    Ragged batches use the batch maximum as padded width. ``clip_ratio`` therefore
    counts rows equal to that width, not rows that hit the configured generation
    cap. Explicit widths also support fixed-width batches.
    """
    prompts, responses = list(prompt_lengths), list(response_lengths)
    if len(prompts) != len(responses) or not prompts:
        raise ValueError("expected a nonempty batch of paired prompt/response lengths")
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in prompts + responses):
        raise ValueError("token lengths must be nonnegative integers")
    elapsed = timing_raw["step"]
    if not isinstance(n_gpus, int) or isinstance(n_gpus, bool) or n_gpus < 1:
        raise ValueError("n_gpus must count allocated physical devices, including rollout")
    if elapsed <= 0 or any(not math.isfinite(t) or t < 0 for t in timing_raw.values()):
        raise ValueError("timings must be finite and nonnegative, with positive step time")
    p, r = sum(prompts), sum(responses)
    total = p + r
    out = {
        "perf/total_num_tokens": total,
        "perf/time_per_step": elapsed,
        "perf/throughput": total / (elapsed * n_gpus),
        "perf/num_gpus": n_gpus,
        "perf/num_samples": len(prompts),
        "perf/num_prompt_tokens": p,
        "perf/num_response_tokens": r,
        "perf/throughput_total": total / elapsed,
        "perf/response_tokens_per_second": r / elapsed,
        "perf/samples_per_second": len(prompts) / elapsed,
        **{f"timing_s/{name}": t for name, t in timing_raw.items()},
    }
    denominators = {"gen": r, **dict.fromkeys(("ref", "values", "adv", "update_critic", "update_actor"), total)}
    for name, count in denominators.items():
        if name in timing_raw:
            out[f"timing_per_token_ms/{name}"] = timing_raw[name] * 1000 / count if count else 0.0
    for name, lengths, width in (
        ("prompt_length", prompts, prompt_width),
        ("response_length", responses, response_width),
    ):
        width = max(lengths) if width is None else width
        if width < max(lengths):
            raise ValueError("padded width cannot be smaller than an actual sequence")
        out.update(
            {
                f"{name}/mean": sum(lengths) / len(lengths),
                f"{name}/min": min(lengths),
                f"{name}/max": max(lengths),
                f"{name}/clip_ratio": sum(n == width for n in lengths) / len(lengths),
            }
        )
    active = [n for n in responses if n > 0]
    width = max(responses) if response_width is None else response_width
    out["response/aborted_ratio"] = 1 - len(active) / len(responses)
    out.update(
        {
            "response_length_non_aborted/mean": sum(active) / len(active) if active else float("nan"),
            "response_length_non_aborted/min": min(active) if active else float("nan"),
            "response_length_non_aborted/max": max(active) if active else float("nan"),
            "response_length_non_aborted/clip_ratio": sum(n == width for n in active) / len(active)
            if active
            else float("nan"),
        }
    )
    return out


def _json_value(value):
    # Nonfinite numbers become JSON null, never invalid NaN literals.
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


class JsonlMetricWriter:
    """Incremental, append-only local output with a {step, data} envelope.

    A new run_id on every construction disambiguates resumed/repeated runs in the
    same directory. Each record is flushed; this does not claim power-loss fsync.
    """

    def __init__(self, path, *, run_id=None):
        self.path = Path(path)
        self.run_id = run_id or uuid.uuid4().hex
        self._stream = None
        self._lock = threading.Lock()

    def write(self, step, data, **metadata):
        row = _json_value(dict(step=step, data=data, run_id=self.run_id, timestamp=time.time(), **metadata))
        line = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        with self._lock:
            if self._stream is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._stream = self.path.open("a", encoding="utf-8")
            self._stream.write(line)
            self._stream.flush()

    def close(self):
        with self._lock:
            if self._stream is not None:
                self._stream.close()
                self._stream = None


def accelerator_memory_metrics():
    """Per-worker allocator high-water marks and host-used RAM.

    These are lifetime peaks (no reset), not device-wide memory usage. CPU RAM is
    host-wide virtual_memory().used, NOT process RSS. Unsupported counters are
    absent, not zero. This helper launches no accelerator work.
    """
    import psutil
    import torch

    from twinkle import Platform

    device = getattr(torch, Platform.device_prefix(), None)
    result = {"perf/cpu_memory_used_gb": psutil.virtual_memory().used / 1024**3}
    if device is not None and getattr(device, "is_available", lambda: False)():
        # The update must have actually finished, not just queued optimizer kernels.
        device.synchronize()
        for name in ("max_memory_allocated", "max_memory_reserved"):
            fn = getattr(device, name, None)
            if fn is not None:
                result[f"perf/{name}_gb"] = fn() / 1024**3
    return result
