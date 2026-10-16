# Copyright (c) ModelScope Contributors. All rights reserved.
"""Process-global PyTorch setup, isolated from the infra / Ray control flow.

Twinkle seeds the RNG, toggles tf32 and pins a few CUDA / tokenizer env vars in *every* process that
runs compute: the driver (which, in local/torchrun mode, is itself the compute process) and each Ray
worker. All of these are per-process globals, so they must be re-applied wherever the compute happens
rather than once on the driver that issued ``initialize``.

Keeping this in its own module stops torch-specific concerns from leaking into ``infra`` (mode /
device group / remote dispatch) or ``_ray`` (worker placement). The whole knob set crosses the process
boundary as ONE JSON blob in the ``EXTRA_TORCH_KWARGS`` env var, so adding a new torch knob never
means threading another argument -- or another env var -- through ``initialize``, ``create_workers``
and the worker bootstrap.
"""
import json
import os
from typing import Any, Dict, Optional

from twinkle.utils import framework_util

#: Env var carrying the JSON-encoded torch-init knobs across the process boundary (driver -> Ray worker
#: runtime_env). One blob replaces the old separate TWINKLE_SEED / TWINKLE_FULL_DETERMINISM /
#: TWINKLE_TF32 vars.
EXTRA_TORCH_KWARGS_ENV = 'EXTRA_TORCH_KWARGS'

_DEFAULT_SEED = 42

#: Per-knob defaults, used for any key a caller leaves out.
_DEFAULTS: Dict[str, Any] = {
    'seed': _DEFAULT_SEED,
    'full_determinism': False,
    # None leaves torch's own tf32 matmul/conv default untouched.
    'tf32': None,
}


def normalize(extra_torch_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a full knob set: defaults for anything *extra_torch_kwargs* leaves out.

    An explicitly-passed value (including ``tf32=None`` or ``seed=None``) is kept as-is; only missing
    keys take a default.
    """
    kwargs = dict(_DEFAULTS)
    if extra_torch_kwargs:
        kwargs.update(extra_torch_kwargs)
    return kwargs


def to_env_value(extra_torch_kwargs: Optional[Dict[str, Any]]) -> str:
    """JSON-encode the knob set into the value stored under ``EXTRA_TORCH_KWARGS``."""
    return json.dumps(normalize(extra_torch_kwargs))


def from_env(fallback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Decode ``EXTRA_TORCH_KWARGS``, filling anything absent from *fallback* then the defaults.

    A Ray worker reaches its knobs this way: it inherited the env var through its runtime_env rather
    than the driver's in-process globals.
    """
    kwargs = normalize(fallback)
    raw = os.environ.get(EXTRA_TORCH_KWARGS_ENV)
    if raw:
        kwargs.update(json.loads(raw))
    return kwargs


def _apply_tf32(allow: Optional[bool]) -> None:
    """Set torch's tf32 matmul/conv switches; ``None`` leaves torch's own default untouched."""
    if allow is None:
        return
    import torch
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow


def apply(extra_torch_kwargs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Seed the RNG and set determinism / tf32 in the CURRENT process; return the normalized knob set.

    ``initialize`` calls this on the driver (and, in local/torchrun mode, that same process does the
    compute). A ``seed`` of ``None`` skips seeding entirely -- determinism rides ``seed_everything``, so
    it is skipped too.
    """
    kwargs = normalize(extra_torch_kwargs)
    if kwargs['seed'] is not None:
        framework_util.seed_everything(kwargs['seed'], kwargs['full_determinism'])
    _apply_tf32(kwargs['tf32'])
    return kwargs


def apply_in_worker(fallback: Optional[Dict[str, Any]] = None) -> None:
    """Re-apply the knob set inside a Ray worker (or a driver-hosted local component).

    Reads ``EXTRA_TORCH_KWARGS`` first (falling back to *fallback*), does what :func:`apply` does, then
    pins the worker-only env vars below. Those must land before the model is constructed: the CUDA driver
    latches ``CUDA_DEVICE_MAX_CONNECTIONS`` when the context is created, so ``setdefault`` (not
    assignment) leaves room for ``MegatronStrategy.apply_process_env`` to raise it for a sharded
    data-parallel wrapper while there is still time. It cannot be delegated to the strategy from here --
    infra sits below the model layer and does not import it.
    """
    apply(from_env(fallback))
    if os.environ.get('WORKER_NAME'):
        # Depress megatron's warnings and cut overhead; a sharded (FSDP) wrapper raises it later.
        os.environ.setdefault('CUDA_DEVICE_MAX_CONNECTIONS', '1')
        # Cap the compile thread pool torch would otherwise grow without bound.
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'
        # Keep tokenizers in its parallel mode.
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def seed_from_env(default: int = _DEFAULT_SEED) -> int:
    """The configured RNG seed from ``EXTRA_TORCH_KWARGS``, or *default* when it is absent or None."""
    seed = from_env().get('seed')
    return default if seed is None else int(seed)
