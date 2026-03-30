# Copyright (c) ModelScope Contributors. All rights reserved.
import contextlib
import os

_FSDP_EFFICIENT_LOADING_ENV = {
    'ACCELERATE_USE_FSDP': 'true',
    'FSDP_CPU_RAM_EFFICIENT_LOADING': 'true',
}


@contextlib.contextmanager
def fsdp_pretrained_load_context(enabled: bool):
    """Enable the env flags required for transformers FSDP-aware loading when needed."""
    if not enabled:
        yield
        return

    saved_env = {key: os.environ.get(key) for key in _FSDP_EFFICIENT_LOADING_ENV}
    os.environ.update(_FSDP_EFFICIENT_LOADING_ENV)
    try:
        yield
    finally:
        for key, old_val in saved_env.items():
            if old_val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_val
