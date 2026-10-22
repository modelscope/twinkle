# Copyright (c) ModelScope Contributors. All rights reserved.
"""The contract and shared defaults of the transformers-side training strategies.

A strategy is what ``TrainableModel`` drives a backend's parallelism machinery
through. This base class owns the three layers of that contract:

1. **Required, engine-specific** (``@abstractmethod``): wrapping, state-dict
   collection, optimizer checkpoints -- every strategy implements these against
   its own engine, and a concrete strategy missing one of them fails at
   INSTANTIATION with the missing-method list, not as a bare ``AttributeError``
   deep inside the model layer.
2. **Required, with a universal default**: methods every strategy can answer
   with a trivial default (``init_kwargs`` -> no extra kwargs, EP hooks ->
   pass-through, ``pretrained_load_context`` -> nullcontext) live here once
   instead of being re-declared per strategy.
3. **Optional capabilities**: what the model layer probes before use and what a
   strategy without it must refuse loudly (``offload_to_cpu`` /
   ``reload_to_gpu``, required by colocation).

``MegatronStrategy`` (model/megatron/strategy/megatron.py) does NOT inherit this
base and does not implement most of this face -- and does not need to: its model
is a chunk list whose state-dict collection and optimizer persistence go through
the bridge backend and mcore's dist-checkpoint (MegatronModel never calls
``get_full_state_dict`` / ``save_optimizer_checkpoint`` / ``pretrained_load_context``
on its strategy). What the two families DO share is the smaller contact surface
``MegatronModel`` actually calls (wrap/unwrap, offload/reload, reduce_loss,
the mcore-specific hooks). A second megatron-side strategy, if one ever appears,
would take ``MegatronStrategy`` itself as the reference face.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
import os
from typing import Any, Dict, Optional

# Env flags transformers/accelerate read to pick their FSDP-aware loading path (rank0 load,
# meta-device placement, no full CPU materialization per rank).
_FSDP_EFFICIENT_LOADING_ENV = {
    'ACCELERATE_USE_FSDP': 'true',
    'FSDP_CPU_RAM_EFFICIENT_LOADING': 'true',
}


@contextmanager
def fsdp_pretrained_load_context(enabled: bool):
    """Enable the env flags required for transformers FSDP-aware loading when needed.

    A load-scoped transaction: the flags are restored (or popped) even when loading raises,
    so later imports in the same process never mistake themselves for FSDP-aware loaders.
    """
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


class TransformerStrategyBase(ABC):
    """Required face + shared defaults for accelerate/deepspeed/native_fsdp strategies."""

    # ── layer 1: required, engine-specific ─────────────────────────────────

    @abstractmethod
    def wrap_model(self, model, *args):
        """Attach the parallelism machinery (FSDP/ZeRO/...), returning the wrapped model."""
        ...

    @abstractmethod
    def unwrap_model(self, model):
        """Peel the parallelism wrappers off, returning the underlying module."""
        ...

    @abstractmethod
    def get_full_state_dict(self, model) -> Dict[str, Any]:
        """Collect the full state dict (parameters and persistent buffers)."""
        ...

    @abstractmethod
    def load_full_state_dict(self, model, state_dict) -> None:
        """Load a full (non-sharded) state dict into the model in-place."""
        ...

    @abstractmethod
    def get_adapter_state_dict(self, model, adapter_name: str) -> Dict[str, Any]:
        """Collect LoRA parameters, trained modules_to_save copies and their buffers."""
        ...

    @abstractmethod
    def save_optimizer_checkpoint(self, model, optimizer, output_path: str, *,
                                  param_name_mapping: Optional[Dict[str, str]] = None) -> None:
        """Persist optimizer state under ``output_path``."""
        ...

    @abstractmethod
    def load_optimizer_checkpoint(self, model, optimizer, input_path: str, *,
                                  param_name_mapping: Optional[Dict[str, str]] = None) -> None:
        """Restore optimizer state from ``input_path``."""
        ...

    # ── layer 2: required, with a universal default ────────────────────────

    def init_kwargs(self) -> Dict[str, Any]:
        """Extra kwargs the model construction call needs for this strategy.

        Strategies without engine-specific load kwargs return an empty dict.
        """
        return {}

    def capture_pre_ep_state_if_needed(self, model, *, enable_ep: bool) -> None:
        """Snapshot pre-expert-parallel weights; only needed where EP rewrites modules."""
        return

    def prepare_adapter_config(self, config_or_dir, *, enable_ep: bool):
        """Adjust a PEFT config for expert parallelism; pass-through unless EP-aware."""
        return config_or_dir

    def pretrained_load_context(self):
        """Context manager active while ``from_pretrained`` loads weights."""
        return nullcontext()

    # ── layer 3: optional capabilities ─────────────────────────────────────

    def offload_to_cpu(self, model, optimizer=None) -> None:
        """Weight offload for colocation; strategies without it must refuse loudly."""
        raise NotImplementedError(
            f'{type(self).__name__} does not support weight offload, which the colocation '
            'flow requires. Use a strategy with offload support (accelerate / megatron).')

    def reload_to_gpu(self, model, optimizer=None) -> None:
        """Weight offload for colocation; strategies without it must refuse loudly."""
        raise NotImplementedError(
            f'{type(self).__name__} does not support weight offload, which the colocation '
            'flow requires. Use a strategy with offload support (accelerate / megatron).')
