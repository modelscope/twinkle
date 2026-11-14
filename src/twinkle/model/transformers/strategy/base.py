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

    def _probe_load_precision(self, model, state_dict, *, after):
        """Report whether a full-state-dict load preserves the checkpoint's precision.

        Gated by ``TWINKLE_LOAD_PRECISION_PROBE`` (a path prefix). A checkpoint saved from a
        float32 master holds values that are almost never representable in bf16, so the fraction
        of loaded elements that ARE bf16-representable is a direct read-out of a downcast: near
        zero means full precision survived, near one means the master was rounded to bf16. The
        destination dtype is reported alongside because a bf16 destination at load time followed
        by a later promotion to float32 leaves exactly that signature -- float32 parameters
        carrying bf16-rounded values -- which no post-hoc inspection of the parameters can
        distinguish from a faithful load of genuinely bf16 weights.
        """
        prefix = os.environ.get('TWINKLE_LOAD_PRECISION_PROBE')
        if not prefix:
            return
        import json

        # torch is not a module-level dependency of this abstract base; os already is, and
        # re-importing it here would make it function-local and shadow the module binding that
        # the guard above just used.
        import torch

        # Sample the parameters the load actually writes, not the first ones the model happens to
        # expose. An adapter checkpoint carries only the LoRA keys, so sampling in
        # ``named_parameters()`` order measures the frozen base -- bfloat16 by construction, hence
        # trivially 100% bfloat16-representable -- and reports a clean load over a truncated one.
        targets = set(state_dict.keys())
        candidates = [(name, param) for name, param in model.named_parameters() if name in targets]
        if not candidates:
            candidates = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
        if not candidates:
            candidates = list(model.named_parameters())
        sampled = 0
        representable = 0
        total = 0
        for _, param in candidates:
            if sampled >= 8:
                break
            local = param.to_local() if hasattr(param, 'to_local') else param
            if local.numel() == 0 or not local.is_floating_point():
                continue
            sampled += 1
            flat = local.detach().float().flatten()
            total += flat.numel()
            representable += int((flat.to(torch.bfloat16).to(torch.float32) == flat).sum().item())
        record = {
            'rank': int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0),
            'phase': 'after' if after else 'before',
            # The destination dtypes that matter are those of the parameters being written; over the
            # whole model they mix the frozen base in and hide which side the load landed on.
            'dest_dtypes': sorted({str(p.dtype) for _, p in candidates}),
            'src_dtypes': sorted({str(v.dtype) for v in state_dict.values() if torch.is_tensor(v)}),
            'targeted': len(candidates),
            'params_sampled': sampled,
            'bf16_representable_ratio': (representable / total) if total else None,
        }
        with open(f'{prefix}.rank{record["rank"]}.jsonl', 'a') as handle:
            handle.write(json.dumps(record) + '\n')

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

    def offload_to_cpu(self, model, optimizer=None, *, offload_model: bool = True) -> None:
        """Weight offload for colocation; strategies without it must refuse loudly.

        ``offload_model=False`` asks to leave parameters resident and move only ``optimizer``
        when one is passed. Strategies that cannot move either piece still refuse.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not support weight offload, which the colocation '
            'flow requires. Use a strategy with offload support (accelerate / native FSDP / deepspeed / megatron).')

    def reload_to_gpu(self, model, optimizer=None) -> None:
        """Bring back whatever :meth:`offload_to_cpu` moved; a no-op when nothing was moved."""
        from twinkle.model.offload import offload_record
        if not any(offload_record(self)):
            return
        raise NotImplementedError(
            f'{type(self).__name__} does not support weight offload, which the colocation '
            'flow requires. Use a strategy with offload support (accelerate / native FSDP / deepspeed / megatron).')

    def optimizer_checkpoint_exists(self, output_dir: str) -> bool:
        """Whether this strategy's optimizer checkpoint is present in ``output_dir``.

        The artifact's layout belongs to the strategy that wrote it: the default is a single
        ``optimizer.pt``, while DeepSpeed's ``engine.save_checkpoint`` writes its own directory -- a
        ``latest`` tag file plus ``global_step<N>/`` -- and never creates that filename. A caller that
        hardcodes ``optimizer.pt`` therefore reads every DeepSpeed checkpoint as incomplete and refuses
        a resume that would have worked.
        """
        import os

        return os.path.exists(os.path.join(output_dir, 'optimizer.pt'))

    def owns_update_step(self) -> bool:
        """Whether this strategy, rather than the training loop, drives the parameter update.

        Accelerate's DeepSpeed integration hands back a ``DeepSpeedOptimizerWrapper`` whose ``step``
        and ``zero_grad`` are ``pass``: the contract is that the caller drives the engine
        (``engine.backward`` then ``engine.step``), which performs the gradient-accumulation
        accounting, the clipping, the ZeRO optimizer step and the zeroing. A loop that instead calls
        ``loss.backward()`` and ``optimizer.step()`` therefore trains without ever updating a weight,
        and the engine's own step counter never advances -- so the checkpoint it writes is tagged
        ``global_step0`` whatever the run reached. Strategies that own the update say so here, and
        the loop then routes its backward and its gradient clipping through the hooks below.
        """
        return False

    def engine_backward(self, model, loss, *, sync_gradients: bool) -> None:
        """Run the backward for one micro-batch, applying the update at the accumulation boundary."""
        raise NotImplementedError(f'{type(self).__name__} does not own the update step.')

    def engine_grad_norm(self, model):
        """The gradient norm the engine computed while stepping, in place of clipping a second time."""
        raise NotImplementedError(f'{type(self).__name__} does not own the update step.')

    def engine_counters(self, model) -> Dict[str, int]:
        """The owning engine's own step counters, or an empty dict when no engine owns the update.

        An engine that owns the update decides at which micro-batch it applies one, so its counters --
        not the training loop's -- determine whether the first update after a resume aggregates the same
        micro-batches the source run aggregated. A resume that restores the weights and the optimizer
        state but not these counters shifts the accumulation boundary: the reported loss of that step
        still looks right while the update it produced is not.
        """
        return {}

    def engine_master_digest(self, model) -> Dict[str, Any]:
        """Digest of the float32 master partitions this strategy's optimizer actually updates.

        Offload tiers keep the trained precision in a flat CPU partition that the model's parameters
        are only a bfloat16 view of, so digesting the parameters cannot show whether an update landed:
        they read the same values before and after the step. Sampling is spread across the whole
        partition rather than taken from its head, because a prefix sample is exactly what let a
        partition that differed past its first elements report as identical.
        """
        return {}
