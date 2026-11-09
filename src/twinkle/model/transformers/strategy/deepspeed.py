# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from contextlib import nullcontext
from datetime import timedelta
from typing import Any, Dict, Literal, Mapping, Optional

import torch.nn as nn

from twinkle.model.state_dict_utils import collect_local_state, is_saveable_adapter_key
from twinkle.patch import apply_context, apply_patch
# Re-export for convenience
from twinkle.patch.deepspeed_patches import DeepSpeedLeafModulesPatch  # noqa: F401
from twinkle.patch.deepspeed_patches import (DeepSpeedHookReorderPatch, DeepSpeedModulesToSavePatch,
                                             DeepSpeedParamWrapperPatch)
from .base import TransformerStrategyBase

_OFFLOAD_BUFFER = '_offload_buffer'


def _is_zero_optimizer(obj) -> bool:
    """A ZeRO optimizer owns ``offload_states`` / ``reload_states``. A torch Adam does not."""
    return obj is not None and hasattr(obj, 'offload_states') and hasattr(obj, 'reload_states')


def _resolve_zero(model, optimizer):
    """The ZeRO optimizer behind an Accelerate wrapper, the object itself, or the engine."""
    candidates = []
    if optimizer is not None:
        candidates.append(getattr(optimizer, 'optimizer', None))
        candidates.append(optimizer)
    candidates.append(getattr(model, 'optimizer', None))
    for candidate in candidates:
        if _is_zero_optimizer(candidate):
            return candidate
    return None


def _resolve_engine(model):
    """The DeepSpeedEngine to checkpoint through: the prepared model itself, or behind a wrapper.

    ``unwrap_model`` peels the engine OFF (it returns the underlying nn.Module), so calling
    ``engine.save_checkpoint`` on its result reaches the base model and raises. The prepared model
    IS the engine in the common case; walk the shallow ``.module`` chain otherwise.
    """
    from deepspeed.runtime.engine import DeepSpeedEngine

    candidate = model
    while candidate is not None:
        if isinstance(candidate, DeepSpeedEngine):
            return candidate
        candidate = getattr(candidate, 'module', None)
    raise ValueError(
        f'DeepSpeed engine checkpointing needs the DeepSpeedEngine, got {type(model).__name__} -- the '
        'model was not prepared by the DeepSpeed strategy.')


def _resident_device(module):
    import torch

    for param in module.parameters():
        target = getattr(param, 'ds_tensor', None)
        if torch.is_tensor(target) and target.numel() > 0:
            return target.device
        if param.numel() > 0:
            return param.device
    return torch.device('cpu')


def _offload_parameters(module) -> bool:
    """4.x placement: ZeRO-3 keeps the shard on ``ds_tensor``; everyone else uses ``param.data``."""
    import torch

    moved = False
    for param in module.parameters():
        target = getattr(param, 'ds_tensor', None)
        if not torch.is_tensor(target):
            target = param
        if target.numel() > 0 and target.device.type != 'cpu':
            target.data = target.data.to('cpu', non_blocking=False)
            moved = True
        grad = getattr(param, 'grad', None)
        if torch.is_tensor(grad) and grad.numel() > 0 and grad.device.type != 'cpu':
            # Rebind storage. Assigning param.grad rejects a CPU tensor while param stays on GPU,
            # which is the ZeRO-3 case: the shard moved, the parameter object did not.
            grad.data = grad.data.to('cpu', non_blocking=False)
            moved = True
    return moved


def _reload_parameters(module, device) -> None:
    import torch

    for param in module.parameters():
        target = getattr(param, 'ds_tensor', None)
        if not torch.is_tensor(target):
            target = param
        if target.numel() > 0 and target.device.type == 'cpu':
            target.data = target.data.to(device, non_blocking=False)
        grad = getattr(param, 'grad', None)
        if torch.is_tensor(grad) and grad.numel() > 0 and grad.device.type == 'cpu':
            grad.data = grad.data.to(device, non_blocking=False)


def _state_owner(zero, optimizer):
    """Adam moments live on the inner torch optimizer once ZeRO has wrapped it."""
    if zero is not None:
        inner = getattr(zero, 'optimizer', None)
        if inner is not None and hasattr(inner, 'state'):
            return inner
    return optimizer


def _offload_optimizer_tensors(optimizer) -> bool:
    """Point each GPU state tensor at a CPU buffer. Same rebind swift 4.x uses for DeepSpeed."""
    import torch

    if optimizer is None or not hasattr(optimizer, 'state'):
        return False
    moved = False
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if key.endswith(_OFFLOAD_BUFFER) or not torch.is_tensor(value):
                continue
            if value.numel() == 0 or value.device.type == 'cpu':
                continue
            buf_key = f'{key}{_OFFLOAD_BUFFER}'
            buf = state.get(buf_key)
            if not torch.is_tensor(buf) or buf.shape != value.shape or buf.dtype != value.dtype:
                buf = torch.empty_like(value, device='cpu')
                state[buf_key] = buf
            buf.copy_(value, non_blocking=False)
            value.data = buf
            moved = True
    return moved


def _reload_optimizer_tensors(optimizer, device) -> None:
    import torch

    if optimizer is None or not hasattr(optimizer, 'state') or device is None:
        return
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if key.endswith(_OFFLOAD_BUFFER) or not torch.is_tensor(value):
                continue
            buf = state.get(f'{key}{_OFFLOAD_BUFFER}')
            if torch.is_tensor(buf):
                value.data = buf.to(device, non_blocking=False)


def _offload_flats(flats):
    """Rebind ZeRO fp32 master flats at CPU buffers. ``None`` slots were already on CPU."""
    import torch

    saved = []
    moved = False
    for tensor in flats:
        if not torch.is_tensor(tensor) or tensor.numel() == 0 or tensor.device.type == 'cpu':
            saved.append(None)
            continue
        buf = torch.empty_like(tensor, device='cpu')
        buf.copy_(tensor, non_blocking=False)
        tensor.data = buf
        saved.append(buf)
        moved = True
    return saved if moved else None


def _reload_flats(flats, saved, device) -> None:
    import torch

    if device is None:
        return
    for tensor, buf in zip(flats, saved, strict=False):
        if buf is None or not torch.is_tensor(tensor):
            continue
        tensor.data = buf.to(device, non_blocking=False)


def _select_offload(zero, *, offload_model: bool, offload_optimizer: bool):
    """Which ZeRO buffers ``offload_states`` may move, and whether Adam must be moved by hand.

    Stage 3 refuses ``hp_params`` / ``optim_states`` unless the inner optimizer is FusedAdam.
    A JSON ``offload_param`` / ``offload_optimizer`` config already keeps that piece on CPU, so it
    is left alone: calling ``offload_states`` on it would rebind buffers ZeRO still owns.
    """
    from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum as Kind

    include = set()
    manual_optim = False
    params_on_gpu = not getattr(zero, 'offload_param', False)
    optim_on_gpu = not (getattr(zero, 'cpu_offload', False) or getattr(zero, 'offload_optimizer', False))
    if offload_model and params_on_gpu:
        include.update({Kind.lp_params, Kind.lp_grads, Kind.contiguous_grad_buffer})
    if offload_optimizer and optim_on_gpu:
        stage3 = hasattr(zero, 'fp32_partitioned_groups_flat')
        fused = type(getattr(zero, 'optimizer', None)).__name__ == 'FusedAdam'
        if stage3 and not fused:
            manual_optim = True
        else:
            include.update({Kind.hp_params, Kind.optim_states})
    return include, manual_optim


class DeepSpeedStrategy(TransformerStrategyBase):
    """A training strategy backed by HuggingFace accelerate + DeepSpeed.

    Args:
        mixed_precision: Mixed precision type.
        deepspeed_config: DeepSpeed config dict. If ``None``, a minimal
            ZeRO-2 config is used.
        device_mesh: Model device mesh (unused by DeepSpeed but kept for
            interface parity with other strategies).
    """

    def __init__(
        self,
        mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
        deepspeed_config: Optional[Dict[str, Any]] = None,
        device_mesh=None,
    ):
        from accelerate import Accelerator
        from accelerate.utils import InitProcessGroupKwargs

        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision

        config = deepspeed_config or {'zero_optimization': {'stage': 2}}
        self._deepspeed_plugin = self._build_plugin(config)

        self.accelerator = Accelerator(
            deepspeed_plugin=self._deepspeed_plugin,
            mixed_precision=mixed_precision,
            kwargs_handlers=[
                InitProcessGroupKwargs(
                    timeout=timedelta(seconds=int(os.environ.get('TWINKLE_DIST_TIMEOUT_SECONDS', '7200')), ), ),
            ],
        )

    # -- plugin construction ------------------------------------------------

    def fill_auto(self, micro_batch_size: int, gradient_accumulation_steps: int) -> None:
        """Resolve the config's ``"auto"`` placeholders with the caller's real batch geometry.

        HF Trainer fills these via ``HfTrainerDeepSpeedConfig.trainer_config_process`` and
        accelerate's ``prepare`` via the dataloader's ``batch_size``; twinkle runs neither (no
        TrainingArguments, and ``prepare`` gets the model + optimizers, no dataloader), so the
        placeholders would reach DeepSpeed as raw strings and crash its batch assertion
        (``'auto' > 0``) -- in ``deepspeed.zero.Init`` (ZeRO-3) or at ``Accelerator.prepare``.
        Idempotent: a later call with the same geometry is a no-op.
        """
        import torch.distributed as dist

        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        cfg = self._deepspeed_plugin.deepspeed_config
        if not isinstance(cfg, dict):
            cfg = getattr(cfg, 'config', cfg)
        replacements = {
            'train_micro_batch_size_per_gpu': micro_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'train_batch_size': world_size * micro_batch_size * gradient_accumulation_steps,
        }
        for key, value in replacements.items():
            if cfg.get(key) == 'auto':
                cfg[key] = value

    @staticmethod
    def _build_plugin(config: Dict[str, Any]):
        """Build a ``DeepSpeedPlugin`` from a raw config dict.

        Uses ``HfTrainerDeepSpeedConfig`` when available so that ``"auto"``
        values are resolved; falls back to a plain dict otherwise.
        """
        from accelerate.utils import DeepSpeedPlugin

        try:
            from transformers.integrations.deepspeed import HfTrainerDeepSpeedConfig

            hf_ds_config = HfTrainerDeepSpeedConfig(config)
            return DeepSpeedPlugin(hf_ds_config=hf_ds_config)
        except ImportError:
            return DeepSpeedPlugin(hf_ds_config=config)

    # -- state queries ------------------------------------------------------

    def is_zero_stage(self, stage: int) -> bool:
        cfg = self._deepspeed_plugin.deepspeed_config
        if not isinstance(cfg, dict):
            cfg = getattr(cfg, 'config', {})
        return cfg.get('zero_optimization', {}).get('stage', 0) == stage

    @property
    def zero_stage(self) -> int:
        cfg = self._deepspeed_plugin.deepspeed_config
        if not isinstance(cfg, dict):
            cfg = getattr(cfg, 'config', {})
        return cfg.get('zero_optimization', {}).get('stage', 0)

    # -- twinkle Strategy interface -----------------------------------------

    def pretrained_load_context(self):
        """Context manager for loading pretrained weights.

        Under ZeRO-3, returns ``deepspeed.zero.Init`` so that weights are
        partitioned immediately during ``from_pretrained``. For ZeRO-1/2,
        weights fit in full on each rank, so no special context is needed.
        """
        if self.is_zero_stage(3):
            import deepspeed

            cfg = self._deepspeed_plugin.deepspeed_config
            if not isinstance(cfg, dict):
                cfg = getattr(cfg, 'config', cfg)
            return deepspeed.zero.Init(config_dict_or_path=cfg)
        return nullcontext()

    def wrap_model(self, model, *args):
        """Wrap model with DeepSpeed via ``accelerator.prepare``.

        Patch ordering:
        1. ``DeepSpeedLeafModulesPatch`` (permanent) — ZeRO-3 MoE leaf modules
        2. ``DeepSpeedModulesToSavePatch`` (permanent) — PEFT modules_to_save
        3. ``DeepSpeedHookReorderPatch`` (temporary) — wrap ``deepspeed.initialize``
        4. ``DeepSpeedParamWrapperPatch`` (temporary) — PEFT param metadata

        Steps 3-4 are active only during ``accelerator.prepare``; the wrapper
        restores originals on exit.
        """
        if self.is_zero_stage(3):
            self._force_reentrant_gradient_checkpointing(model)
        apply_patch(model, DeepSpeedLeafModulesPatch)
        apply_patch(model, DeepSpeedModulesToSavePatch)

        with apply_context(model, DeepSpeedHookReorderPatch):
            with apply_context(model, DeepSpeedParamWrapperPatch):
                return self.accelerator.prepare(model, *args)

    def _force_reentrant_gradient_checkpointing(self, model) -> None:
        """Under ZeRO-3, switch an enabled gradient checkpointing to ``use_reentrant=True``.

        The non-reentrant variant (transformers' default) validates recomputed saved-tensor
        metadata; deepspeed's zero-stage-3 ``LinearFunctionForZeroStage3`` saves the gathered
        (full-shaped) weight, and the recompute finds the released, partitioned ``[0]``-shaped
        parameter, so the check raises and the backward dies. The reentrant variant carries no
        such hooks, and the deepspeed engine explicitly tracks its backward phases
        (``clear_backward_seen_flag``), making it the supported mode under ZeRO-3. Other kwargs the
        caller enabled are preserved; an already-reentrant model is left alone (idempotent).
        """
        if not getattr(model, 'is_gradient_checkpointing', False):
            return
        module = next((m for m in model.modules() if getattr(m, 'gradient_checkpointing', False)), None)
        keywords = dict(getattr(getattr(module, '_gradient_checkpointing_func', None), 'keywords', None) or {})
        if keywords.get('use_reentrant'):
            return
        keywords['use_reentrant'] = True
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=keywords)
        from twinkle.utils import get_logger

        get_logger().info('ZeRO-3: switched gradient checkpointing to use_reentrant=True '
                          '(non-reentrant recompute sees partitioned parameters).')

    def unwrap_model(self, model):
        return self.accelerator.unwrap_model(model, keep_torch_compile=False)

    def offload_to_cpu(self, model, optimizer=None, *, offload_model: bool = True) -> None:
        """Move this rank's ZeRO parameters and/or Adam state to CPU for colocation.

        ZeRO keeps the bytes in flat buffers, so a live ZeRO optimizer is asked to
        ``offload_states`` for the pieces that are still on GPU. Stage 3 can only move
        optimizer state that way for FusedAdam; any other Adam is rebound onto a CPU buffer,
        which is what swift 4.x does with ``ds_tensor`` and ``optimizer.state``. A JSON
        ``offload_param`` / ``offload_optimizer`` config already lives on CPU and is left
        there. Passing neither piece still releases the caching allocator before return.
        """
        import torch

        from twinkle.model.offload import record_offload, release_caching_allocator
        from twinkle.utils import get_logger

        if not offload_model and optimizer is None:
            record_offload(self, model=False, optimizer=False)
            release_caching_allocator()
            return

        module = self.unwrap_model(model)
        if getattr(self, '_offload_device', None) is None:
            self._offload_device = _resident_device(module)
        zero = _resolve_zero(model, optimizer)
        self._ds_used_offload_states = False
        self._ds_manual_params = False
        self._ds_manual_optim = False
        self._ds_manual_flats = None
        include = set()
        manual = False
        moved_model = False
        moved_optim = False

        if zero is not None:
            include, manual = _select_offload(
                zero, offload_model=offload_model, offload_optimizer=optimizer is not None)
            if include:
                zero.offload_states(include=include, pin_memory=True, non_blocking=False)
                self._ds_used_offload_states = True
                names = {getattr(kind, 'value', str(kind)) for kind in include}
                moved_model = 'lp_params' in names
                moved_optim = bool(names.intersection({'hp_params', 'optim_states'}))
            if manual:
                flats = getattr(zero, 'fp32_partitioned_groups_flat', ())
                self._ds_manual_flats = _offload_flats(flats)
                self._ds_manual_optim = _offload_optimizer_tensors(_state_owner(zero, optimizer))
                moved_optim = self._ds_manual_flats is not None or self._ds_manual_optim
            if not include and not manual:
                if offload_model:
                    self._ds_manual_params = _offload_parameters(module)
                    moved_model = self._ds_manual_params
                if optimizer is not None:
                    self._ds_manual_optim = _offload_optimizer_tensors(_state_owner(zero, optimizer))
                    moved_optim = self._ds_manual_optim
        else:
            if offload_model:
                self._ds_manual_params = _offload_parameters(module)
                moved_model = self._ds_manual_params
            if optimizer is not None:
                self._ds_manual_optim = _offload_optimizer_tensors(optimizer)
                moved_optim = self._ds_manual_optim

        record_offload(self, model=moved_model, optimizer=moved_optim)
        names = sorted(getattr(kind, 'value', str(kind)) for kind in include)
        get_logger().info(
            'DeepSpeed colocate offload model=%s optimizer=%s via=%s manual_optim=%s',
            moved_model, moved_optim, names or 'parameter-storage', manual)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload_to_gpu(self, model, optimizer=None) -> None:
        """Bring back the pieces :meth:`offload_to_cpu` moved. A no-op when nothing moved."""
        import torch

        from twinkle.model.offload import clear_offload_record, offload_record

        moved_model, moved_optim = offload_record(self)
        if not moved_model and not moved_optim:
            return
        device = getattr(self, '_offload_device', None)
        module = self.unwrap_model(model)
        zero = _resolve_zero(model, optimizer)
        if getattr(self, '_ds_used_offload_states', False) and zero is not None:
            zero.reload_states(non_blocking=False)
        self._ds_used_offload_states = False
        saved = getattr(self, '_ds_manual_flats', None)
        if saved and zero is not None:
            _reload_flats(getattr(zero, 'fp32_partitioned_groups_flat', ()), saved, device)
        self._ds_manual_flats = None
        if getattr(self, '_ds_manual_optim', False):
            _reload_optimizer_tensors(_state_owner(zero, optimizer), device)
        self._ds_manual_optim = False
        if getattr(self, '_ds_manual_params', False):
            _reload_parameters(module, device)
        self._ds_manual_params = False
        clear_offload_record(self)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def load_peft_weights(
        self,
        model,
        adapter_weights: Mapping[str, Any],
        adapter_name: str,
    ) -> None:
        from peft.utils import set_peft_model_state_dict

        set_peft_model_state_dict(model, adapter_weights, adapter_name=adapter_name)

    def needs_wrapped_optimizer_state(self) -> bool:
        return True

    def save_optimizer_checkpoint(
        self,
        model,
        optimizer,
        output_path: str,
        *,
        param_name_mapping=None,
    ) -> None:
        """Save optimizer state via DeepSpeed engine checkpoint.

        ``output_path`` is a file path (e.g. ``.../optimizer.pt``); DeepSpeed
        ``save_checkpoint`` takes a directory, so we use the parent dir.
        Both model and optimizer state are saved — the model portion is
        redundant with ``save()`` but is the standard DeepSpeed format.
        """
        # Engine checkpoints round-trip optimizer state under DeepSpeed's internal
        # parameter names, so a caller-side renaming map has no meaning here.
        if param_name_mapping:
            raise ValueError('DeepSpeed engine checkpoints do not support param_name_mapping.')
        engine = _resolve_engine(model)
        save_dir = os.path.dirname(output_path)
        engine.save_checkpoint(save_dir)

    def load_optimizer_checkpoint(
        self,
        model,
        optimizer,
        input_path: str,
        *,
        param_name_mapping=None,
    ) -> None:
        """Load optimizer state from a DeepSpeed engine checkpoint."""
        from twinkle.utils import get_logger

        logger = get_logger()
        if param_name_mapping:
            raise ValueError('DeepSpeed engine checkpoints do not support param_name_mapping.')
        engine = _resolve_engine(model)
        load_dir = os.path.dirname(input_path)
        try:
            engine.load_checkpoint(load_dir)
        except (KeyError, FileNotFoundError, RuntimeError) as e:
            logger.warning(f'Failed to load deepspeed checkpoint: {e}')

    def get_full_state_dict(self, model) -> dict:
        """Collect full state dict (with persistent buffers), gathering ZeRO-3 partitioned params."""
        unwrapped = self.unwrap_model(model)
        state = collect_local_state(unwrapped, include_buffers=True)
        if self.is_zero_stage(3):
            params = [t for t in state.values() if isinstance(t, nn.Parameter)]
            self._gather_zero3_params_for_snapshot(model, params)
        return {name: tensor.cpu() for name, tensor in state.items()}

    def load_full_state_dict(self, model, state_dict) -> None:
        """Load a full state dict into the (possibly partitioned) model."""
        unwrapped = self.unwrap_model(model)
        if self.is_zero_stage(3):
            from deepspeed.runtime.zero import GatheredParameters

            params = list(unwrapped.parameters())
            with GatheredParameters(params):
                unwrapped.load_state_dict(state_dict, strict=False)
        else:
            unwrapped.load_state_dict(state_dict, strict=False)

    def _gather_zero3_params_for_snapshot(self, model, params) -> None:
        """Drain in-flight gathers, then collect ZeRO-3 params for a read-only snapshot.

        ``GatheredParameters``' ``__exit__`` partitions everything it gathered, which breaks the
        adapter/full save in two observed ways: a persist-flagged parameter (under the stage-3
        persistence threshold, e.g. every LoRA matrix) is still referenced by live submodules, so
        freeing it raises ``Cannot free a ZeRO-3 parameter while it is still active in
        submodules``; and parameters an async pre-forward gather left in-flight (the recompute
        leg of reentrant checkpointing left 51 of them after the last backward) trip the
        partition assert. The official drain is the parameter offloader's
        ``partition_all_parameters`` -- its ``release_and_reset_all`` waits every in-flight
        handle, clears the active-module references and releases all parameters to the
        partitioned state. A plain synchronous ``all_gather`` then brings the snapshot's own
        parameters back gathered, and they stay gathered: a snapshot only reads ``.cpu()``
        copies, and deep speed re-partitions them on the next forward/backward cycle.
        """
        engine = _resolve_engine(model)
        optimizer = getattr(engine, 'optimizer', None)
        parameter_offload = getattr(optimizer, 'parameter_offload', None)
        if parameter_offload is None:
            # An engine shape we have not met: fall back to a synchronous gather with no
            # re-partition (the old behavior) rather than guessing at internals.
            zero_params = [p for p in params if hasattr(p, 'ds_id')]
            if zero_params:
                zero_params[0].all_gather(param_list=zero_params)
            return
        parameter_offload.partition_all_parameters()
        zero_params = [p for p in params if hasattr(p, 'ds_id')]
        if zero_params:
            zero_params[0].all_gather(param_list=zero_params)

    def get_adapter_state_dict(self, model, adapter_name: str) -> dict:
        """Collect LoRA parameters, trained modules_to_save copies and their buffers, gathering under ZeRO-3."""
        unwrapped = self.unwrap_model(model)
        adapter_suffix = f'.{adapter_name}.'
        state = collect_local_state(unwrapped, include_buffers=True)
        selected = {n: t for n, t in state.items() if is_saveable_adapter_key(n) and adapter_suffix in n}

        if self.is_zero_stage(3):
            # ZeRO-3 partitions parameters only; buffers are plain tensors and pass through.
            params = [t for t in selected.values() if isinstance(t, nn.Parameter)]
            self._gather_zero3_params_for_snapshot(model, params)
        return {n: t.cpu() for n, t in selected.items()}

    # -- DeepSpeed-specific -------------------------------------------------

    def gather_parameters(self, params):
        """Context manager to gather ZeRO-3 partitioned parameters.

        Usage::

            with strategy.gather_parameters(list(model.parameters())):
                # params are fully gathered here
                ...
        """
        if self.is_zero_stage(3):
            from deepspeed.runtime.zero import GatheredParameters

            return GatheredParameters(params)
        return nullcontext()
