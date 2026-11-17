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

    def optimizer_checkpoint_exists(self, output_dir: str) -> bool:
        # ``engine.save_checkpoint`` records the tag it wrote in ``latest`` and puts the state under
        # that subdirectory; both halves must be there for ``engine.load_checkpoint`` to find it.
        latest = os.path.join(output_dir, 'latest')
        if not os.path.exists(latest):
            return False
        with open(latest) as handle:
            tag = handle.read().strip()
        return bool(tag) and os.path.isdir(os.path.join(output_dir, tag))

    def owns_update_step(self) -> bool:
        return True

    def engine_counters(self, model) -> Dict[str, int]:
        """The engine's own update and micro-batch counters, which decide the accumulation boundary."""
        engine = _resolve_engine(model)
        return {name: int(getattr(engine, name)) for name in ('global_steps', 'micro_steps')}

    #: Where each offload tier keeps the float32 master it actually updates: ZeRO-2 one flat partition
    #: per group, ZeRO-3 the live partitioned copies, and the saved-state form read when restoring.
    _FP32_PARTITION_ATTRS = ('single_partition_of_fp32_groups', 'fp32_partitioned_groups_flat', 'fp32_flat_groups')

    @classmethod
    def _fp32_partitions(cls, model):
        """The attribute name and the list of float32 master partitions this tier updates."""
        optimizer = getattr(_resolve_engine(model), 'optimizer', None)
        for attr in cls._FP32_PARTITION_ATTRS:
            candidate = getattr(optimizer, attr, None)
            if candidate:
                return attr, candidate
        return None, None

    def engine_master_digest(self, model, *, stepped: bool = False, step: Optional[int] = None) -> Dict[str, Any]:
        """Per-group float statistics over a sample, plus an exact fingerprint of every element.

        ``agg`` is ``[sum, absmax, norm]`` over a strided sample and is what localizes and sizes a
        difference; ``bits`` is exact over the whole partition and is what decides whether two runs are
        identical at all. ``TWINKLE_STEP_DELTA_PROBE_FULL_STEP`` additionally names one global step at
        which the float statistics are computed over every element, for when a difference has to be
        sized and located rather than merely detected.

        ``stepped`` says whether ``engine.step`` has already run for the step being digested, which is
        what selects the counter reading the full-coverage gate compares against.
        """
        _, partitions = self._fp32_partitions(model)
        if not partitions:
            return {}
        full_step = os.environ.get('TWINKLE_STEP_DELTA_PROBE_FULL_STEP')
        # ``global_steps`` counts completed updates, so before ``engine.step`` the update about to run
        # is ``global_steps + 1``, and after it the one just run is ``global_steps`` itself. Applying
        # the pre-step reading at a post-step call site misses the target step by one and silently
        # drops that digest back to the 64-chunk sample -- measured at FULL_STEP=16 on a ZeRO-2 offload
        # run: ``master_pre[0]`` came back ``covered=True`` with 1108097 chunks while ``master_post[0]``
        # came back ``covered=False`` with 64, i.e. 0.37% of a 4,538,761,984-element partition, which is
        # exactly the side a difference has to be sized and located on.
        engine_step = _resolve_engine(model).global_steps + (0 if stepped else 1)
        at_full_step = bool(full_step) and engine_step == int(full_step)
        out = {}
        for index, group in enumerate(partitions):
            if group is None:
                continue
            flat = group.detach().flatten()
            chunks = self._full_coverage_chunks(flat.numel()) if at_full_step else 64
            stats = self._chunk_stats(flat, chunks=chunks)
            entry = {
                'agg': self._aggregate(stats),
                'chunks': chunks,
                'covered': self._coverage_complete(flat.numel(), chunks) if at_full_step else False,
                'bits': self._bits_fingerprint(flat),
                'numel': int(flat.numel()),
                # The master partition is the buffer ``adam_update`` writes, so it is the one surface
                # whose identity was still unmeasured after the moments and the gradient partitions were
                # both shown to keep one storage across the update. Reading it here means a divergence
                # that survives identical inputs and identical moment output can still be traced to the
                # destination buffer rather than left as an unexplained residue.
                **self._buffer_identity(group),
            }
            if at_full_step:
                # ``agg`` sizes a difference and ``bits`` decides whether there is one, but neither says
                # where: on a 4,538,761,984-element group the element budget inside ``_full_coverage_chunks``
                # makes even the full-step read partial, so a difference confined to that group could not be
                # located. Keeping the per-chunk sums for the step under investigation turns a differing
                # chunk range into parameter names through the layout digest, which is what separates a
                # per-parameter branch from a difference spread evenly over the whole group. Only at that
                # step: the list is as long as the chunk count, ~16k here and ~1.1M without the budget.
                entry['chunk_sums'] = [s[0] for s in stats]
            out[str(index)] = entry
        return out

    @staticmethod
    def _buffer_identity(tensor) -> Dict[str, Any]:
        """Where a buffer physically is, beside the fingerprint of what it contains.

        ``bits`` proves two readings have identical *contents*; it cannot tell whether the C++
        ``adam_update`` consumed the same storage the Python-side ``state`` dict points at. Two tensors
        holding equal values at different addresses are indistinguishable by any content fingerprint, so
        the addresses are recorded beside it. ``storage_ptr`` differs from ``ptr`` whenever the tensor is
        a view -- equal ``storage_ptr`` with different ``ptr`` is the signature of two views into one
        buffer, which is the aliased-buffer shape of the divergence. ``pinned`` separates the
        pinned-host-buffer mechanism from the plain-host one without needing a second run.
        """
        identity = {
            'ptr': int(tensor.data_ptr()),
            'storage_ptr': int(tensor.untyped_storage().data_ptr()),
            'device': str(tensor.device),
            'dtype': str(tensor.dtype),
        }
        if tensor.device.type == 'cpu':
            identity['pinned'] = bool(tensor.is_pinned())
        return identity

    def engine_optim_moment_digest(self, model, optimizer=None, *, step: Optional[int] = None) -> Dict[str, Any]:
        """Per-group ``[sum, absmax, norm]`` of the Adam moments the next update will read.

        ``optimizer`` and ``step`` are the loop-side handles a strategy needs when the loop owns the
        update; ZeRO owns it, so both are unused here and the moments come from the engine's own
        inner optimizer, resolved from ``model``.

        The last unmeasured input to the update. On a ZeRO-2 offload resume, the fp32 master, the
        gradient the update consumes, the lr, betas, eps, weight decay and the Adam step counter all
        read identically to the fresh run at the first resumed step, and the post-update master still
        differs -- so either the moments differ or the update itself does, and nothing else is left to
        distinguish those. They live in the *inner* optimizer's state, keyed by the fp32 partition
        tensor the ZeRO optimizer updates rather than by any model parameter, which on this tier the
        inner optimizer never sees.
        """
        import torch

        chunks = int(os.environ.get('TWINKLE_STEP_DELTA_PROBE_CHUNKS', 64))
        _, partitions = self._fp32_partitions(model)
        if not partitions:
            return {}
        inner = getattr(_resolve_engine(model).optimizer, 'optimizer', None)
        state = getattr(inner, 'state', None) or {}
        out: Dict[str, Any] = {'sources': [], 'chunks': chunks}
        for index, partition in enumerate(partitions):
            if partition is None:
                continue
            # Tensors hash by identity, and the partition tensor is itself the inner optimizer's
            # "parameter", so this is the same lookup the update performs.
            entry = state.get(partition)
            if not isinstance(entry, dict):
                continue
            step = entry.get('step')
            out[f'adam_step[{index}]'] = int(step.item()) if hasattr(step, 'item') else (
                int(step) if step is not None else None)
            # The value is not the whole input. ``ops/adam/cpu_adam.py`` hands ``state['step']``
            # directly to the C++ ``adam_update``, so a restore that puts a tensor where a fresh run
            # holds an int reaches the binding as a different type while ``int(...)`` above reports
            # the same number for both. The type is recorded beside the value to keep that visible.
            out[f'adam_step_type[{index}]'] = type(step).__name__
            for moment in ('exp_avg', 'exp_avg_sq'):
                value = entry.get(moment)
                if torch.is_tensor(value):
                    name = f'{moment}[{index}]'
                    flat = value.detach().flatten()
                    stats = self._chunk_stats(flat, chunks=chunks)
                    out[name] = {
                        'agg': self._aggregate(stats),
                        'chunks': [s[0] for s in stats],
                        # Exact over every element, so an identical fingerprint means the restored moments
                        # are bit-identical rather than merely agreeing on a 0.37% sample.
                        'bits': self._bits_fingerprint(flat),
                        'numel': int(value.numel()),
                        **self._buffer_identity(value),
                    }
                    out['sources'].append(name)
        if not out['sources']:
            out['unreadable'] = ['<inner optimizer>.state[fp32_partition]']
        return out

    def engine_grad_partition_digest(self, model) -> Dict[str, Any]:
        """``[sum, absmax, norm]`` of the float32 gradients the offload optimizer updates from.

        ZeRO-2 keeps the reduced gradients in ``averaged_gradients`` and hands the CPU optimizer the
        ``.grad`` of its own float32 partition; ZeRO-3 exposes neither under those names. Each tier is
        read through whatever it has, and the names actually found are reported in ``sources``, so a
        tier carrying no readable gradient shows up as such instead of reading as "gradients matched".
        """
        import torch

        # An offload tier fills this partition with a ``non_blocking`` device-to-host copy, so reading it
        # before that copy lands measures a mixture of two steps rather than one gradient. Measured, not
        # hypothetical: inside a single digest of a single arm, the strided ``agg`` and the all-element
        # ``bits`` -- taken one after the other from the same tensor -- disagreed with each other across a
        # fresh/resume pair (``bits`` equal while ``agg`` differed), which is only possible if the bytes
        # changed between the two reads. Synchronizing first lands every copy already issued, so one digest
        # describes one instant. Same guard, same reason as ``_digest_parameters``. This method only runs
        # when the step-delta probe is on, so training that is not being measured pays nothing.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # The default stride samples 2.6% of a multi-billion element partition. That is enough to say a
        # difference exists and roughly where, but not enough to say it is confined there: a post-update
        # parameter digest can show changes in parameters whose gradient chunk was never sampled. The
        # count is tunable so a single investigation can pay for a denser read without every run doing so.
        chunks = int(os.environ.get('TWINKLE_STEP_DELTA_PROBE_CHUNKS', 64))
        out: Dict[str, Any] = {'sources': [], 'chunks': chunks}
        attr, partitions = self._fp32_partitions(model)
        if partitions:
            for index, group in enumerate(partitions):
                grad = getattr(group, 'grad', None) if group is not None else None
                if torch.is_tensor(grad):
                    name = f'{attr}[{index}].grad'
                    flat = grad.detach().flatten()
                    stats = self._chunk_stats(flat, chunks=chunks)
                    out[name] = {
                        'agg': self._aggregate(stats),
                        'chunks': [s[0] for s in stats],
                        'bits': self._bits_fingerprint(flat),
                        'numel': int(grad.numel()),
                        # The pinned-host-buffer and async-copy mechanisms are properties of WHERE this
                        # gradient lives, not of what it contains, so no content fingerprint can separate
                        # them; the address and the pinned flag can, from the same run.
                        **self._buffer_identity(grad),
                    }
                    out['sources'].append(name)
        averaged = getattr(getattr(_resolve_engine(model), 'optimizer', None), 'averaged_gradients', None)
        if isinstance(averaged, dict):
            for index in sorted(averaged):
                for slot, grad in enumerate(averaged[index] or []):
                    if torch.is_tensor(grad):
                        name = f'averaged_gradients[{index}][{slot}]'
                        out[name] = self._aggregate(self._chunk_stats(grad.detach().flatten(), chunks=chunks))
                        out['sources'].append(name)
        if not out['sources']:
            # Name the failure instead of returning a bare empty digest. Two empty digests compare equal,
            # so a tier whose gradient is not readable from here would otherwise sign off as "gradients
            # matched". ZeRO-2 without offload reaches this after ``step`` by design: it assigns the flat
            # gradient to the fp32 partition inside ``step`` and clears both it and ``averaged_gradients``
            # before returning, so the gradient that update consumed is only readable beforehand.
            out['unreadable'] = list(self._FP32_PARTITION_ATTRS) + ['averaged_gradients']
        return out

    def engine_grad_partition_layout(self, model) -> Dict[str, Any]:
        """Where each parameter sits inside the flat fp32 gradient partition, per group.

        A digest can only say that a range of the flat partition differs; turning that range into
        parameter names is what points at a code path, since the flat layout follows the order the
        parameters were registered. Gated by its own env flag because the layout is a few hundred
        entries per group and is identical at every step -- it is read once, not tracked per step.
        """
        if not os.environ.get('TWINKLE_STEP_DELTA_PROBE_LAYOUT'):
            return {}
        optimizer = getattr(_resolve_engine(model), 'optimizer', None)
        names = {id(param): name for name, param in model.named_parameters()}
        out: Dict[str, Any] = {}
        for index, params in enumerate(getattr(optimizer, 'params_in_partition', None) or []):
            entries = []
            offset = 0
            for param in params:
                numel = param.numel()
                entries.append([names.get(id(param), '?'), offset, numel])
                offset += numel
            out[str(index)] = {'total': offset, 'params': entries}
        return out

    def take_engine_step_probe(self) -> Dict[str, Any]:
        """The digests ``engine_backward`` recorded around ``engine.step``, consumed once."""
        return self.__dict__.pop('_engine_step_probe', {})

    def engine_backward(self, model, loss, *, sync_gradients: bool) -> None:
        """Drive the engine for one micro-batch: backward always, the update at the boundary.

        This is the split accelerate's own ``DeepSpeedEngineWrapper.backward`` performs, mapped onto
        the loop's explicit backward-then-step sequence. ``set_gradient_accumulation_boundary`` tells
        the engine whether this micro-batch closes an accumulation window, which is what makes
        ``engine.step`` apply the update (and scale the loss by the window size) only there; the
        engine's ``step`` then also does the clipping from ``gradient_clipping`` in the DeepSpeed
        config and the zero_grad, so the loop's own ``step`` and ``zero_grad`` -- both no-ops on
        accelerate's ``DeepSpeedOptimizerWrapper`` -- leave the update exactly once.
        """
        engine = _resolve_engine(model)
        engine.set_gradient_accumulation_boundary(is_boundary=sync_gradients)
        engine.backward(loss)
        if sync_gradients:
            # The step probe brackets the update where it actually happens. The loop's
            # ``clip_grad_and_step`` runs after this, so a digest taken there sees an already-updated
            # master on both sides and would report "no change" over a step that did change it.
            probe = os.environ.get('TWINKLE_STEP_DELTA_PROBE')
            if probe:
                self.__dict__['_engine_step_probe'] = {
                    'grad_partition_pre': self.engine_grad_partition_digest(model),
                    'layout': self.engine_grad_partition_layout(model),
                    'master_pre': self.engine_master_digest(model),
                    # Read before the update, since the update overwrites the moments in place: this is
                    # the value the update is about to consume, which is the one that has to match.
                    'optim_moment_pre': self.engine_optim_moment_digest(model),
                }
            engine.step()
            if probe:
                step_probe = self.__dict__['_engine_step_probe']
                step_probe['master_post'] = self.engine_master_digest(model, stepped=True)
                # ZeRO-2 assigns this step's reduced gradient to the float32 partition's ``.grad``
                # inside ``step`` and the CPU optimizer consumes it there, so the partition read BEFORE
                # the step is the previous step's gradient -- on a first step after a restore it is a
                # cold buffer that means something else entirely. The gradient the update actually used
                # is only readable afterwards.
                step_probe['grad_partition_post'] = self.engine_grad_partition_digest(model)
                # The moments after the update. Without this the only readable moment state on a tier
                # whose engine owns the step is the pre-update one, so "the update consumed a buffer
                # other than the one the state dict points at" stays indistinguishable from "the update
                # is not reproducible": both leave ``optim_moment_pre`` equal across the two arms. Equal
                # pre with unequal post is the former, and unequal ``ptr``/``storage_ptr`` between the
                # two readings names the buffer that was written instead.
                step_probe['optim_moment_post'] = self.engine_optim_moment_digest(model)

    def engine_grad_norm(self, model):
        engine = _resolve_engine(model)
        norm = engine.get_global_grad_norm()
        # The engine keeps the norm as a tensor on the model's device, while the metric path mixes
        # it with values gathered from the other ranks -- whose tensors stay on THEIR device -- so a
        # tensor here dies with "Expected all tensors to be on the same device". Hand back a number.
        if norm is None:
            return None
        return float(norm) if hasattr(norm, 'item') else norm

    def unwrap_model(self, model):        return self.accelerator.unwrap_model(model, keep_torch_compile=False)

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

    def _probe_fp32_master(self, engine, load_dir: str) -> None:
        """Compare the live fp32 master weights against the ones in the checkpoint.

        Gated by ``TWINKLE_OPT_MASTER_PROBE`` (a path prefix). With ZeRO offload the model's
        parameters are bfloat16 and the precision the optimizer actually updates lives in a flat
        float32 partition held by the CPU optimizer, so a checkpoint restore that brings back the
        moments but not that partition leaves the run updating a master rebuilt from bf16 weights.
        Restoring the moments alone looks like success: the first resumed step reproduces the source
        run's loss, grad_norm and gradients exactly, because the forward only ever sees the bf16
        weights, and the drift appears one step later.

        Every element of each partition is compared, in chunks. These files are tens of gigabytes, so
        the read goes through mmap and the comparison streams rather than materialising the artifact.
        """
        prefix = os.environ.get('TWINKLE_OPT_MASTER_PROBE')
        if not prefix:
            return
        import glob
        import json

        import torch

        rank = int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0)
        optimizer = getattr(engine, 'optimizer', None)
        live = None
        live_attr = None
        for attr in ('single_partition_of_fp32_groups', 'fp32_partitioned_groups_flat', 'fp32_flat_groups'):
            candidate = getattr(optimizer, attr, None)
            if candidate:
                live, live_attr = candidate, attr
                break
        paths = (sorted(glob.glob(os.path.join(load_dir, 'global_step*', f'*rank_{rank}_*optim_states.pt')))
                 or sorted(glob.glob(os.path.join(load_dir, f'*rank_{rank}_*optim_states.pt'))))
        record = {
            'rank': rank,
            'live_attr': live_attr,
            'live_groups': len(live) if live is not None else 0,
            'artifact': os.path.basename(paths[0]) if paths else None,
        }
        if live is None or not paths:
            record['error'] = 'no live master or no optim-states artifact found'
            with open(f'{prefix}.rank{rank}.jsonl', 'a') as handle:
                handle.write(json.dumps(record) + '\n')
            return
        try:
            saved_obj = torch.load(paths[0], map_location='cpu', mmap=True, weights_only=True)
        except Exception:
            # DeepSpeed's optim-states artifact can carry non-tensor members (a loss-scaler object,
            # the ds_config dict) that weights-only loading refuses. This file was written by this
            # process's own save into this run's own output directory, so it is not untrusted input.
            saved_obj = torch.load(paths[0], map_location='cpu', mmap=True, weights_only=False)
        saved_osd = saved_obj.get('optimizer_state_dict', saved_obj)
        saved = None
        for key in ('single_partition_of_fp32_groups', 'fp32_flat_groups'):
            if saved_osd.get(key):
                saved = saved_osd[key]
                record['saved_key'] = key
                break
        detail = []
        base = getattr(optimizer, 'optimizer', None)
        live_state = getattr(base, 'state', None) or {}
        # ZeRO-2 offload nests the moments under ``base_optimizer_state``, ZeRO-3 under
        # ``optimizer_state_dict``; both then hold a torch-style ``{'state': {group: {...}},
        # 'param_groups': ...}`` whose per-group leaf carries step / exp_avg / exp_avg_sq.
        saved_container = None
        saved_container_key = None
        for key in ('base_optimizer_state', 'optimizer_state_dict'):
            if isinstance(saved_osd.get(key), dict):
                saved_container, saved_container_key = saved_osd[key], key
                break
        saved_state = saved_container.get('state') if saved_container is not None else None
        record['saved_moment_container'] = saved_container_key
        record['saved_moment_groups'] = len(saved_state) if isinstance(saved_state, dict) else None
        for index, live_group in enumerate(live):
            if live_group is None or saved is None or index >= len(saved) or saved[index] is None:
                detail.append({'group': index, 'skipped': True})
                continue
            saved_group = saved[index]
            entry = {'group': index}
            entry.update(self._compare_flat(live_group, saved_group, check_bf16=True))
            entry.update(self._compare_moments(live_state, live_group, saved_state, index))
            detail.append(entry)
        record['detail'] = detail
        with open(f'{prefix}.rank{rank}.jsonl', 'a') as handle:
            handle.write(json.dumps(record) + '\n')

    @staticmethod
    def _compare_flat(live_tensor, saved_tensor, *, chunk: int = 1 << 24, check_bf16: bool = False) -> dict:
        """Compare two flat partitions element by element, in full, one chunk at a time.

        A bounded window left most of a multi-gigabyte partition unexamined, so a restore that was
        wrong past the first million elements still reported ``exact``. Chunking covers every element
        while keeping peak memory at a chunk: the saved side is mmap-backed, so a slice reads only what
        the comparison touches, and the live side is never copied whole.
        """
        import torch

        live = live_tensor.detach().flatten()
        saved = saved_tensor.detach().flatten()
        total = min(live.numel(), saved.numel())
        differing = 0
        worst = 0.0
        bf16_of_saved = True
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            live_values = live[start:stop].to(torch.float32)
            saved_values = saved[start:stop].to(torch.float32)
            differing += int((live_values != saved_values).sum())
            if differing:
                worst = max(worst, float((live_values - saved_values).abs().max()))
            if check_bf16 and bf16_of_saved:
                bf16_of_saved = bool((live_values == saved_values.to(torch.bfloat16).to(torch.float32)).all())
        out = {
            'numel': int(live_tensor.numel()),
            'compared': int(total),
            'differing': differing,
            'exact': differing == 0,
            'worst_abs_diff': worst,
        }
        if check_bf16:
            # True means the master holds bf16-rounded values, whether because the partition was never
            # restored and was rebuilt from the bf16 parameters, or because it was restored through a
            # narrowing copy. Either way the fp32 precision is gone.
            out['live_is_bf16_of_saved'] = bf16_of_saved
        return out

    @staticmethod
    def _compare_moments(live_state, live_group, saved_state, index) -> dict:
        """Compare the restored Adam moments against the checkpoint's, over their whole partition.

        A faithful master with unfaithful moments is indistinguishable from a faithful everything on
        the first resumed step: that step's loss, grad_norm and clipped gradients all reproduce the
        source run exactly, because they depend only on the weights, and the wrong moments change the
        update that follows. ZeRO-2 offload stores them under ``base_optimizer_state`` keyed by group
        then by position within the group; the live ones hang off the inner CPU optimizer, keyed by
        the fp32 partition tensor itself.
        """
        import torch

        out = {}
        live_entry = None
        try:
            live_entry = live_state.get(live_group) if hasattr(live_state, 'get') else None
        except TypeError:
            live_entry = None
        saved_entry = saved_state.get(index) if isinstance(saved_state, dict) else None
        for key in ('exp_avg', 'exp_avg_sq'):
            live_tensor = live_entry.get(key) if isinstance(live_entry, dict) else None
            saved_tensor = saved_entry.get(key) if isinstance(saved_entry, dict) else None
            if not torch.is_tensor(live_tensor) or not torch.is_tensor(saved_tensor):
                out[key] = {
                    'present': (torch.is_tensor(live_tensor), torch.is_tensor(saved_tensor)),
                }
                continue
            out[key] = DeepSpeedStrategy._compare_flat(live_tensor, saved_tensor)
        live_step = live_entry.get('step') if isinstance(live_entry, dict) else None
        saved_step = saved_entry.get('step') if isinstance(saved_entry, dict) else None
        out['step'] = {
            'live': int(live_step.item()) if torch.is_tensor(live_step) else live_step,
            'saved': int(saved_step.item()) if torch.is_tensor(saved_step) else saved_step,
        }
        return out

    @staticmethod
    def _has_optimizer_state(load_dir: str) -> bool:
        """Whether the checkpoint carries a DeepSpeed optimizer-state artifact for this rank.

        Distinguishes "this checkpoint has no optimizer state to restore" (a module-only resume, or a
        checkpoint written by a different world size) from "it has one and loading it failed". Only the
        first is a legitimate reason to keep going.
        """
        import glob

        rank = int(os.environ.get('RANK') or os.environ.get('LOCAL_RANK') or 0)
        return any(
            glob.glob(pattern) for pattern in (
                os.path.join(load_dir, 'global_step*', f'*rank_{rank}_*optim_states.pt'),
                os.path.join(load_dir, f'*rank_{rank}_*optim_states.pt'),
            ))

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
            # A resume whose optimizer state did not load keeps training and looks healthy: the forward
            # only reads the model weights, so the first resumed step reproduces the source run's loss
            # and grad_norm exactly and the divergence appears one update later, with nothing in the log
            # to explain it. Swallowing that into a warning turns a stopped run into a silently wrong one.
            if self._has_optimizer_state(load_dir):
                raise RuntimeError(
                    f'Failed to load the DeepSpeed optimizer state from {load_dir}: {e}. Training on an '
                    'unrestored optimizer diverges from the source run one step after the resume.') from e
            logger.warning(f'{load_dir} carries no optimizer state for this rank; resumed the model only: {e}')
        self._probe_fp32_master(engine, load_dir)

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
