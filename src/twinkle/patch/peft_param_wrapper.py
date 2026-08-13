# Copyright (c) ModelScope Contributors. All rights reserved.
"""Make peft ``ParamWrapper.get_param`` correct + cheap under DeepSpeed ZeRO-3.

``ParamWrapper`` is peft's wrapper for applying LoRA to a raw ``nn.Parameter``
(``target_parameters``), e.g. fused MoE expert weights. Its ``get_param`` returns
the targeted base parameter; peft callers read only *metadata* from it:

  * ``_get_in_out_features`` -> ``param.ndim`` / ``param.shape`` (derives
    ``num_experts`` / ``in_features`` / ``out_features`` to build the LoRA layer);
  * ``_move_adapter_to_device_of_base_layer`` / ``get_delta_weight`` -> ``.device`` / ``.dtype``;
  * ``_activate_lora`` -> ``.requires_grad``.

Under DeepSpeed ZeRO-3 a partitioned parameter can be ``NOT_AVAILABLE``: its
``.data`` is a placeholder tensor with the *wrong* shape/ndim (``.device`` /
``.dtype`` / ``.requires_grad`` stay correct). The shape-reading caller
(``_get_in_out_features``) then builds a LoRA layer with wrong dimensions — a
correctness bug, not just a memory one. Gathering the full parameter just to read
its shape would cost O(N) memory per expert block.

This patch rewrites ``get_param`` so that, when the parameter is ``NOT_AVAILABLE``,
it returns a stride-0 tensor built from ``ds_shape`` (the true full shape) via
``torch.empty(1, ...).expand(ds_shape)``: correct metadata (shape / ndim / dtype /
device / requires_grad) at O(1) memory, no gather. Mirrors legacy swift's
``_patch_param_wrapper`` (swift/tuners/peft.py).

Double-gated so it is a no-op outside the exact ZeRO-3 + ParamWrapper combo:
  1. if peft has no ``ParamWrapper`` (older peft, or ``target_parameters`` unused)
     -> nothing to patch;
  2. per call, only the ``ds_status == NOT_AVAILABLE`` branch is rewritten;
     regular tensors (no DeepSpeed, or gathered params) fall through unchanged.

Wire this into the DeepSpeed ZeRO-3 strategy setup when MoE-expert LoRA
(``target_parameters``) is in use::

    from twinkle.patch import apply_patch
    from twinkle.patch.peft_param_wrapper import PeftParamWrapperZero3Patch
    apply_patch(None, PeftParamWrapperZero3Patch())
"""
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_twinkle_origin_get_param'


def _param_wrapper_cls():
    """Return peft's ``ParamWrapper`` class, or ``None`` when it is unavailable
    (older peft, or ``target_parameters`` never used)."""
    try:
        from peft.tuners.lora.layer import ParamWrapper
        return ParamWrapper
    except ImportError:
        return None


class PeftParamWrapperZero3Patch(Patch):
    """Return correct O(1) metadata from ``ParamWrapper.get_param`` for ZeRO-3
    ``NOT_AVAILABLE`` params. Idempotent, reversible, no-op without ParamWrapper."""

    def __call__(self, module=None, *args, **kwargs):
        ParamWrapper = _param_wrapper_cls()
        if ParamWrapper is None or hasattr(ParamWrapper, _MARKER):
            return module

        origin_get_param = ParamWrapper.get_param

        def get_param(self):
            import torch
            param = origin_get_param(self)
            if hasattr(param, 'ds_id'):
                from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
                if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                    # ds_shape is always set by DeepSpeed for managed params.
                    # Build a 1-element tensor then expand with stride-0: correct
                    # shape/ndim/dtype/device at O(1) memory, no gather.
                    ds_shape = param.ds_shape
                    fake = torch.empty((1, ) * len(ds_shape), dtype=param.dtype, device=param.device)
                    if param.requires_grad and param.dtype.is_floating_point:
                        fake.requires_grad_(True)
                    return fake.expand(ds_shape)
            return param

        setattr(ParamWrapper, _MARKER, origin_get_param)
        ParamWrapper.get_param = get_param
        logger.info('Patched peft ParamWrapper.get_param for DeepSpeed ZeRO-3 metadata correctness.')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        ParamWrapper = _param_wrapper_cls()
        if ParamWrapper is None:
            return module
        origin = getattr(ParamWrapper, _MARKER, None)
        if origin is not None:
            ParamWrapper.get_param = origin
            delattr(ParamWrapper, _MARKER)
        return module
