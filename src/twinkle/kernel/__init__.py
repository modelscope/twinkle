# Copyright (c) ModelScope Contributors. All rights reserved.
"""Twinkle Kernel Module - Kernel orchestration layer."""
import torch
from logging import getLogger
from typing import Any, Dict, Optional, Union

from twinkle.utils.framework import Torch
from .base import DeviceType, ModeType, is_kernels_enabled
from .function import apply_function_kernel, register_function_kernel
from .layer import apply_layer_kernel, register_layer_batch, register_layer_kernel
from .monkey_patch_npu import apply_npu_patch, register_npu_fused_function_kernels
from .registry import register_external_layer as _register_external_layer

logger = getLogger(__name__)

__all__ = [
    'kernelize_model',
    'register_layer_kernel',
    'register_function_kernel',
    'register_external_layer',
    'register_kernels',
    'apply_npu_patch',
    'apply_npu_fused_ops',
]


def kernelize_model(
    model,
    mode: ModeType = 'inference',
    device: Optional[DeviceType] = None,
    use_fallback: bool = True,
) -> Any:
    """Apply kernels to model (main entry point).

    For NPU devices, this also applies Ascend fused operators (RMSNorm, RoPE,
    SwiGLU, SDPA Attention) unconditionally when running on NPU.

    Args:
        model: The PyTorch model to kernelize.
        mode: The mode for kernel selection ("inference" or "train").
        device: The device type (auto-detected if None).
        use_fallback: Whether to use original forward when no compatible kernel found.
            If False, raises ValueError when kernel is unavailable.

    Returns:
        The kernelized model.
    """
    # Step 0: NPU monkey-patches must be applied BEFORE layer kernel replacement
    # so that patched module classes are used when new instances are created.
    if device == 'npu' or (device is None and _is_npu_device(model)):
        try:
            apply_npu_patch()
        except Exception:
            logger.warning('NPU patch failed. Continuing without fused ops.', exc_info=True)

    model = apply_layer_kernel(model, mode=mode, device=device, use_fallback=use_fallback)

    apply_function_kernel(device=device, mode=mode)

    return model


def apply_npu_fused_ops(config) -> None:
    """Apply NPU fused operators patch manually.
    """
    logger.warning('apply_npu_fused_ops(config) is deprecated. '
                   'Use apply_npu_patch() instead, which enables all patches unconditionally.')
    apply_npu_patch()


def register_external_layer(layer_class: type, kernel_name: str) -> None:
    _register_external_layer(layer_class, kernel_name)


def register_kernels(config: Dict[str, Dict[str, Any]]) -> None:
    """Batch register kernels (framework integration API)."""
    if 'layers' in config:
        for kernel_name, spec in config['layers'].items():
            device = spec.pop('device', 'cuda')
            register_layer_kernel(kernel_name=kernel_name, device=device, **spec)

    if 'functions' in config:
        from .function import register_function_batch

        functions = config['functions']
        if isinstance(functions, dict):
            function_specs = []
            for func_name, spec in functions.items():
                if not isinstance(spec, dict):
                    raise TypeError(f'Function spec for {func_name} must be a dict.')
                if 'func_name' not in spec:
                    spec['func_name'] = func_name
                function_specs.append(spec)
            register_function_batch(function_specs)
        else:
            register_function_batch(functions)


def _is_npu_device(model=None) -> bool:
    """Check if the model (or current environment) is on NPU device."""
    # Priority 1: Check model's actual device (kernel-specific inference)
    if model is not None:
        try:
            param_device = next(model.parameters()).device
            if param_device.type == 'npu':
                return True
        except StopIteration:
            pass

    # Priority 2: Fallback to global NPU availability
    return Torch.is_npu_available()
