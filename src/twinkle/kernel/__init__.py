# Copyright (c) ModelScope Contributors. All rights reserved.
"""Twinkle Kernel Module - Kernel orchestration layer."""
from typing import Optional, Union, Any, Dict
from logging import getLogger

from .base import (
    ModeType,
    DeviceType,
)
from .layer import register_layer_kernel, apply_layer_kernel, register_layer_batch
from .registry import (
    register_external_layer as _register_external_layer,
)

logger = getLogger(__name__)


__all__ = [
    "kernelize_model",
    "register_layer_kernel",
    "register_external_layer",
    "register_kernels",
]


def kernelize_model(
    model,
    mode: ModeType = "inference",
    device: Optional[DeviceType] = None,
    use_fallback: bool = True,
) -> Any:
    """Apply kernels to model (main entry point).

    Args:
        model: The PyTorch model to kernelize.
        mode: The mode for kernel selection ("inference" or "train").
        device: The device type (auto-detected if None).
        use_fallback: Whether to use original forward when no compatible kernel found.
            If False, raises ValueError when kernel is unavailable.

    Returns:
        The kernelized model.
    """
    model = apply_layer_kernel(model, mode=mode, device=device, use_fallback=use_fallback)

    # TODO: apply function-level kernel (Monkey Patch)
    # from .function import apply_function_kernel
    # model = apply_function_kernel(model, mode=mode, device=device)

    return model


def register_external_layer(layer_class: type, kernel_name: str) -> None:
    _register_external_layer(layer_class, kernel_name)


def register_kernels(config: Dict[str, Dict[str, Any]]) -> None:
    """Batch register kernels (framework integration API)."""
    if "layers" in config:
        for kernel_name, spec in config["layers"].items():
            device = spec.pop("device", "cuda")
            register_layer_kernel(kernel_name=kernel_name, device=device, **spec)

    if "functions" in config:
        logger.info("Function-level kernel registration is not implemented yet.")
