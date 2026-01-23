# Copyright (c) ModelScope Contributors. All rights reserved.
"""Twinkle Kernel Module - Kernel orchestration layer."""
from typing import Optional, Union, Any, Dict
from logging import getLogger
from .function import apply_function_kernel, register_function_kernel
from .base import (
    ModeType,
    DeviceType,
    is_kernels_enabled,
)
from .layer import register_layer_kernel, apply_layer_kernel, register_layer_batch
from .registry import (
    register_external_layer as _register_external_layer,
)

logger = getLogger(__name__)


__all__ = [
    "kernelize_model",
    "register_layer_kernel",
    "register_function_kernel",
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

    apply_function_kernel(device=device, mode=mode)

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
        from .function import register_function_batch

        functions = config["functions"]
        if isinstance(functions, dict):
            function_specs = []
            for func_name, spec in functions.items():
                if not isinstance(spec, dict):
                    raise TypeError(f"Function spec for {func_name} must be a dict.")
                if "func_name" not in spec:
                    spec["func_name"] = func_name
                function_specs.append(spec)
            register_function_batch(function_specs)
        else:
            register_function_batch(functions)
