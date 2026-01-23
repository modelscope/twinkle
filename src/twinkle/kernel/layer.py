# Copyright (c) ModelScope Contributors. All rights reserved.
"""Kernel module layer - Layer-level replacement with HF kernels integration."""
from logging import getLogger
from pathlib import Path
from typing import Optional, Union, Any

from .base import (
    DeviceType,
    ModeType,
    is_kernels_available,
    is_kernels_enabled,
    to_kernels_mode,
    get_device_type,
)
from .registry import register_layer, get_global_layer_registry

logger = getLogger(__name__)


def register_layer_kernel(
    kernel_name: str,
    repo_id: Optional[str] = None,
    repo_path: Optional[Union[str, Path]] = None,
    package_name: Optional[str] = None,
    layer_name: Optional[str] = None,
    version: Optional[str] = None,
    device: DeviceType = "cuda",
    mode: Optional[ModeType] = None,
) -> None:
    """Register a layer kernel with the registry.

    Args:
        kernel_name: Unique kernel name (can register multiple modes with same name)
        repo_id: Hub repository ID
        repo_path: Local repository path
        package_name: Package name (required when using repo_path)
        layer_name: Layer name (defaults to kernel_name)
        version: Version constraint
        device: Device type
        mode: Mode (train/inference/compile), None means FALLBACK
    """
    if not is_kernels_available():
        logger.warning(f"HF kernels package not available. Skipping registration for kernel: {kernel_name}")
        return

    from kernels import LayerRepository, LocalLayerRepository

    if repo_path is not None:
        if package_name is None:
            raise ValueError(f"package_name must be provided when using repo_path for kernel: {kernel_name}")
        if isinstance(repo_path, str):
            repo_path = Path(repo_path)
        repo_spec = LocalLayerRepository(
            repo_path=repo_path,
            package_name=package_name,
            layer_name=layer_name or kernel_name,
        )
    else:
        if repo_id is None:
            raise ValueError(f"Either repo_id or repo_path must be provided for kernel: {kernel_name}")
        repo_spec = LayerRepository(
            repo_id=repo_id,
            layer_name=layer_name or kernel_name,
            version=version,
        )

    hf_mode = _to_hf_mode(mode)
    register_layer(kernel_name, repo_spec, device, mode=hf_mode)

    mode_str = mode or "FALLBACK"
    logger.info(f"Registered layer kernel: {kernel_name} for device: {device}, mode: {mode_str}")


def _to_hf_mode(mode: Optional[ModeType]) -> Any:
    """Convert Twinkle mode to HF kernels Mode."""
    if mode is None:
        from kernels import Mode
        return Mode.FALLBACK
    return to_kernels_mode(mode)


def apply_layer_kernel(
    model,
    mode: ModeType = "inference",
    device: Optional[DeviceType] = None,
    use_fallback: bool = True,
) -> Any:
    """Apply layer kernels to model.

    Args:
        model: The PyTorch model to kernelize.
        mode: The mode for kernel selection ("inference" or "train").
        device: The device type (auto-detected if None).
        use_fallback: Whether to use original forward when no compatible kernel found.
            If False, raises ValueError when kernel is unavailable.

    Returns:
        The kernelized model.
    """
    if not is_kernels_enabled():
        logger.debug("Kernels not enabled, returning original model")
        return model

    get_global_layer_registry().sync_to_hf_kernels()

    if device is None:
        device = get_device_type() or "cuda"

    kernel_mode = to_kernels_mode(mode)

    try:
        from kernels import kernelize
        logger.debug(f"Applying kernels with mode: {mode}, device: {device}, use_fallback: {use_fallback}")
        return kernelize(model, mode=kernel_mode, device=device, use_fallback=use_fallback)
    except Exception as e:
        if use_fallback:
            logger.warning(f"Failed to apply kernels: {e}. Returning original model.")
            return model
        raise


def register_layer_batch(mapping: dict, default_device: DeviceType = "cuda") -> None:
    """Batch register layer kernels."""
    for kernel_name, spec in mapping.items():
        device = spec.pop("device", default_device)
        register_layer_kernel(kernel_name=kernel_name, device=device, **spec)
