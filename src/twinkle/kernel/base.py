# Copyright (c) ModelScope Contributors. All rights reserved.
"""Kernel module base - Base classes, env vars, device detection."""
import os
from typing import Optional, Literal, Any
from ..utils import exists


def _kernels_enabled() -> bool:
    """Check if kernels are enabled (default: enabled)."""
    env_val = os.getenv("TWINKLE_USE_KERNELS", "YES").upper()
    return env_val in ("YES", "TRUE", "1", "ON")


def _trust_remote_code() -> bool:
    """Check if remote code is trusted (default: not trusted)."""
    env_val = os.getenv("TWINKLE_TRUST_REMOTE_CODE", "NO").upper()
    return env_val in ("YES", "TRUE", "1", "ON")


ModeType = Literal["train", "inference", "compile"]
DeviceType = Literal["cuda", "npu", "mps", "cpu", "rocm", "metal"]


def get_device_type() -> Optional[DeviceType]:
    """Auto-detect current device type."""
    if exists("torch"):
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        if hasattr(torch, "backends") and hasattr(torch.backends, "mps"):
            if torch.backends.mps.is_available():
                return "mps"
    return None


def detect_backend() -> Optional[str]:
    """Detect training framework backend: "transformers" | "megatron" | None."""
    if exists("transformers"):
        return "transformers"
    return None


def is_kernels_available() -> bool:
    """Check if HF kernels package is available."""
    return exists("kernels")


def is_kernels_enabled() -> bool:
    """Check if kernels are enabled by env var."""
    return _kernels_enabled() and is_kernels_available()


def to_kernels_mode(mode: ModeType) -> Any:
    """Convert Twinkle mode to HF kernels mode."""
    if not is_kernels_available():
        return None
    from kernels import Mode
    if isinstance(mode, Mode):
        return mode
    mode_map = {
        "train": Mode.TRAINING,
        "inference": Mode.INFERENCE,
        "compile": Mode.TORCH_COMPILE,
    }
    return mode_map.get(mode, Mode.INFERENCE)


def validate_mode(mode: str) -> None:
    from kernels.layer.mode import Mode
    mode = to_kernels_mode(mode)

    if mode == Mode.FALLBACK:
        raise ValueError("Mode.FALLBACK can only be used to register kernel mappings.")
    if Mode.INFERENCE not in mode and Mode.TRAINING not in mode:  # type: ignore[operator]
        raise ValueError("kernelize mode must contain Mode.INFERENCE or Mode.TRAINING.")


def supports_mode(target: object, mode: str) -> bool:
    from kernels.layer.mode import Mode
    mode = to_kernels_mode(mode)
    if Mode.TORCH_COMPILE in mode and not getattr(target, "can_torch_compile", False):
        return False
    if Mode.TRAINING in mode and not getattr(target, "has_backward", True):
        return False
    return True


def validate_device_type(device_type: str) -> None:
    supported_devices = {"cpu", "cuda", "mps", "npu", "rocm", "xpu"}
    if device_type not in supported_devices:
        raise ValueError(
            "Unsupported device type "
            f"'{device_type}'. Supported device types are: "
            f"{', '.join(sorted(supported_devices))}"
        )
