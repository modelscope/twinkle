from __future__ import annotations

"""Kernel module function - Function-level replacement with HF kernels integration."""
import importlib
from logging import getLogger
from typing import Callable, Iterable, List, Optional

from kernels.layer.func import FuncRepositoryProtocol
from kernels._versions import select_revision_or_version
from kernels.utils import get_kernel
from .registry import FunctionKernelSpec, get_global_function_registry
from .base import (
    ModeType,
    is_kernels_available,
    validate_device_type,
    validate_mode,
)

logger = getLogger(__name__)


def _load_from_hub(
    *,
    repo: Optional[FuncRepositoryProtocol],
    repo_id: Optional[str],
    revision: Optional[str],
    version: Optional[str],
    func_name: str,
) -> tuple[Callable, object]:
    """Resolve function implementation from a repo or Hub repo_id."""
    if repo is not None:
        module_cls = repo.load()
        module_instance = module_cls()

        def impl(*args, **kwargs):
            return module_instance(*args, **kwargs)

        return impl, module_instance

    assert repo_id is not None
    resolved = select_revision_or_version(repo_id, revision, version)
    kernel = get_kernel(repo_id, revision=resolved)
    func = getattr(kernel, func_name, None)
    if func is None:
        raise AttributeError(f"Kernel repo {repo_id} does not export {func_name}.")
    return func, func


def register_function_kernel(
    *,
    func_name: str,
    target_module: str,
    func_impl: Optional[Callable] = None,
    repo: Optional[FuncRepositoryProtocol] = None,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    version: Optional[str] = None,
    device: Optional[str] = None,
    mode: Optional[ModeType] = None,
) -> None:
    """Register a function kernel with the registry."""
    sources = [func_impl is not None, repo is not None, repo_id is not None]
    if sum(sources) != 1:
        raise ValueError("Provide exactly one of func_impl, repo, or repo_id.")
    if revision is not None and version is not None:
        raise ValueError("Either revision or version must be specified, not both.")
    if mode is not None:
        validate_mode(mode)

    get_global_function_registry().register(
        FunctionKernelSpec(
            func_name=func_name,
            target_module=target_module,
            func_impl=func_impl,
            repo=repo,
            repo_id=repo_id,
            revision=revision,
            version=version,
            device=device,
            mode=mode,
        )
    )


def register_function_batch(function_registry: Iterable[dict]) -> None:
    """Batch register function kernels from a list of spec dicts."""
    for spec in function_registry:
        register_function_kernel(
            func_name=spec["func_name"],
            target_module=spec["target_module"],
            func_impl=spec.get("func_impl"),
            repo=spec.get("repo"),
            repo_id=spec.get("repo_id"),
            revision=spec.get("revision"),
            version=spec.get("version"),
            device=spec.get("device"),
            mode=spec.get("mode"),
        )


def apply_function_kernel(
    *,
    target_module: Optional[str] = None,
    device: Optional[str] = None,
    mode: Optional[ModeType] = None,
    strict: bool = False,
) -> List[str]:
    """Apply registered function kernels by monkey-patching target modules.
    target_module: If specified, only apply kernels targeting this module.
    device: If specified, only apply kernels matching this device or with no device.
    mode: If specified, only apply kernels matching this mode or with no mode.
    strict: If True, raise errors on failures; otherwise log warnings.
    """
    applied = []
    if device is not None:
        validate_device_type(device)

    for spec in get_global_function_registry().list_specs():
        # Filter by target module and device/mode constraints.
        if target_module is not None and spec.target_module != target_module:
            continue
        if device is not None and spec.device is not None and spec.device != device:
            continue
        if spec.mode is not None and mode is None:
            msg = (
                "Function kernel registered with mode but apply_function_kernel "
                "was called without mode; skipping."
            )
            if strict:
                raise ValueError(msg)
            logger.warning(msg)
            continue
        if spec.mode is not None and mode is not None and spec.mode != mode:
            continue

        try:
            # Import the module that will be monkey-patched.
            module = importlib.import_module(spec.target_module)
        except Exception as exc:
            if strict:
                raise
            logger.warning(
                "Failed to import target module %s: %s",
                spec.target_module,
                exc,
            )
            continue

        # Resolve implementation and capability target for mode checks.
        if spec.func_impl is not None:
            impl = spec.func_impl
        else:
            if not is_kernels_available():
                msg = (
                    "HF kernels package not available. "
                    f"Cannot load function kernel: {spec.func_name}. "
                    "Install it with `pip install kernels`."
                )
                raise RuntimeError(msg)
            impl, _ = _load_from_hub(
                repo=spec.repo,
                repo_id=spec.repo_id,
                revision=spec.revision,
                version=spec.version,
                func_name=spec.func_name,
            )
        # Final patch (or reapply when no mode gating is used).
        setattr(module, spec.func_name, impl)
        applied.append(f"{spec.target_module}.{spec.func_name}")

    if strict and not applied:
        raise ValueError("No function kernels applied for the given filters.")

    return applied
