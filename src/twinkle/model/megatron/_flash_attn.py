# Copyright (c) ModelScope Contributors. All rights reserved.
"""FlashAttention version pinning for the Megatron backend.

``attn_impl`` can name a SPECIFIC FlashAttention major version (``flash_3`` in Megatron spelling,
``flash_attention_3`` in transformers spelling). transformer_engine has no per-call version
selector, so the only lever is its module-level availability flags: assert the requested build is
installed, then switch the others OFF so TE's dispatcher can only pick the requested one. This
reproduces legacy Megatron-SWIFT's ``MegatronArguments._init_attention_backend``.

These are transformer_engine MODULE GLOBALS, so the pin only affects the process that applies it.
It must therefore run on the worker that builds the model -- ``MegatronStrategy.__init__`` consumes
``attn_impl`` and calls ``apply_flash_version_pin`` there, which in Ray mode is the worker, not the
driver. (The kernel CHOICE -- which AttnBackend enum value reaches the config -- is resolved
separately and driver-side; this module only enforces the version.)
"""
from __future__ import annotations

from typing import Optional

# Both spellings of "pin FlashAttention to version N". Kept as prefixes rather than an explicit list
# so a future FA5 is refused too instead of falling through to the generic "unknown name" error.
_FLASH_VERSION_PIN_PREFIXES = ('flash_attention_', 'flash_')

# TE exposes one module-level availability flag per FlashAttention major version; pinning works by
# turning the others off (legacy does the same, megatron_args.py:905-915).
_FLASH_PIN_FLAGS = {2: 'is_installed', 3: 'v3_is_installed', 4: 'v4_is_installed'}
FLASH_PIN_SUPPORTED_VERSIONS = frozenset(_FLASH_PIN_FLAGS)


def flash_version_pin(attn_impl: Optional[str]) -> Optional[int]:
    """The FlashAttention version ``attn_impl`` pins, or None when it pins nothing.

    Accepts both spellings (Megatron ``flash_3`` / transformers ``flash_attention_3``) so the same
    intent resolves identically. Unversioned names ('flash', 'flash_attn', 'sdpa', ...) -> None.
    """
    if attn_impl is None:
        return None
    key = str(attn_impl).lower()
    for prefix in _FLASH_VERSION_PIN_PREFIXES:
        if key.startswith(prefix) and key[len(prefix):].isdigit():
            return int(key[len(prefix):])
    return None


def apply_flash_version_pin(attn_impl: Optional[str]) -> Optional[int]:
    """Force transformer_engine to use exactly the FlashAttention version ``attn_impl`` pins.

    Returns the pinned version, or None when nothing was pinned (then TE keeps its own choice).

    MUST run in the process that builds the model: these are transformer_engine module globals, so a
    driver-side call would not reach a Ray worker.
    """
    version = flash_version_pin(attn_impl)
    if version is None:
        return None
    from transformer_engine.pytorch.attention.dot_product_attention.utils import FlashAttentionUtils as fa_utils

    if version not in FLASH_PIN_SUPPORTED_VERSIONS:
        raise NotImplementedError(
            f'attn_impl={attn_impl!r} pins FlashAttention v{version}, which transformer_engine has no '
            f'availability flag for. Supported: {sorted(FLASH_PIN_SUPPORTED_VERSIONS)}.')
    # Only flags this TE build actually defines: v4_is_installed is absent on older versions, and a
    # bare getattr would surface as AttributeError instead of an actionable message.
    present = {v: flag for v, flag in _FLASH_PIN_FLAGS.items() if hasattr(fa_utils, flag)}
    if version not in present:
        raise ValueError(f'attn_impl={attn_impl!r} requests flash-attn v{version}, but the installed '
                         f'transformer_engine has no {_FLASH_PIN_FLAGS[version]!r} flag, so this version cannot be '
                         f'selected. Versions this TE build can pin: {sorted(present)}.')
    installed = {v: getattr(fa_utils, flag) for v, flag in present.items()}
    if not installed[version]:
        raise ValueError(f'attn_impl={attn_impl!r} requests flash-attn v{version}, which is not installed. '
                         f'Detected: ' + ', '.join(f'FA{v}={state}' for v, state in sorted(installed.items())))
    # Switch the other versions off so TE's dispatcher cannot fall back to one of them.
    for other, flag in present.items():
        if other != version:
            setattr(fa_utils, flag, False)
    return version
