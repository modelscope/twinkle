# Copyright (c) ModelScope Contributors. All rights reserved.
"""Propagate DeepSpeed ZeRO-3 ``ds_grads_remaining`` into peft ``modules_to_save``.

``ModulesToSaveWrapper`` is peft's wrapper for modules trained in full precision
alongside LoRA (``modules_to_save``, e.g. embeddings / lm_head / a value head). It
holds the real trainable submodules in ``self.modules_to_save`` (a ``ModuleDict``).

Under DeepSpeed ZeRO-3, gradient partitioning drives reduction off a per-module
``ds_grads_remaining`` counter (how many grads are still pending before the module's
grads can be reduced). DeepSpeed sets that counter on the *wrapper*, but the params
actually producing grads live in the inner ``modules_to_save`` submodules, which never
receive the update — so their reduce timing is wrong and those full-trained modules get
**silently incorrect gradients**. This is a correctness bug, not a memory one.

This patch rewrites ``ModulesToSaveWrapper.__setattr__`` so that whenever
``ds_grads_remaining`` is set on the wrapper, the same value is forwarded to every inner
``modules_to_save`` submodule. Mirrors legacy swift's ``_patch_modules_to_save_zero3``
(swift/pipelines/train/tuner.py).

Double-gated so it is a no-op outside the exact ZeRO-3 + modules_to_save combo:
  1. if peft has no ``ModulesToSaveWrapper`` -> nothing to patch;
  2. per setattr, only the ``ds_grads_remaining`` name forwards; every other attribute
     assignment falls through to the original ``__setattr__`` unchanged.

Wire this into the DeepSpeed ZeRO-3 strategy setup when ``modules_to_save`` (full-precision
trainable modules alongside LoRA) is in use::

    from twinkle.patch import apply_patch
    from twinkle.patch.peft_modules_to_save import PeftModulesToSaveZero3Patch
    apply_patch(None, PeftModulesToSaveZero3Patch())
"""
from twinkle.patch import Patch
from twinkle.utils import get_logger

logger = get_logger()

_MARKER = '_twinkle_origin_setattr'


def _modules_to_save_wrapper_cls():
    """Return peft's ``ModulesToSaveWrapper`` class, or ``None`` when unavailable."""
    try:
        from peft.utils import ModulesToSaveWrapper
        return ModulesToSaveWrapper
    except ImportError:
        return None


class PeftModulesToSaveZero3Patch(Patch):
    """Forward ZeRO-3 ``ds_grads_remaining`` from ``ModulesToSaveWrapper`` to its inner
    ``modules_to_save`` submodules. Idempotent, reversible, no-op without the wrapper."""

    def __call__(self, module=None, *args, **kwargs):
        ModulesToSaveWrapper = _modules_to_save_wrapper_cls()
        if ModulesToSaveWrapper is None or hasattr(ModulesToSaveWrapper, _MARKER):
            return module

        origin_setattr = ModulesToSaveWrapper.__setattr__

        def __setattr__(self, name, value):
            origin_setattr(self, name, value)
            if name == 'ds_grads_remaining':
                # modules_to_save is a ModuleDict of the real full-precision submodules;
                # guard in case ds_grads_remaining is ever set before it exists.
                modules_to_save = getattr(self, 'modules_to_save', None)
                if modules_to_save:
                    for submodule in modules_to_save.values():
                        submodule.ds_grads_remaining = value

        setattr(ModulesToSaveWrapper, _MARKER, origin_setattr)
        ModulesToSaveWrapper.__setattr__ = __setattr__
        logger.info('Patched peft ModulesToSaveWrapper.__setattr__ for DeepSpeed ZeRO-3 grad propagation.')
        return module

    def unpatch(self, module=None, *args, **kwargs):
        ModulesToSaveWrapper = _modules_to_save_wrapper_cls()
        if ModulesToSaveWrapper is None:
            return module
        origin = getattr(ModulesToSaveWrapper, _MARKER, None)
        if origin is not None:
            ModulesToSaveWrapper.__setattr__ = origin
            delattr(ModulesToSaveWrapper, _MARKER)
        return module
