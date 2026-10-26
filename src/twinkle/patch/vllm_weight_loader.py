# Copyright (c) ModelScope Contributors. All rights reserved.
"""Keep vLLM's loaded-parameter names and tied-weight checks valid during sync."""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_pending_alias_checks = ContextVar("twinkle_vllm_weight_alias_checks", default=None)


@contextmanager
def streaming_weight_sync():
    """Check tied weights against the complete sync, rather than one bucket."""
    patch_lora_weight_loader()
    checks = []
    token = _pending_alias_checks.set(checks)
    try:
        yield
        # Per-bucket validation can miss tied parameters loaded by another bucket.
        # Merge loaded names per module before running the original alias checks.
        loaded_by_module = {}
        for _, loader, loaded in checks:
            loaded_by_module.setdefault(id(loader.module), set()).update(loaded)
        for check, loader, _ in checks:
            check(loader, loaded_by_module[id(loader.module)])
    finally:
        _pending_alias_checks.reset(token)


def patch_lora_weight_loader():
    """Preserve vLLM's alias validation when LoRA delegates base-weight loads.

    The standard LoRA wrapper accepts both checkpoint names (``weight``)
    and internal names (``base_layer.weight``). Its result reports internal
    names for tied-weight validation. Custom loaders keep their own contract.
    """
    from vllm.lora.layers import BaseLayerWithLoRA
    from vllm.model_executor.models.utils import AutoWeightsLoader

    # Delegating "base_layer.weight" to the inner loader fails because it expects "weight".
    # Adapt input names, then report internal names so tied-weight validation still works.
    original_load = getattr(BaseLayerWithLoRA, "load_weights", None)
    if original_load is not None and not getattr(original_load, "_twinkle_weight_sync_patch", False):

        @wraps(original_load)
        def load_weights(self, weights):
            base_weights = ((name.removeprefix("base_layer."), tensor) for name, tensor in weights)
            loaded = original_load(self, base_weights)
            if loaded is None:
                return None
            internal_names = self.state_dict().keys()
            return {f"base_layer.{name}" if f"base_layer.{name}" in internal_names else name for name in loaded}

        load_weights._twinkle_weight_sync_patch = True
        BaseLayerWithLoRA.load_weights = load_weights

    original_check = getattr(AutoWeightsLoader, "_check_skipped_aliases", None)
    if original_check is not None and not getattr(original_check, "_twinkle_weight_sync_patch", False):

        @wraps(original_check)
        def check_skipped_aliases(self, autoloaded_weights):
            checks = _pending_alias_checks.get()
            if checks is None:
                return original_check(self, autoloaded_weights)
            checks.append((original_check, self, set(autoloaded_weights)))

        check_skipped_aliases._twinkle_weight_sync_patch = True
        AutoWeightsLoader._check_skipped_aliases = check_skipped_aliases
