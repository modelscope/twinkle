# Copyright (c) ModelScope Contributors. All rights reserved.
import contextvars

from peft.tuners.lora import LoraLayer, LoraModel

from .base import Patch


class MultiAdapter(Patch):
    """Support thread-local loras in one base model, make each user forward and backward with the unique lora name.
    """

    _adapter_var = contextvars.ContextVar('adapter_names', default=None)

    def patch(self, module, **kwargs):
        if getattr(LoraLayer, '_patched', False):
            return module

        def get_active_adapter(*args, **kwargs):
            return MultiAdapter._adapter_var.get()

        def get_active_adapters(*args, **kwargs):
            adapter_name = MultiAdapter._adapter_var.get()
            if adapter_name:
                return [adapter_name]
            else:
                return []

        def set_active_adapters(_, value, *args, **kwargs):
            pass

        def set_adapter(self, adapter_names, *args, **kwargs):
            pass

        LoraLayer.active_adapter = property(get_active_adapter, set_active_adapters)
        LoraLayer.active_adapters = property(get_active_adapters, set_active_adapters)
        LoraLayer.set_adapter = set_adapter
        LoraLayer._patched = True
        LoraModel.active_adapter = property(get_active_adapter, set_active_adapters)
        LoraModel.active_adapters = property(get_active_adapters, set_active_adapters)
        LoraModel.set_adapter = set_adapter
        LoraModel._patched = True

        def _mark_only_adapters_as_trainable(self, model) -> None:
            for n, p in model.named_parameters():
                if "lora_" not in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

        LoraModel._mark_only_adapters_as_trainable = _mark_only_adapters_as_trainable
        module.set_current_adapter_name = MultiAdapter.set_current_adapter_name
        return module

    @staticmethod
    def set_current_adapter_name(adapter_name):
        MultiAdapter._adapter_var.set(adapter_name)