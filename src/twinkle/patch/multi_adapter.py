import contextvars

from peft.tuners.lora import LoraLayer

from .base import Patch


class MultiAdapter(Patch):

    _adapter_var = contextvars.ContextVar('adapter_names', default=None)

    def __call__(self, module, **kwargs):
        if getattr(LoraLayer, '_patched', False):
            return module

        def get_active_adapters(_):
            adapter_name = MultiAdapter._adapter_var.get()
            if adapter_name:
                return [adapter_name]
            else:
                return []

        def set_active_adapters(_, value):
            pass

        LoraLayer.active_adapters = property(get_active_adapters, set_active_adapters)
        LoraLayer._patched = True

        module.set_current_adapter_name = MultiAdapter.set_current_adapter_name
        return module

    @staticmethod
    def set_current_adapter_name(adapter_name):
        MultiAdapter._adapter_var.set(adapter_name)