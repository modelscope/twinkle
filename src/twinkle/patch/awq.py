# Copyright (c) ModelScope Contributors. All rights reserved.
"""Preserve tied and explicitly excluded modules during native AWQ loading."""
from functools import wraps


def apply_awq_compat(config) -> None:
    quantization = getattr(config, 'quantization_config', None)
    method = (quantization.get('quant_method') if isinstance(quantization, dict)
              else getattr(quantization, 'quant_method', None))
    if method != 'awq':
        return
    from packaging.version import Version
    from transformers import __version__

    if Version(__version__) < Version('4.50'):
        return
    try:
        from transformers.quantizers.base import get_keys_to_not_convert
    except ImportError:
        from transformers.integrations import get_keys_to_not_convert
    from transformers.quantizers.quantizer_awq import AwqQuantizer

    original = AwqQuantizer._process_model_before_weight_loading
    if getattr(original, '_twinkle_awq_compat', False):
        return

    @wraps(original)
    def process(self, model, *args, **kwargs):
        excluded = self.quantization_config.modules_to_not_convert
        if excluded is not None:
            self.quantization_config.modules_to_not_convert = list(dict.fromkeys(
                [*excluded, *get_keys_to_not_convert(model)]))
        return original(self, model, *args, **kwargs)

    process._twinkle_awq_compat = True
    AwqQuantizer._process_model_before_weight_loading = process
