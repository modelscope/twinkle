# Copyright (c) ModelScope Contributors. All rights reserved.
from .awq import AwqQuantizer
from .base import CalibrationQuantizer, ConfigQuantizer, Quantizer
from .gptq import GptqQuantizer, GptqV2Quantizer
from .load_time import BnbQuantizer, EetqQuantizer, Fp8Quantizer, HqqQuantizer, QuantoQuantizer

# Keyed by the quant_method string transformers/vLLM use, so a config value maps straight to a class.
QUANTIZER_MAPPING = {
    cls.quant_method: cls
    for cls in (
        AwqQuantizer,
        GptqQuantizer,
        GptqV2Quantizer,
        BnbQuantizer,
        Fp8Quantizer,
        HqqQuantizer,
        QuantoQuantizer,
        EetqQuantizer,
    )
}


def get_quantizer(quant_method: str, **kwargs) -> Quantizer:
    """Build the quantizer registered under ``quant_method``."""
    if quant_method not in QUANTIZER_MAPPING:
        raise ValueError(f'Unknown quant_method: {quant_method!r}. '
                         f'Supported: {", ".join(sorted(QUANTIZER_MAPPING))}.')
    return QUANTIZER_MAPPING[quant_method](**kwargs)
