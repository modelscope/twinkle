# Copyright (c) ModelScope Contributors. All rights reserved.
"""Quantization schemes transformers applies while LOADING the model.

Each class only builds the matching ``transformers`` quantization_config. See
``ConfigQuantizer`` for why ``quantize()`` is not the entry point here.
"""
from typing import List, Optional, Union

from twinkle.quantizer.base import ConfigQuantizer
from twinkle.utils import requires


class BnbQuantizer(ConfigQuantizer):
    """bitsandbytes 4bit/8bit -- the 'Q' in QLoRA.

    4bit + nf4 + double quant is the QLoRA recipe; pair it with a LoRA adapter to train.
    """

    quant_method = 'bnb'

    def __init__(self,
                 quant_bits: Optional[int] = 4,
                 bnb_4bit_compute_dtype: Optional[str] = None,
                 bnb_4bit_quant_type: str = 'nf4',
                 bnb_4bit_use_double_quant: bool = True,
                 bnb_4bit_quant_storage: Optional[str] = None,
                 llm_int8_skip_modules: Optional[List[str]] = None,
                 **kwargs):
        requires('bitsandbytes')
        super().__init__(quant_bits=quant_bits, **kwargs)
        self.bnb_4bit_compute_dtype = bnb_4bit_compute_dtype
        self.bnb_4bit_quant_type = bnb_4bit_quant_type
        self.bnb_4bit_use_double_quant = bnb_4bit_use_double_quant
        self.bnb_4bit_quant_storage = bnb_4bit_quant_storage
        self.llm_int8_skip_modules = llm_int8_skip_modules

    def get_quantization_config(self):
        import torch
        from transformers import BitsAndBytesConfig

        if self.quant_bits == 4:
            load_in_4bit, load_in_8bit = True, False
        elif self.quant_bits == 8:
            load_in_4bit, load_in_8bit = False, True
        else:
            raise ValueError(f'bnb does not support quant_bits: {self.quant_bits}, only 4 or 8.')
        compute_dtype = self.bnb_4bit_compute_dtype
        if isinstance(compute_dtype, str):
            compute_dtype = getattr(torch, compute_dtype)
        return BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
            bnb_4bit_quant_storage=self.bnb_4bit_quant_storage,
            llm_int8_skip_modules=self.llm_int8_skip_modules)


class Fp8Quantizer(ConfigQuantizer):
    """Fine-grained FP8. Bit-width is implied by the scheme, so quant_bits is not used."""

    quant_method = 'fp8'

    def __init__(self, modules_to_not_convert: Optional[List[str]] = None, **kwargs):
        # No requires(): FineGrainedFP8Config is implemented inside transformers itself.
        kwargs.pop('quant_bits', None)
        super().__init__(quant_bits=None, **kwargs)
        self.modules_to_not_convert = modules_to_not_convert

    def get_quantization_config(self):
        from transformers import FineGrainedFP8Config

        return FineGrainedFP8Config(modules_to_not_convert=self.modules_to_not_convert)


class HqqQuantizer(ConfigQuantizer):
    """Half-Quadratic Quantization (calibration-free, 1/2/3/4/8 bit)."""

    quant_method = 'hqq'

    def __init__(self,
                 quant_bits: Optional[int] = 4,
                 group_size: int = 64,
                 axis: Optional[int] = None,
                 skip_modules: Optional[List[str]] = None,
                 **kwargs):
        requires('hqq')
        super().__init__(quant_bits=quant_bits, **kwargs)
        self.group_size = group_size
        self.axis = axis
        self.skip_modules = skip_modules

    def get_quantization_config(self):
        from transformers import HqqConfig

        kwargs = {}
        if self.skip_modules is not None:
            kwargs['skip_modules'] = self.skip_modules
        return HqqConfig(nbits=self.quant_bits, group_size=self.group_size, axis=self.axis, **kwargs)


class QuantoQuantizer(ConfigQuantizer):
    """optimum-quanto. Maps quant_bits onto quanto's weight dtype strings."""

    quant_method = 'quanto'

    _WEIGHTS_MAP = {2: 'int2', 4: 'int4', 8: 'int8', 'float8': 'float8'}

    def __init__(self,
                 quant_bits: Optional[Union[int, str]] = 8,
                 modules_to_not_convert: Optional[List[str]] = None,
                 **kwargs):
        # Distribution is 'optimum-quanto'; the import path is optimum.quanto.
        requires('optimum-quanto')
        super().__init__(quant_bits=quant_bits, **kwargs)
        self.modules_to_not_convert = modules_to_not_convert

    def get_quantization_config(self):
        from transformers import QuantoConfig

        weights = self._WEIGHTS_MAP.get(self.quant_bits)
        if weights is None:
            raise ValueError('quanto quantization only support quant bits 2/4/8/float8, '
                             f'got {self.quant_bits}.')
        return QuantoConfig(weights=weights, modules_to_not_convert=self.modules_to_not_convert)


class EetqQuantizer(ConfigQuantizer):
    """EETQ int8 weight-only."""

    quant_method = 'eetq'

    def __init__(self, quant_bits: Optional[int] = 8, modules_to_not_convert: Optional[List[str]] = None, **kwargs):
        requires('eetq')
        super().__init__(quant_bits=quant_bits, **kwargs)
        self.modules_to_not_convert = modules_to_not_convert

    def get_quantization_config(self):
        from transformers import EetqConfig

        if self.quant_bits != 8:
            raise ValueError(f'eetq only supports quant_bits=8, got {self.quant_bits}.')
        return EetqConfig('int8', modules_to_not_convert=self.modules_to_not_convert)
