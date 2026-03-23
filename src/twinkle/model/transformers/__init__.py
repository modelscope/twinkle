# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from twinkle.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .models import (
        TwinkleQwen3_5DecoderLayer,
        TwinkleQwen3_5ForCausalLM,
        TwinkleQwen3_5GatedDeltaNet,
        TwinkleQwen3_5PreTrainedModel,
        TwinkleQwen3_5TextModel,
    )
    from .multi_lora_transformers import MultiLoraTransformersModel
    from .transformers import TransformersModel
else:
    _import_structure = {
        'transformers': ['TransformersModel'],
        'multi_lora_transformers': ['MultiLoraTransformersModel'],
        'models': [
            'TwinkleQwen3_5PreTrainedModel',
            'TwinkleQwen3_5TextModel',
            'TwinkleQwen3_5DecoderLayer',
            'TwinkleQwen3_5GatedDeltaNet',
            'TwinkleQwen3_5ForCausalLM',
        ],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,  # noqa
        extra_objects={},
    )
