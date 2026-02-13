# Copyright (c) ModelScope Contributors. All rights reserved.

# Megatron-related dependencies are optional (megatron-core / transformer-engine, etc.).
# We cannot import them unconditionally at package import time, because `twinkle.model.megatron.*`
# submodules import this file first, which would crash even if the user only wants the transformers backend.
# Follow the same LazyModule approach as `twinkle.model`: only import when those symbols are actually accessed.
from typing import TYPE_CHECKING

from twinkle.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .megatron import MegatronModel, MegatronStrategy
    from .multi_lora_megatron import MultiLoraMegatronModel
else:
    _import_structure = {
        'megatron': ['MegatronStrategy', 'MegatronModel'],
        'multi_lora_megatron': ['MultiLoraMegatronModel'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,  # noqa
        extra_objects={},
    )
