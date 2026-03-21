# Copyright (c) ModelScope Contributors. All rights reserved.

# Megatron-related dependencies are optional (megatron-core / transformer-engine, etc.).
# We cannot import them unconditionally at package import time, because `twinkle.model.megatron.*`
# submodules import this file first, which would crash even if the user only wants the transformers backend.
# Follow the same LazyModule approach as `twinkle.model`: only import when those symbols are actually accessed.
from typing import TYPE_CHECKING

from twinkle import Platform
from twinkle.utils.import_utils import _LazyModule

if Platform.device_prefix() == 'npu':
    # MindSpeed needs to patch `torch.compile`/TE symbols before any `megatron.core`
    # module binds them by value. Keeping this import early is the smallest reliable hook.
    import mindspeed.megatron_adaptor  # noqa: F401

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
