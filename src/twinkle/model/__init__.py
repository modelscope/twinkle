# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from twinkle.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .base import TwinkleModel
    from .megatron import MegatronModel, MultiLoraMegatronModel
    from .micro_batch import MicroBatchConfig
    from .transformers import MultiLoraTransformersModel, TransformersModel, TransformersValueModel

else:
    _import_structure = {
        'base': ['TwinkleModel'],
        'micro_batch': ['MicroBatchConfig'],
        'transformers': ['TransformersModel', 'MultiLoraTransformersModel', 'TransformersValueModel'],
        'megatron': ['MegatronModel', 'MultiLoraMegatronModel'],
    }

    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,  # noqa
        extra_objects={},
    )
