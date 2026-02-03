# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING
from twinkle.utils.import_utils import _LazyModule

if TYPE_CHECKING:
    from .base import TwinkleModel
    from .transformers import TransformersModel, MultiLoraTransformersModel
    from .megatron import MegatronModel, MultiLoraMegatronModel

else:
    _import_structure = {
        'base': ['TwinkleModel'],
        'transformers': ['TransformersModel', 'MultiLoraTransformersModel'],
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
