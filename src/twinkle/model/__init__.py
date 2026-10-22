# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING

from twinkle._lazy_module import _LazyModule

if TYPE_CHECKING:
    from .base import ModelLoaderProtocol, TrainableModel
    from .megatron import MegatronModel, MultiLoraMegatronModel
    from .micro_batch import MicroBatchConfig
    from .state_dict_utils import (MODULES_TO_SAVE_SEGMENT, collect_local_state, is_lora_state_key,
                                   is_saveable_adapter_key, normalize_adapter_keys, remap_optimizer_state_names,
                                   with_tied_weight_aliases)
    from .transformers import (MultiLoraTransformersModel, SpectralHybridTransformersModel, TransformersModel,
                               TransformersValueModel)

else:
    _import_structure = {
        'base': ['ModelLoaderProtocol', 'TrainableModel'],
        'micro_batch': ['MicroBatchConfig'],
        'state_dict_utils': [
            'MODULES_TO_SAVE_SEGMENT', 'collect_local_state', 'is_lora_state_key', 'is_saveable_adapter_key',
            'normalize_adapter_keys', 'remap_optimizer_state_names', 'with_tied_weight_aliases'
        ],
        'transformers': [
            'TransformersModel', 'MultiLoraTransformersModel', 'SpectralHybridTransformersModel',
            'TransformersValueModel'
        ],
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
