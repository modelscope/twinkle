# Copyright (c) ModelScope Contributors. All rights reserved.
from ..constant import ModelType, MegatronModelType
from ..register import MegatronModelMeta, register_megatron_model

register_megatron_model(
    MegatronModelMeta(
        MegatronModelType.gpt,
        [
            ModelType.qwen2,
            ModelType.qwen3,
            ModelType.qwen2_moe,
            ModelType.qwen3_moe,
        ],
    ))
