# Copyright (c) ModelScope Contributors. All rights reserved.

# LLMModelType/MLLMModelType: model_type attribute in model config
class LLMModelType:
    qwen2 = 'qwen2'
    qwen2_moe = 'qwen2_moe'
    qwen3 = 'qwen3'
    qwen3_moe = 'qwen3_moe'

class MLLMModelType:
    qwen2_vl = 'qwen2_vl'
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'
    qwen3_vl_moe = 'qwen3_vl_moe'

class ModelType(LLMModelType, MLLMModelType):
    pass

# LLMMegatronModelType/MLLMMegatronModelType: megatron model architecture type
class LLMMegatronModelType:
    gpt = 'gpt'


class MLLMMegatronModelType:
    qwen2_vl = 'qwen2_vl'
    qwen2_5_vl = 'qwen2_5_vl'
    qwen3_vl = 'qwen3_vl'

class MegatronModelType(LLMMegatronModelType, MLLMMegatronModelType):
    pass
