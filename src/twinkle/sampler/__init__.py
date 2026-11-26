# Copyright (c) ModelScope Contributors. All rights reserved.
from importlib import import_module

# Loading the queue bridge must not initialize model backends in HTTP workers.
_LAZY_EXPORTS = {
    "Sampler": (".base", "Sampler"),
    "BaseSamplerEngine": (".base_engine", "BaseSamplerEngine"),
    "SGLangEngine": (".sglang_sampler.sglang_engine", "SGLangEngine"),
    "TransformersEngine": (".transformers_sampler.transformers_engine", "TransformersEngine"),
    "VLLMEngine": (".vllm_sampler.vllm_engine", "VLLMEngine"),
    "SGLangSampler": (".sglang_sampler", "SGLangSampler"),
    "TransformersSampler": (".transformers_sampler", "TransformersSampler"),
    "vLLMSampler": (".vllm_sampler", "vLLMSampler"),
}
__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
