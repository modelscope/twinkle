# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations


def _find_layers(model, cond_fn):
    """Every full mcore module path under ``model`` satisfying ``cond_fn(name, module)``."""
    return [name for name, module in model.named_modules() if cond_fn(name, module)]


def _find_embedding(model):
    """mcore's embedding modules, used to expand the ``'all-embedding'`` keyword."""
    from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
    return _find_layers(model, lambda name, module: isinstance(module, LanguageModelEmbedding) and 'lora' not in name)
