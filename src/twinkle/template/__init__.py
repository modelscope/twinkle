# Copyright (c) ModelScope Contributors. All rights reserved.
from importlib import import_module

from .contract import (
    CompletionEncoding,
    ConversationSample,
    DecodeContext,
    DecodedResponse,
    EncodedRow,
    InputOrderRow,
    KTOSample,
    MediaBundle,
    PreferencePair,
    PreferenceSample,
    PreparedRow,
    PromptEncoding,
    RolloutState,
)

_LAZY_EXPORTS = {
    "Template": (".base", "Template"),
    "DeepseekV4Template": (".deepseek_v4", "DeepseekV4Template"),
    "Qwen3_5Template": (".qwen3_5_vl", "Qwen3_5Template"),
    "ClineParser": (".tools", "ClineParser"),
    "HermesQwenParser": (".tools", "HermesQwenParser"),
    "ReActParser": (".tools", "ReActParser"),
    "ToolCallParser": (".tools", "ToolCallParser"),
    "ToolCallRegistry": (".tools", "ToolCallRegistry"),
    "VCPParser": (".tools", "VCPParser"),
}


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    value = getattr(import_module(target[0], __name__), target[1])
    globals()[name] = value
    return value
