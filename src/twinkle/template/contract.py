# Copyright (c) ModelScope Contributors. All rights reserved.
"""Twinkle-side view of the shared template contract owned by ``swift.dev.template_contract``."""

import numpy as np

try:
    from swift.dev.template_contract import (
        SHIFTED_FIELDS,
        TOKEN_FILL,
        BatchHooks,
        BatchTemplate,
        CompletionEncoding,
        Continuation,
        ConversationSample,
        DatasetTemplate,
        DecodeContext,
        DecodedResponse,
        EncodedRow,
        InputOrderCompletionMask,
        InputOrderLabels,
        InputOrderLossScale,
        InputOrderRow,
        KTOSample,
        MediaBundle,
        Message,
        ModelHook,
        ModelInfo,
        ModelRef,
        OutputOrderCompletionMask,
        OutputOrderLabels,
        OutputOrderLossScale,
        PreferencePair,
        PreferenceSample,
        PreparedRow,
        PromptEncoder,
        PromptEncoding,
        ResponseDecoder,
        RolloutState,
        SamplerTemplate,
        TemplateProtocol,
        TemplateSpec,
        Tool,
        append_and_finalize as _swift_append_and_finalize,
        append_input_order,
        coerce_tool_call_arguments,
        feature_has_media,
        refresh,
        render_delta,
        shift,
        shift_values,
        splice,
        splice_state,
        split_reasoning,
        split_reasoning_state,
        strip_special_tokens,
        unshift,
        unshift_values,
    )
except ImportError as error:
    raise ImportError(
        "twinkle requires a matching swift.dev.template_contract; install the pinned Swift revision before importing "
        "twinkle.template"
    ) from error


def append_and_finalize(row, tokens, *, scored, sampled, finalize):
    """Append through Swift after restoring token-aligned arrays lost to JSON serialization.

    HTTP samplers represent a batched token-type field such as Qwen's ``[1, T]``
    ``mm_token_type_ids`` as nested Python lists. Swift appends ndarray/tensor
    fields on their token axis, but would append scalars to the outer list and
    produce ``[[...], 0, 0]``. Normalize at this shared boundary so both the
    server-side sampler and client-side multi-turn ledger use the same layout.
    """
    normalized = dict(row)
    for key in TOKEN_FILL:
        value = normalized.get(key)
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            normalized[key] = np.asarray(value)
    return _swift_append_and_finalize(normalized, tokens, scored=scored, sampled=sampled, finalize=finalize)
