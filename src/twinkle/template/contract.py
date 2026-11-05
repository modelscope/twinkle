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
    from swift.dev.template_contract import append_and_finalize as _swift_append_and_finalize
except ImportError as error:
    raise ImportError(
        "twinkle requires a matching swift.dev.template_contract; install the pinned Swift revision before importing "
        "twinkle.template"
    ) from error


def _batched_token_fields_as_arrays(row):
    # swift extends numpy/torch token fields on their last axis but extends a Python list flat. A prompt that went
    # through a dataset stores the processor's batched [1, L] mm_token_type_ids as a nested list, which a flat extend
    # turns into the ragged [[0] * L, 0, 0, ...]; as an array it is extended to [1, L + n].
    converted = None
    for key in TOKEN_FILL:
        value = row.get(key)
        if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (list, tuple)):
            if converted is None:
                converted = dict(row)
            converted[key] = np.asarray(value)
    return row if converted is None else converted


def append_and_finalize(row, tokens, *, scored, sampled, finalize):
    """``swift.dev.template_contract.append_and_finalize``, extending batched list token fields on their last axis."""
    return _swift_append_and_finalize(
        _batched_token_fields_as_arrays(row), tokens, scored=scored, sampled=sampled, finalize=finalize
    )
