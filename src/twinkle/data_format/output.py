# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import sys
from typing import Any, Dict, List, Optional, Union

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

OutputType = Union[np.ndarray, 'torch.Tensor', List[Any], float]

# Per-field frame declarations for a response-framed ModelOutput: 'response' fields are
# per-sample Python numeric lists aligned with response_token_ids; 'sequence' fields keep
# full-input tensors; 'row' fields keep per-sample task outputs (pooling/decision/seq_cls).
FIELD_FRAMES = ('response', 'sequence', 'row')


class ModelOutput(TypedDict, total=False):
    """The output structure for the LLM/MLLM.

    Text-related fields:
        logits: The logits output by the model.
        loss: The loss calculated by the model.
        logps: The log-probabilities of correct tokens by the model.
        num_tokens: The token denominator associated with ``loss``.
        values: Per-token scalar value estimates used by critic models.
        embeddings: The embeddings output by the model, used be embedding task.

    Response-frame metadata (only when requested via ``output_frame='response'``):
        field_frames: Per-field frame declaration, one of FIELD_FRAMES.
        episode_ids: Per-sample trajectory identities, same order as the response rows.
        response_token_ids: Per-sample response token ids each response row aligns with.
        tokenizer_fingerprint: Identity of the tokenizer that produced the labels.
    """
    logits: Optional[OutputType]
    loss: Optional[OutputType]
    logps: Optional[OutputType]
    values: Optional[OutputType]
    num_tokens: Optional[OutputType]
    embeddings: Optional[OutputType]
    routed_experts: Optional[OutputType]
    channel_loss: dict[str | None, OutputType] | None
    field_frames: Dict[str, str]
    episode_ids: List[str]
    response_token_ids: List[List[int]]
    tokenizer_fingerprint: Optional[str]


class LossOutput(TypedDict, total=False):
    """The output structure for the Losses."""

    loss: Optional[OutputType]
    num_tokens: Optional[OutputType]
    channel_loss: dict[str | None, OutputType] | None
