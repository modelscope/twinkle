# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import sys
from typing import Any, List, Optional, Union

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

OutputType = Union[np.ndarray, 'torch.Tensor', List[Any], float]


class ModelOutput(TypedDict, total=False):
    """The output structure for the LLM/MLLM.

    Text-related fields:
        logits: The logits output by the model.
        loss: The loss calculated by the model.
        logps: The log-probabilities of correct tokens by the model.
    """
    logits: Optional[OutputType]
    loss: Optional[OutputType]
    logps: Optional[OutputType]


class LossOutput(TypedDict, total=False):
    """The output structure for the Losses"""

    loss: Optional[OutputType]
    num_tokens: Optional[int]
