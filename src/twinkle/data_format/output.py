# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import sys
from typing import TYPE_CHECKING, Any, List, Union

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

if TYPE_CHECKING:
    import torch

OutputType = Union[np.ndarray, 'torch.Tensor', List[Any]]


class ModelOutput(TypedDict, total=False):
    """The output structure for the LLM/MLLM.

    Text-related fields:
        logits: The logits output by the model.
        loss: The loss calculated by the model.
    """
    logits: OutputType
    loss: OutputType
