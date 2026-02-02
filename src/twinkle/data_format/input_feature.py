# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import List, Union, Optional, Any
import numpy as np

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

InputType = Union[List[List[int]], List[int], np.ndarray, Any]


class InputFeature(TypedDict, total=False):
    """The input features for the LLM/MLLM.

    Text-related fields:
        input_ids: The input token list.
        attention_mask: The attention mask of the input_ids.
        position_ids: The position ids of the input_ids, can be used to distinguish sentences.
        labels: The labels of the input_ids, used to calculate loss.
        completion_mask: Boolean array used in RL algorithms, matched with logits_to_keep 
            to indicate which tokens need to calculate loss.
        logits_to_keep: The logits to keep when calculating loss.
        num_items_in_batch: The number of valid tokens in the batch.
        length: The length of input_ids.

    Multimodal fields (raw data, processed by engine/model):
        images: List of images (PIL.Image, file paths, or URLs). 
            These are raw images before model-specific processing.
        videos: List of videos (file paths or list of frames).
            These are raw videos before model-specific processing.
    """
    # Text-related fields
    input_ids: InputType
    attention_mask: InputType
    position_ids: InputType
    labels: InputType
    completion_mask: InputType
    length: int
    logits_to_keep: Optional[int]
    num_items_in_batch: Optional[int]
    
    # Multimodal fields (raw data)
    images: List[Any]
    videos: List[Any]



def to_transformers_dict(feature: InputFeature) -> dict:
    """Transfer the InputFeature object to a dict needed by `transformers` models."""
    import torch
    output = {}
    _keys = ['input_ids', 'input_embeddings', 'attention_mask', 'position_ids', 'labels', 'completion_mask', 'logits_to_keep', 'num_items_in_batch']
    for key in list(feature.keys()):
        if key in _keys:
            output[key] = np.array(feature[key]) if not isinstance(feature[key], torch.Tensor) else feature[key]
    return output
