from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from twinkle.data_format.input_feature import InputFeature

if TYPE_CHECKING:
    from tinker import types
    
def datum_to_input_feature(datum: types.Datum) -> InputFeature:
    """Convert a Datum to a dictionary of input features for model inference."""
    input_feature: InputFeature = {}

    # 1. Flatten model_input chunks to get input_ids
    input_ids = datum.model_input.to_ints()
    input_feature['input_ids'] = input_ids
    input_feature['attention_mask'] = [1] * len(input_ids)
    input_feature['length'] = len(input_ids)
    input_feature['position_ids'] = list(range(len(input_ids)))
    
    # 2. Map loss function inputs
    # 'target_tokens' -> 'labels'
    if 'target_tokens' in datum.loss_fn_inputs and 'weights' in datum.loss_fn_inputs:
        weights = datum.loss_fn_inputs['weights'].to_numpy()
        labels = datum.loss_fn_inputs['target_tokens'].to_numpy()
        
        input_feature['labels'] = np.where(weights > 0, labels, -100).tolist()

    return input_feature