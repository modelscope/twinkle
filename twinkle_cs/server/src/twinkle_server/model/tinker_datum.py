# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import numpy as np
from tinker import types

from twinkle.data_format.input_feature import InputFeature
from twinkle.template import Template


def datum_to_input_feature(datum: types.Datum | list[types.Datum],
                           template: Template) -> InputFeature | list[InputFeature]:
    """Convert a Datum to a dictionary of input features for model inference.

    Supports multimodal inputs (images) via ImageChunk and ImageAssetPointerChunk,
    and can restore preprocessed multimodal tensors packed in ``loss_fn_inputs``
    (e.g. ``pixel_values``, ``image_grid_thw``) for Tinker-Twinkle bridge.
    """
    if isinstance(datum, list):
        return [datum_to_input_feature(d, template) for d in datum]

    input_feature: InputFeature = {}

    # 1. Flatten model_input chunks to get input_ids
    input_ids = datum.model_input.to_ints()
    input_feature['input_ids'] = input_ids

    # 2. Map loss function inputs
    # 'target_tokens' -> 'labels'
    assert 'target_tokens' in datum.loss_fn_inputs, f"Missing 'target_tokens' in loss_fn_inputs {datum.loss_fn_inputs}"

    labels = datum.loss_fn_inputs['target_tokens'].to_numpy()
    if 'weights' in datum.loss_fn_inputs:
        # remove weights 0 from labels
        weights = datum.loss_fn_inputs['weights'].to_numpy()
        input_feature['labels'] = np.where(weights != 0, labels, -100).tolist()
    else:
        # remove padding (0-id)
        input_feature['labels'] = np.where(labels != 0, labels, -100).tolist()
        # add weights to loss_fn_inputs
        weights = (labels != 0).astype(np.float32)
        datum.loss_fn_inputs['weights'] = types.TensorData.from_numpy(weights)

    # 3. Restore multimodal features packed in loss_fn_inputs (if any)
    #    For multimodal models, the client may send preprocessed image tensors
    #    such as ``pixel_values`` and ``image_grid_thw`` via loss_fn_inputs to
    #    avoid re-running image preprocessing on the server.
    if 'pixel_values' in datum.loss_fn_inputs:
        input_feature['pixel_values'] = datum.loss_fn_inputs['pixel_values'].to_numpy()
    if 'image_grid_thw' in datum.loss_fn_inputs:
        input_feature['image_grid_thw'] = datum.loss_fn_inputs['image_grid_thw'].to_numpy()

    # 4. Invoke post-pipeline hooks
    input_feature = template._add_attention_fields(input_feature)[0]
    return input_feature


def extract_rl_features_for_loss(datum: types.Datum | list[types.Datum]) -> dict:
    """Extract RL features from datums for use as loss kwargs.

    Converts per-datum feature lists into the format expected by loss functions:
    - 'logprobs'  -> 'old_logps'   : list of per-datum log-probability lists (for GRPO)
    - 'advantages'-> 'advantages'  : list of per-datum advantage lists (for GRPO)
    - 'ref_logps' -> 'ref_outputs' : {'logps': torch.Tensor [B, T]} (for DPO)
    """
    import torch
    if not isinstance(datum, list):
        datum = [datum]

    old_logps, advantages, ref_logps_lists = [], [], []
    for d in datum:
        if 'logprobs' in d.loss_fn_inputs:
            old_logps.append(d.loss_fn_inputs['logprobs'].to_numpy().tolist())
        if 'advantages' in d.loss_fn_inputs:
            advantages.append(d.loss_fn_inputs['advantages'].to_numpy().tolist())
        if 'ref_logps' in d.loss_fn_inputs:
            ref_logps_lists.append(d.loss_fn_inputs['ref_logps'].to_numpy().tolist())

    result = {}
    if old_logps:
        result['old_logps'] = old_logps
    if advantages:
        result['advantages'] = advantages
    if ref_logps_lists:
        from twinkle.utils import pad_and_stack_tensors
        ref_logps_tensors = [torch.tensor(r, dtype=torch.float32) for r in ref_logps_lists]
        result['ref_outputs'] = {'logps': pad_and_stack_tensors(ref_logps_tensors, pad_value=0.0, concat=False)}
    return result
