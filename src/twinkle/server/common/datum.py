# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import numpy as np
from collections import defaultdict
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


def extract_rl_feature(datum: types.Datum | list[types.Datum]) -> dict:
    if not isinstance(datum, list):
        datum = [datum]

    result = defaultdict(list)
    for d in datum:
        # 'logprobs' -> 'old_logps' (for GRPO loss)
        if 'logprobs' in d.loss_fn_inputs:
            old_logps = d.loss_fn_inputs['logprobs'].to_numpy().tolist()
            result['old_logps'].append(old_logps)

        # 'advantages' -> 'advantages' (for GRPO loss)
        if 'advantages' in d.loss_fn_inputs:
            advantages = d.loss_fn_inputs['advantages'].to_numpy().tolist()
            result['advantages'].append(advantages)
    return result


def input_feature_to_datum(input_feature: InputFeature) -> types.Datum:
    """Convert an input feature dictionary to a Datum object.

    This assumes a single sequence in ``input_ids``. ``labels`` values of
    ``-100`` are treated as masked positions and will be encoded with
    zero weights so that converting back via ``datum_to_input_feature``
    reproduces the same labels.

    For multimodal models, this function can optionally pack already
    preprocessed image tensors (e.g. ``pixel_values``, ``image_grid_thw``)
    into ``loss_fn_inputs`` so that the server can restore them without
    re-running image preprocessing.
    """

    # 1. Build ModelInput from input_ids
    input_ids = input_feature['input_ids']
    if isinstance(input_ids, np.ndarray):
        tokens = input_ids.astype(np.int64).flatten().tolist()
    elif isinstance(input_ids, list):
        # If it's a batched shape [B, T], take the first sequence by
        # convention; otherwise treat it as a flat token list.
        if input_ids and isinstance(input_ids[0], list):
            tokens = [int(t) for t in input_ids[0]]
        else:
            tokens = [int(t) for t in input_ids]
    else:
        tokens = np.asarray(input_ids.cpu(), dtype=np.int64).flatten().tolist()

    model_input = types.ModelInput.from_ints(tokens)

    # 2. Build loss_fn_inputs from labels (if present)
    loss_fn_inputs: types.LossFnInputs = {}

    if 'labels' in input_feature and input_feature['labels'] is not None:
        labels_raw = input_feature['labels']
        if isinstance(labels_raw, np.ndarray):
            labels_arr = labels_raw.astype(np.int64)
        elif isinstance(labels_raw, list):
            labels_arr = np.asarray(labels_raw, dtype=np.int64)
        else:
            labels_arr = np.asarray(labels_raw.cpu(), dtype=np.int64)

        labels_arr = labels_arr.reshape(-1)

        # Ensure labels length does not exceed token length to avoid shape
        # mismatches; if shorter, we leave tokens as-is since extra tokens
        # will simply not have associated loss.
        if labels_arr.shape[0] > len(tokens):
            labels_arr = labels_arr[:len(tokens)]

        # Positions with label == -100 are considered padding/ignored.
        weights_arr = (labels_arr != -100).astype(np.float32)
        target_tokens_arr = np.where(labels_arr == -100, 0, labels_arr)

        loss_fn_inputs['target_tokens'] = types.TensorData.from_numpy(target_tokens_arr)
        loss_fn_inputs['weights'] = types.TensorData.from_numpy(weights_arr)

    # 3. Optionally pack multimodal tensors into loss_fn_inputs so that
    #    the server-side ``datum_to_input_feature`` can restore them.
    for key in ('pixel_values', 'image_grid_thw'):
        if key in input_feature and input_feature[key] is not None:
            value = input_feature[key]
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            elif not isinstance(value, np.ndarray):
                value = np.asarray(value)
            loss_fn_inputs[key] = types.TensorData.from_numpy(value)

    return types.Datum(loss_fn_inputs=loss_fn_inputs, model_input=model_input)
