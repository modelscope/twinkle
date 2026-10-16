# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from twinkle.tq_utils import columns_to_tq_fields, rows_to_tq_fields

TRANSFORMERS_INPUT_FIELDS = (
    'input_ids',
    'labels',
    'attention_mask',
    'position_ids',
    'cu_seqlens',
    'completion_mask',
    'pixel_values',
    'image_grid_thw',
    'video_pixel_values',
    'video_grid_thw',
    'input_features',
    'feature_attention_mask',
)
REQUIRED_MODEL_INPUT_FIELDS = ('input_ids', 'labels', 'attention_mask', 'position_ids')
ROLLOUT_TRAIN_FIELDS = (*TRANSFORMERS_INPUT_FIELDS, 'logprobs', 'rewards', 'advantages', 'returns')

__all__ = [
    'ROLLOUT_TRAIN_FIELDS',
    'REQUIRED_MODEL_INPUT_FIELDS',
    'TRANSFORMERS_INPUT_FIELDS',
    'columns_to_tq_fields',
    'rows_to_tq_fields',
]
