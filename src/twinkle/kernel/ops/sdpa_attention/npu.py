# Copyright (c) ModelScope Contributors. All rights reserved.
"""SDPA forward with Ascend NPU compatibility fixes."""
from __future__ import annotations

import torch


def npu_sdpa_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    dropout=0.0,
    scaling=None,
    is_causal=None,
    **kwargs,
):
    """Drop-in replacement for ``transformers.integrations.sdpa_attention.sdpa_attention_forward``.

    Fixes:
      - Repeats KV heads (NPU SDPA does not auto-broadcast num_kv_groups).
      - Truncates causal_mask to key length.
      - Forces contiguous tensors (NPU SDPA requirement).
      - Inverts boolean masks (NPU treats ``True`` as masked).
    """
    from transformers.integrations.sdpa_attention import repeat_kv

    if hasattr(module, 'num_key_value_groups'):
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None and causal_mask.ndim == 4:
        causal_mask = causal_mask[:, :, :, :key.shape[-2]]

    query, key, value = query.contiguous(), key.contiguous(), value.contiguous()

    if is_causal is None:
        is_causal = query.shape[2] > 1 and causal_mask is None

    if causal_mask is not None and causal_mask.dtype != torch.bool:
        causal_mask = torch.logical_not(causal_mask.bool()).to(query.device)

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=causal_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
    )
    return attn_output.transpose(1, 2).contiguous(), None
