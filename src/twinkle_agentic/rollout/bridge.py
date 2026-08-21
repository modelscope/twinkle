# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shared, pure bridge-token stitching logic for multi-turn rollouts.

This module hosts :func:`extend_with_bridge`, a ``self``-free function that
appends tool messages and the next generation prompt to a running
``InputFeature`` (``pif``) as ``-100`` "bridge" tokens. It is shared between
the core-library ``MultiTurnRollout`` and the client-side rollout so the two
paths cannot drift.

The logic was lifted verbatim from ``MultiTurnRollout._extend_with_bridge`` and
``MultiTurnRollout._append_bridge_tokens``; every ``self.template`` access was
rewritten to use the ``template`` parameter. No Ray decorators
(``@remote_function`` / ``@remote_class``) are applied here.
"""
import numpy as np
from typing import Any, Dict, List, Optional

from twinkle.template.base import Template


def _to_plain(obj: Any) -> Any:
    """Recursively convert numpy arrays/scalars to plain Python lists/numbers.

    Mirrors ``vllm_sampler._convert_ndarray_to_list`` but lives locally so we
    do not depend on a private symbol.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        conv = [_to_plain(x) for x in obj]
        return type(obj)(conv) if isinstance(obj, tuple) else conv
    return obj


def extend_with_bridge(
    pif: Dict[str, Any],
    tool_messages: List[Dict[str, Any]],
    template: Template,
) -> Optional[Dict[str, Any]]:
    """Append tool messages and the next generation prompt as -100 bridge.

    Strategy: compute the bridge ENTIRELY in template space. Render
    ``messages_before`` and ``messages_before + tool_messages`` with the
    same chat template and take ``s_after[len(s_before):]`` as the delta.

    We deliberately do NOT diff against ``tokenizer.decode(pif.input_ids)``
    because raw vLLM output and canonical template rendering differ in
    whitespace (e.g. Qwen inserts ``\\n\\n`` between assistant content and
    a ``<tool_call>`` block, while the model generates only ``\\n``). Such
    cosmetic divergences would break a ``startswith`` alignment but do not
    affect training correctness: history tokens stay in ``pif.input_ids``
    verbatim; only the newly appended bridge is tokenized from the
    canonical template output.

    Returns ``None`` when the trajectory exceeds ``max_length`` and the
    template's truncation strategy is ``'delete'``.
    """
    tokenizer = template.tokenizer

    messages_before = list(pif.get('messages') or [])
    messages_after = messages_before + list(tool_messages)

    enable_thinking = getattr(template, 'enable_thinking', False)
    s_before = tokenizer.apply_chat_template(
        messages_before, tokenize=False, add_generation_prompt=False, enable_thinking=enable_thinking)
    s_after = tokenizer.apply_chat_template(
        messages_after, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)

    if not s_after.startswith(s_before):
        raise RuntimeError('Canonical chat_template output for messages_after is not a '
                           'prefix-extension of messages_before; cannot compute bridge '
                           'delta. This indicates the template is non-monotonic in the '
                           'message list (e.g. reorders / rewrites earlier turns).\n'
                           f's_before tail: {s_before[-80:]!r}\n'
                           f's_after at same offset: '
                           f'{s_after[max(0, len(s_before) - 80):len(s_before) + 80]!r}')
    bridge_text = s_after[len(s_before):]
    if not bridge_text:
        raise RuntimeError('Bridge text computation returned empty string; '
                           'tool turn would add no tokens (template misconfiguration?).')

    bridge_ids = tokenizer.encode(bridge_text, add_special_tokens=False)
    if not bridge_ids:
        raise RuntimeError(f'Bridge text tokenised to empty id list: {bridge_text!r}')

    new_pif = _append_bridge_tokens(pif, bridge_ids, template)
    if new_pif is None:
        # Trajectory exceeds max_length and strategy is 'delete'
        return None
    new_pif['messages'] = messages_after
    return new_pif


def _append_bridge_tokens(
    pif: Dict[str, Any],
    bridge_ids: List[int],
    template: Template,
) -> Optional[Dict[str, Any]]:
    """Append bridge tokens with labels = -100.

    Mirrors the unroll-append-reroll pattern of
    :meth:`Template.concat_input_feature` so that ``labels`` semantics
    stay consistent with the sampler-produced pif.

    Shallow copy is deliberately used: every mutation below is a
    top-level key reassignment, never an in-place change to nested
    tensors. Multimodal payloads (``images``, ``pixel_values``,
    ``image_grid_thw`` ...) are shared by reference so we avoid
    re-copying image buffers every turn.
    """
    result = dict(pif)

    input_ids = list(result['input_ids'])
    labels = list(result.get('labels') or [])
    # labels arrive in output/shifted order (post _roll_labels). Unroll by
    # one position (shift right by 1) to get back to input order.
    if labels:
        if len(labels) != len(input_ids):
            raise RuntimeError(f'labels length ({len(labels)}) != input_ids length '
                               f'({len(input_ids)}); cannot safely append bridge tokens.')
        labels = labels[-1:] + labels[:-1]
    else:
        labels = [-100] * len(input_ids)

    input_ids = input_ids + list(bridge_ids)
    labels = labels + [-100] * len(bridge_ids)

    result['input_ids'] = input_ids
    result['labels'] = labels

    if 'mm_token_type_ids' in result:
        import torch
        mm = result['mm_token_type_ids']
        if not isinstance(mm, torch.Tensor):
            mm = torch.as_tensor(mm)
        # Pad along the last (sequence) dim — handles 1D [T] and 2D [1, T] uniformly.
        leading_shape = mm.shape[:-1]
        pad = torch.zeros((*leading_shape, len(bridge_ids)), dtype=mm.dtype, device=mm.device)
        result['mm_token_type_ids'] = torch.cat([mm, pad], dim=-1)

    # Replay the post pipeline: refresh attention_mask / position_ids /
    # length and re-roll labels back into output/shifted order.
    refreshed_list = template._invoke_post_pipeline([result])
    if not refreshed_list:
        # truncation_strategy='delete': trajectory exceeds max_length
        return None
    result.update(refreshed_list[0])
    return _to_plain(result)
