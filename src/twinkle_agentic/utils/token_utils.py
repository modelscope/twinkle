# Copyright (c) ModelScope Contributors. All rights reserved.
"""Growing a token sequence one turn at a time, in template space.

``self``-free functions that extend a running ``InputFeature`` (``pif``), all
measuring what a turn adds by diffing rendered chat-template output rather than
by pasting special tokens together -- which is what makes them hold for any chat
template:

* :func:`append_ids` grows the sequence by raw ids, trainable or not.
* :func:`extend_with_bridge` appends tool messages and the next generation
  prompt as ``-100`` "bridge" tokens.
* :func:`encode_appended_turn` returns the tokens an assistant turn written
  outside the sampler (an API, an agent, a human) contributes.

They live here rather than on a rollout because several callers need the same
answers -- the core-library ``MultiTurnRollout``, the client-side one, and the
ledger an external agent's rounds are booked into -- and a second copy of this
arithmetic is a second set of off-by-one bugs. No Ray decorators
(``@remote_function`` / ``@remote_class``) are applied here.
"""


import numpy as np
from typing import Any, Dict, List, Optional

from twinkle.template.base import Template

# Stand-in history for the fallback delta computation in
# :func:`extend_with_bridge`. A single user turn, because what precedes the
# appended message must itself render the same way with and without it: a user
# turn has no reasoning block for the template to move or drop.
_ANCHOR = [{'role': 'user', 'content': 'x'}]


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


def _delta_text(
    template: Template,
    messages_before: List[Dict[str, Any]],
    appended: List[Dict[str, Any]],
    *,
    gen_prompt_before: bool,
    gen_prompt_after: bool,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Text the chat template adds when ``appended`` is tacked onto history.

    ``gen_prompt_*`` place the delta relative to the generation prompt: a bridge
    ends on one (``False -> True``), a completion consumes one
    (``True -> False``).
    """
    tokenizer = template.tokenizer
    enable_thinking = getattr(template, 'enable_thinking', False)

    def render(messages: List[Dict[str, Any]], add_generation_prompt: bool) -> str:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking)

    s_before = render(messages_before, gen_prompt_before)
    s_after = render(list(messages_before) + list(appended), gen_prompt_after)

    if not s_after.startswith(s_before):
        # Appending a *user* message moves where Qwen3's template thinks the
        # conversation's last question is, and it renders assistant turns either
        # side of that point differently: the turn before it loses its <think>
        # block, and the turn after it gains an empty one when it had none.
        # Measured on Qwen3-4B with three messages -- rendered alone, the
        # assistant turn reads '<think>\nthinking hard\n</think>\n\nAll tasks are
        # complete.'; rendered with a user turn after it, just 'All tasks are
        # complete.'. Tool messages do not move that point, which is why
        # appending tool observations has always been a clean extension.
        #
        # So the delta is measured against a stand-in history instead: render one
        # user turn, then the same turn plus these messages, and take the
        # difference. That is exact as long as a message block does not depend on
        # what precedes it, which the prefix check below still enforces.
        #
        # What stays on record is the history as generated, thinking included --
        # those are the tokens the policy read back when it produced the next
        # turn, and a later training step has to see the same.
        s_anchor = render(_ANCHOR, gen_prompt_before)
        s_anchor_after = render(_ANCHOR + list(appended), gen_prompt_after)
        if not s_anchor_after.startswith(s_anchor):
            raise RuntimeError('Canonical chat_template output for messages_after is not a '
                               'prefix-extension of messages_before, and the same is true '
                               'of a one-message stand-in history; cannot compute the '
                               'delta. This indicates the template is non-monotonic in the '
                               'message list (e.g. reorders / rewrites earlier turns).\n'
                               f's_before tail: {s_before[-80:]!r}\n'
                               f's_after at same offset: '
                               f'{s_after[max(0, len(s_before) - 80):len(s_before) + 80]!r}')
        s_before, s_after = s_anchor, s_anchor_after
    return s_after[len(s_before):]


def encode_appended_turn(
    messages_before: List[Dict[str, Any]],
    message: Dict[str, Any],
    template: Template,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Tokens an assistant turn authored elsewhere contributes to the sequence.

    A sampler returns the ids it generated; an API returns text, whose tokens are
    only part of the turn -- the template also writes the turn terminator and
    whatever follows it. Diffing the rendered template recovers those without
    naming a single special token, so this holds for any chat template.

    The result is what :meth:`Template.concat_input_feature` expects as
    ``new_tokens``, and is token-for-token what :meth:`Template.encode` would
    have produced for the same conversation.
    """
    delta = _delta_text(
        template,
        messages_before, [template.decode_tool_calls(message)],
        gen_prompt_before=True,
        gen_prompt_after=False,
        tools=tools)
    if not delta:
        raise RuntimeError(f'Appending {message.get("role")!r} turn added no text; '
                           'the chat template dropped it entirely.')
    tokens = template.tokenizer.encode(delta, add_special_tokens=False)
    if not tokens:
        raise RuntimeError(f'Appended turn tokenised to an empty id list: {delta!r}')
    return tokens


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
    messages_before = list(pif.get('messages') or [])
    messages_after = messages_before + list(tool_messages)

    bridge_text = _delta_text(
        template, messages_before, tool_messages, gen_prompt_before=False, gen_prompt_after=True)
    if not bridge_text:
        raise RuntimeError('Bridge text computation returned empty string; '
                           'tool turn would add no tokens (template misconfiguration?).')

    bridge_ids = template.tokenizer.encode(bridge_text, add_special_tokens=False)
    if not bridge_ids:
        raise RuntimeError(f'Bridge text tokenised to empty id list: {bridge_text!r}')

    new_pif = append_ids(pif, bridge_ids, template, trainable=False)
    if new_pif is None:
        # Trajectory exceeds max_length and strategy is 'delete'
        return None
    new_pif['messages'] = messages_after
    return new_pif


def append_ids(
    pif: Dict[str, Any],
    ids: List[int],
    template: Template,
    *,
    trainable: bool,
) -> Optional[Dict[str, Any]]:
    """Grow the sequence by ``ids``; ``trainable`` says whose tokens they are.

    Mirrors the unroll-append-reroll pattern of
    :meth:`Template.concat_input_feature` so that ``labels`` and
    ``completion_mask`` semantics stay consistent with the sampler-produced
    pif.

    ``trainable=False`` is an observation -- a tool result, a bridge to the next
    generation prompt, anything the environment put in front of the model. It is
    nobody's completion, neither scored nor log-prob-bearing, so labels are
    ``-100`` and the mask is 0.

    ``trainable=True`` is the policy's own continuation, and each id is its own
    label: in input order position ``i`` is trained to produce ``input_ids[i]``,
    which the post pipeline then shifts. Callers pass ids the sampler emitted --
    never ids re-encoded from text, which is the drift this module exists to
    avoid.

    Shallow copy is deliberately used: every mutation below is a
    top-level key reassignment, never an in-place change to nested
    tensors. Multimodal payloads (``images``, ``pixel_values``,
    ``image_grid_thw`` ...) are shared by reference so we avoid
    re-copying image buffers every turn.
    """
    result = dict(pif)

    input_ids = list(result.get('input_ids') or [])
    labels = list(result.get('labels') or [])
    # labels arrive in output/shifted order (post _roll_labels). Unroll by
    # one position (shift right by 1) to get back to input order.
    if labels:
        if len(labels) != len(input_ids):
            raise RuntimeError(f'labels length ({len(labels)}) != input_ids length '
                               f'({len(input_ids)}); cannot safely append tokens.')
        labels = labels[-1:] + labels[:-1]
    else:
        labels = [-100] * len(input_ids)
    # Written back before the mask is read off it, so an empty account -- the
    # first call of an externally driven episode -- is a valid starting point.
    result['input_ids'] = input_ids
    completion_mask = template._prefix_completion_mask(result, labels)

    input_ids = input_ids + list(ids)
    labels = labels + (list(ids) if trainable else [-100] * len(ids))
    completion_mask = completion_mask + [1 if trainable else 0] * len(ids)

    result['input_ids'] = input_ids
    result['labels'] = labels
    result['completion_mask'] = completion_mask

    if 'mm_token_type_ids' in result:
        import torch
        mm = result['mm_token_type_ids']
        if not isinstance(mm, torch.Tensor):
            mm = torch.as_tensor(mm)
        # Pad along the last (sequence) dim — handles 1D [T] and 2D [1, T] uniformly.
        leading_shape = mm.shape[:-1]
        pad = torch.zeros((*leading_shape, len(ids)), dtype=mm.dtype, device=mm.device)
        result['mm_token_type_ids'] = torch.cat([mm, pad], dim=-1)

    # Replay the post pipeline: refresh attention_mask / position_ids /
    # length and re-roll labels back into output/shifted order.
    refreshed_list = template._invoke_post_pipeline([result])
    if not refreshed_list:
        # truncation_strategy='delete': trajectory exceeds max_length
        return None
    result.update(refreshed_list[0])
    return _to_plain(result)
