# Copyright (c) ModelScope Contributors. All rights reserved.
"""Growing a token sequence one turn at a time, in template space.

``self``-free functions that extend a running ``InputFeature`` (``pif``), all
measuring what a turn adds through the template's own ``render_delta`` rather
than by pasting special tokens together -- so the same renderer that encoded the
opening prompt also renders every later turn:

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

from typing import Any, Optional

import numpy as np

from twinkle.template.contract import Continuation, append_and_finalize


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


def encode_appended_turn(
    messages_before: list[dict[str, Any]],
    message: dict[str, Any],
    template: Continuation,
    tools: list[dict[str, Any]] | None = None,
) -> list[int]:
    """Tokens an assistant turn authored elsewhere contributes to the sequence.

    A sampler returns the ids it generated; an API returns text, whose tokens are
    only part of the turn -- the template also writes the turn terminator and
    whatever follows it. ``render_delta`` recovers those without naming a single
    special token.

    The result is what :meth:`Template.append_completion` expects as
    ``new_tokens``, and is token-for-token what :meth:`Template.encode` would
    have produced for the same conversation.
    """
    tokens = template.render_delta(
        messages_before, [message], gen_prompt_before=True, gen_prompt_after=False, tools=tools
    )
    if not tokens:
        raise RuntimeError(
            f"Appending {message.get('role')!r} turn added no tokens; the chat template dropped it entirely."
        )
    return tokens


def extend_with_bridge(
    pif: dict[str, Any],
    tool_messages: list[dict[str, Any]],
    template: Continuation,
) -> dict[str, Any] | None:
    """Append tool messages and the next generation prompt as -100 bridge.

    The bridge is computed ENTIRELY in template space by ``render_delta`` over
    ``messages_before`` and ``messages_before + tool_messages``, with the tools the
    opening prompt was rendered with.

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
    messages_before = list(pif.get("messages") or [])
    messages_after = messages_before + list(tool_messages)

    bridge_ids = template.render_delta(
        messages_before, list(tool_messages), gen_prompt_before=False, gen_prompt_after=True, tools=pif.get("tools")
    )
    if not bridge_ids:
        raise RuntimeError(
            "Bridge computation returned no tokens; tool turn would add nothing (template misconfiguration?)."
        )

    new_pif = append_ids(pif, bridge_ids, template, trainable=False)
    if new_pif is None:
        # Trajectory exceeds max_length and strategy is 'delete'
        return None
    new_pif["messages"] = messages_after
    return new_pif


def append_ids(
    pif: dict[str, Any],
    ids: list[int],
    template: Continuation,
    *,
    trainable: bool,
) -> dict[str, Any] | None:
    """Grow the sequence by ``ids``; ``trainable`` says whose tokens they are.

    Mirrors the unroll-append-reroll pattern of
    :meth:`Template.append_completion` so that ``labels`` and
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
    labels = pif.get("labels")
    if labels is not None and len(labels) != len(pif.get("input_ids") or []):
        raise RuntimeError(
            f"labels length ({len(labels)}) != input_ids length "
            f"({len(pif.get('input_ids') or [])}); cannot safely append tokens."
        )
    result = append_and_finalize(
        pif,
        ids,
        scored=trainable,
        sampled=trainable,
        finalize=template.finalize_appended_features,
    )
    return None if result is None else _to_plain(result)
