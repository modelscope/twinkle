# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for :func:`twinkle_agentic.rollout.bridge.extend_with_bridge`.

These tests target the pure, ``self``-free bridge-stitching function directly
(rather than through ``MultiTurnRollout``). They exercise:

    - normal bridge append: tool messages + next generation prompt are
      appended as ``-100`` bridge tokens, history is preserved verbatim, and
      ``messages`` is updated to ``messages_before + tool_messages``.
    - RuntimeError when the template's ``s_after`` rendering is NOT a
      prefix-extension of ``s_before`` (non-monotonic template).
    - RuntimeError when the computed bridge text is empty (template adds no
      tokens for the tool turn).
    - RuntimeError when ``labels`` length != ``input_ids`` length in the pif.

The fakes mirror the char-level FakeTokenizer / FakeTemplate infrastructure in
``test_multi_turn_rollout.py``: ``decode(encode(s)) == s`` for any mix of raw
chars and registered specials, and ``_invoke_post_pipeline`` replays the
label-roll / attention-mask semantics the real Template applies.
"""
from __future__ import annotations

from typing import Any

import pytest

from twinkle_agentic.rollout.bridge import extend_with_bridge


# =============================================================================
# Fakes (mirrors test_multi_turn_rollout.py)
# =============================================================================
class FakeTokenizer:
    """Char-level tokenizer with atomic special tokens.

    Guarantees ``decode(encode(s)) == s`` for any mix of raw chars and
    registered specials, which keeps the template-space bridge diff exact.
    """
    SPECIALS = ('<|im_start|>', '<|im_end|>')

    def __init__(self) -> None:
        self._s2i: dict[str, int] = {}
        self._i2s: dict[int, str] = {}
        for s in self.SPECIALS:
            self._add(s)

    def _add(self, tok: str) -> int:
        if tok not in self._s2i:
            i = len(self._s2i)
            self._s2i[tok] = i
            self._i2s[i] = tok
        return self._s2i[tok]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for sp in self.SPECIALS:
                if text.startswith(sp, i):
                    ids.append(self._add(sp))
                    i += len(sp)
                    matched = True
                    break
            if not matched:
                ids.append(self._add(text[i]))
                i += 1
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        specials = set(self.SPECIALS)
        toks = [self._i2s[int(i)] for i in ids]
        if skip_special_tokens:
            toks = [t for t in toks if t not in specials]
        return ''.join(toks)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **_,
    ):
        s = ''
        for m in messages:
            role = m['role']
            content = m['content']
            s += f'<|im_start|>{role}\n{content}<|im_end|>\n'
        if add_generation_prompt:
            s += '<|im_start|>assistant\n'
        if tokenize:
            return self.encode(s)
        return s


class NonMonotonicTokenizer(FakeTokenizer):
    """Renders a length-prefix that changes with message count.

    Because the leading ``[<count>]`` marker differs between ``messages_before``
    and ``messages_after``, ``s_after`` is NOT a prefix-extension of
    ``s_before`` — this is the non-monotonic template case that
    ``extend_with_bridge`` must reject.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        s = f'[{len(messages)}]'
        s += super().apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        if tokenize:
            return self.encode(s)
        return s


class EmptyBridgeTokenizer(FakeTokenizer):
    """Always renders the same constant string regardless of messages.

    ``s_after == s_before`` ⇒ the computed bridge text is empty, which
    ``extend_with_bridge`` must reject.
    """

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **_):
        s = 'CONSTANT'
        if tokenize:
            return self.encode(s)
        return s


class FakeTemplate:
    """Minimal Template mirroring the parts ``extend_with_bridge`` touches."""
    model_id = 'qwen-fake'
    truncation_strategy = 'right'

    def __init__(self, tokenizer: FakeTokenizer) -> None:
        self.tokenizer = tokenizer

    def encode(self, trajectory: dict[str, Any], add_generation_prompt: bool = False) -> dict[str, Any]:
        messages = trajectory.get('messages', [])
        s = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        input_ids = self.tokenizer.encode(s, add_special_tokens=False)
        pif: dict[str, Any] = dict(trajectory)
        pif['input_ids'] = input_ids
        pif['labels'] = [-100] * len(input_ids)
        return self._invoke_post_pipeline([pif])[0]

    def _invoke_post_pipeline(self, inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for pif in inputs:
            pif = dict(pif)
            input_ids = list(pif['input_ids'])
            labels = list(pif.get('labels') or [])
            if labels:
                if len(labels) != len(input_ids):
                    raise RuntimeError(f'FakeTemplate post_pipeline: labels({len(labels)}) '
                                       f'!= input_ids({len(input_ids)})')
                # np.roll(labels, -1): shift LEFT by 1 (output/shifted order)
                labels = labels[1:] + labels[:1]
            pif['input_ids'] = input_ids
            pif['labels'] = labels
            pif['attention_mask'] = [1] * len(input_ids)
            pif['position_ids'] = list(range(len(input_ids)))
            pif['length'] = len(input_ids)
            out.append(pif)
        return out


# =============================================================================
# Helpers
# =============================================================================
def _make_pif(template: FakeTemplate, messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a post-pipeline pif for ``messages`` in inference mode."""
    return template.encode({'messages': list(messages)}, add_generation_prompt=True)


def _count_trainable(labels: list[int]) -> int:
    return sum(1 for label in labels if label != -100)


# =============================================================================
# Tests
# =============================================================================
def test_normal_bridge_append():
    """Tool messages + generation prompt are appended as -100 bridge tokens.

    The pre-existing history is preserved verbatim (prefix of the new
    ``input_ids``), the appended positions are all masked (-100), and
    ``messages`` is updated to ``messages_before + tool_messages``.
    """
    tokenizer = FakeTokenizer()
    template = FakeTemplate(tokenizer)

    messages = [{'role': 'user', 'content': 'What is the weather?'}]
    pif = _make_pif(template, messages)
    before_ids = list(pif['input_ids'])
    before_trainable = _count_trainable(pif['labels'])

    tool_messages = [{'role': 'tool', 'content': 'sunny'}]
    new_pif = extend_with_bridge(pif, tool_messages, template)

    assert new_pif is not None
    # History preserved verbatim as a prefix.
    assert new_pif['input_ids'][:len(before_ids)] == before_ids
    # Bridge actually added tokens.
    assert len(new_pif['input_ids']) > len(before_ids)
    # All newly appended positions are masked (-100 → not trainable).
    assert _count_trainable(new_pif['labels']) == before_trainable
    # input_ids / labels stay aligned.
    assert len(new_pif['input_ids']) == len(new_pif['labels'])
    # messages updated to before + tool.
    assert new_pif['messages'] == messages + tool_messages

    # The bridge delta equals the template-space difference exactly.
    s_before = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    s_after = tokenizer.apply_chat_template(
        messages + tool_messages, tokenize=False, add_generation_prompt=True)
    expected_bridge_ids = tokenizer.encode(s_after[len(s_before):], add_special_tokens=False)
    assert new_pif['input_ids'][len(before_ids):] == expected_bridge_ids


def test_non_prefix_extension_raises():
    """``s_after`` not a prefix-extension of ``s_before`` → RuntimeError."""
    tokenizer = NonMonotonicTokenizer()
    template = FakeTemplate(tokenizer)

    messages = [{'role': 'user', 'content': 'hello'}]
    pif = _make_pif(template, messages)

    with pytest.raises(RuntimeError, match='prefix-extension'):
        extend_with_bridge(pif, [{'role': 'tool', 'content': 'x'}], template)


def test_empty_bridge_text_raises():
    """Bridge text computing to empty string → RuntimeError."""
    tokenizer = EmptyBridgeTokenizer()
    template = FakeTemplate(tokenizer)

    messages = [{'role': 'user', 'content': 'hello'}]
    pif = _make_pif(template, messages)

    with pytest.raises(RuntimeError, match='empty string'):
        extend_with_bridge(pif, [{'role': 'tool', 'content': 'x'}], template)


def test_labels_length_mismatch_raises():
    """``labels`` length != ``input_ids`` length in the pif → RuntimeError."""
    tokenizer = FakeTokenizer()
    template = FakeTemplate(tokenizer)

    messages = [{'role': 'user', 'content': 'hello'}]
    pif = _make_pif(template, messages)
    # Corrupt the pif: drop one label so lengths disagree.
    pif['labels'] = list(pif['labels'])[:-1]

    with pytest.raises(RuntimeError, match='labels length'):
        extend_with_bridge(pif, [{'role': 'tool', 'content': 'x'}], template)
