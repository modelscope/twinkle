# Copyright (c) ModelScope Contributors. All rights reserved.
"""Keyword-free structural noise-turn tagging (AUDIT D5, optional).

Existing heartbeat stripping (``message_normalizer``) is keyword-based and, per
the audit, already covers the common OpenHands/OpenClaw formats. This optional
tagger catches *keyword-free* structural noise: near-identical, very short turns
that repeat across the trajectory (polling / retries with no new signal), using
only cheap structural signals (length + exact repetition) — no embeddings, no
LLM. It **tags** a per-trajectory noise ratio into ``user_data`` (never drops),
so a downstream filter can act on it if desired.

The embedding-distance variant sketched in the audit is deferred until the D1
near-dup infrastructure (which provides the embedding index) exists.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from twinkle.preprocessor import Mapper

from . import label_schema as L
from .message_utils import msg_content_text, normalize_tool_calls

KEY_NOISE_RATIO = 'structural_noise_ratio'


class StructuralNoiseTagger(Mapper):
    """Tag the fraction of assistant turns that are short, repeated boilerplate.

    Args:
        short_chars: an assistant turn with visible text at/under this length is a
            noise candidate (tool-call turns are exempt — they carry structure).
        min_repeat: a candidate counts as noise only if its normalized text recurs
            at least this many times across the trajectory's assistant turns.
    """

    def __init__(self, *, short_chars: int = 40, min_repeat: int = 3):
        self.short_chars = int(short_chars)
        self.min_repeat = int(min_repeat)

    def map(self, row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row.get('messages')
        if not isinstance(messages, list) or not messages:
            return row
        asst = [m for m in messages if isinstance(m, dict) and m.get('role') == 'assistant']
        if not asst:
            return row
        texts = []
        for m in asst:
            if normalize_tool_calls(m) is not None:
                texts.append(None)  # tool-call turn: never noise
            else:
                texts.append(msg_content_text(m).strip())
        counts = Counter(t for t in texts if t)
        noise = 0
        for t in texts:
            if t and len(t) <= self.short_chars and counts[t] >= self.min_repeat:
                noise += 1
        ratio = noise / len(asst)
        return L.set_label(row, KEY_NOISE_RATIO, round(ratio, 6))
