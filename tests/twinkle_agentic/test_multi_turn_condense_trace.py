# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for :class:`MultiTurnCondenseRollout` trace augmentation.

The subclass extends the base trace record with a ``blocks`` field:
``{'block_N': {'original': raw_text_or_None, 'compressed': post_text}}``.
Having both sides of the mapping in the dumped JSON means the trace
alone is enough to audit compression quality.
"""
from __future__ import annotations

from typing import Any, Dict, List

from twinkle_agentic.data_format import Chunks
from twinkle_agentic.rollout.multi_turn_condense import MultiTurnCondenseRollout


def _chunks(specs: list[dict[str, Any]]) -> Chunks:
    out = []
    for s in specs:
        raw: dict[str, Any] = {'condensed': bool(s.get('condensed', True))}
        if s.get('original') is not None:
            raw['original'] = s['original']
        out.append({
            'type': s.get('type', 'text'),
            'role': s.get('role', 'user'),
            'content': s['content'],
            'raw': raw,
        })
    return Chunks(chunks=out)


class _Stub(MultiTurnCondenseRollout):
    """Bypass ``__init__`` to exercise only ``_build_trace_record``."""

    def __init__(self, block_chunks):  # noqa: D401 -- minimal stub
        self._trace_block_chunks = block_chunks


def test_build_trace_record_pairs_original_and_compressed():
    chunks = _chunks([
        {
            'content': 'short A',
            'original': 'long raw passage A ...'
        },
        {
            'content': 'short B',
            'original': 'long raw passage B ...'
        },
    ])
    rollout = _Stub(block_chunks=[chunks])
    traj = {'messages': [], 'stop_reason': 'stop', 'truncated': False}

    record = rollout._build_trace_record(traj, idx=0, success=False)

    assert record['blocks'] == {
        'block_1': {
            'original': 'long raw passage A ...',
            'compressed': 'short A',
        },
        'block_2': {
            'original': 'long raw passage B ...',
            'compressed': 'short B',
        },
    }
    # Base fields still intact.
    assert record['stop_reason'] == 'stop'


def test_build_trace_record_preserves_missing_snapshot_as_none():
    """Compressed content is always kept even when ``raw.original`` is None."""
    chunks = _chunks([{'content': 'short A', 'original': None}])
    rollout = _Stub(block_chunks=[chunks])
    record = rollout._build_trace_record({'messages': []}, idx=0, success=False)
    assert record['blocks'] == {
        'block_1': {
            'original': None,
            'compressed': 'short A'
        },
    }


def test_build_trace_record_skips_non_condensed_and_tool_chunks():
    """Numbering only counts condensed, non-tool, non-empty text chunks."""
    chunks = Chunks(chunks=[
        # skipped: not condensed
        {
            'type': 'text',
            'role': 'user',
            'content': 'plain',
            'raw': {}
        },
        # counted: condensed user text
        {
            'type': 'text',
            'role': 'user',
            'content': 'cA',
            'raw': {
                'condensed': True,
                'original': 'rawA'
            }
        },
        # skipped: tool role
        {
            'type': 'text',
            'role': 'tool',
            'content': 'toolmsg',
            'raw': {
                'condensed': True,
                'original': 'xxx'
            }
        },
        # counted: condensed assistant text
        {
            'type': 'text',
            'role': 'assistant',
            'content': 'cB',
            'raw': {
                'condensed': True,
                'original': 'rawB'
            }
        },
    ])
    rollout = _Stub(block_chunks=[chunks])
    record = rollout._build_trace_record({'messages': []}, idx=0, success=False)
    assert list(record['blocks']) == ['block_1', 'block_2']
    assert record['blocks']['block_1']['original'] == 'rawA'
    assert record['blocks']['block_2']['original'] == 'rawB'


def test_build_trace_record_is_noop_when_stash_missing():
    rollout = _Stub(block_chunks=None)
    record = rollout._build_trace_record({'messages': []}, idx=0, success=False)
    assert 'blocks' not in record
