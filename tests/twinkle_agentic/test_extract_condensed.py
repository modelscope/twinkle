# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for :class:`twinkle_agentic.tools.extract_condensed.ExtractCondensed`.

Covers:
- block-index enumeration matches :meth:`Chunks.to_trajectory` exactly
- retrieval returns pre-compression text when ``raw.original`` is present
- fallback to current ``content`` when ``raw.original`` missing
- bad / missing arguments produce actionable error strings (no exceptions)
- tool metadata is complete and JSON-serializable
- integration with :class:`ToolManager`
- end-to-end: KeywordCondenser → Chunks → ExtractCondensed round-trips
"""
from __future__ import annotations

import json
import pytest

from twinkle_agentic.data_format import Chunks
from twinkle_agentic.tools.extract_condensed import TOOL_NAME, ExtractCondensed
from twinkle_agentic.tools.tool_manager import ToolManager


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _condensed(content, *, original=None, role='user', round_idx=1):
    raw = {'condensed': True}
    if original is not None:
        raw['original'] = original
    ch = {'type': 'text', 'role': role, 'content': content, 'raw': raw, 'round': round_idx}
    return ch


def _plain(content, *, role='user'):
    return {'type': 'text', 'role': role, 'content': content}


# ---------------------------------------------------------------------------
# block enumeration parity with Chunks.to_trajectory
# ---------------------------------------------------------------------------
def test_blocks_indexed_from_1_in_document_order():
    chunks = Chunks(chunks=[
        _condensed('cmp1', original='orig one'),
        _condensed('cmp2', original='orig two'),
        _condensed('cmp3', original='orig three'),
    ])
    tool = ExtractCondensed(chunks)
    assert tool.blocks == [1, 2, 3]
    assert len(tool) == 3
    assert 1 in tool and 3 in tool and 4 not in tool


def test_non_condensed_text_chunks_are_not_indexed():
    chunks = Chunks(chunks=[
        _plain('system prelude', role='system'),  # not condensed
        _condensed('cmp1', original='orig one'),
        _plain('user follow-up'),  # not condensed
        _condensed('cmp2', original='orig two'),
    ])
    tool = ExtractCondensed(chunks)
    assert tool.blocks == [1, 2]
    assert tool(TOOL_NAME, {'block': 1}) == 'orig one'
    assert tool(TOOL_NAME, {'block': 2}) == 'orig two'


def test_tool_role_condensed_chunks_are_skipped():
    # Mirrors Chunks.to_trajectory: role=='tool' is NEVER wrapped, even
    # if marked condensed, so it must not consume a block index either.
    chunks = Chunks(chunks=[
        _condensed('cmp_user', original='user orig', role='user'),
        _condensed('cmp_tool', original='tool orig', role='tool'),
        _condensed('cmp_asst', original='asst orig', role='assistant'),
    ])
    tool = ExtractCondensed(chunks)
    # Only the user + assistant blocks count.
    assert tool.blocks == [1, 2]
    assert tool(TOOL_NAME, {'block': 1}) == 'user orig'
    assert tool(TOOL_NAME, {'block': 2}) == 'asst orig'


def test_empty_content_condensed_chunks_are_skipped():
    chunks = Chunks(chunks=[
        _condensed('', original=''),  # empty, skipped
        _condensed('cmp', original='orig'),
    ])
    tool = ExtractCondensed(chunks)
    assert tool.blocks == [1]
    assert tool(TOOL_NAME, {'block': 1}) == 'orig'


def test_non_text_chunks_ignored():
    chunks = Chunks(chunks=[
        {
            'type': 'image',
            'content': 'image bytes',
            'raw': {
                'type': 'image',
                'image': 'x'
            },
            'role': 'user'
        },
        _condensed('cmp', original='orig text'),
    ])
    tool = ExtractCondensed(chunks)
    assert tool.blocks == [1]
    assert tool(TOOL_NAME, {'block': 1}) == 'orig text'


# ---------------------------------------------------------------------------
# retrieval semantics
# ---------------------------------------------------------------------------
def test_returns_original_when_present():
    chunks = Chunks(chunks=[_condensed('CMP', original='THE ORIGINAL')])
    tool = ExtractCondensed(chunks)
    assert tool(TOOL_NAME, {'block': 1}) == 'THE ORIGINAL'


def test_missing_original_returns_error_not_compressed_content():
    # Contract: ExtractCondensed returns the *original* text. When the
    # upstream pipeline forgot to snapshot it, the tool MUST fail loud
    # rather than silently handing back the compressed stand-in, which
    # would deceive the LLM into thinking it had recovered the source.
    chunks = Chunks(chunks=[_condensed('CMP', original=None)])
    tool = ExtractCondensed(chunks)
    # The block is still enumerated so numbering stays aligned.
    assert tool.blocks == [1]
    out = tool(TOOL_NAME, {'block': 1})
    assert out.startswith('Error:')
    assert 'no original-text snapshot' in out
    # And crucially, the compressed stand-in is NOT leaked.
    assert 'CMP' not in out


def test_original_empty_string_also_reports_missing_snapshot():
    chunks = Chunks(chunks=[_condensed('CMP', original='')])
    tool = ExtractCondensed(chunks)
    out = tool(TOOL_NAME, {'block': 1})
    assert out.startswith('Error:')
    assert 'no original-text snapshot' in out


# ---------------------------------------------------------------------------
# bad input handling (never raises)
# ---------------------------------------------------------------------------
def test_missing_block_argument_returns_error_string():
    tool = ExtractCondensed(Chunks(chunks=[_condensed('cmp', original='orig')]))
    out = tool(TOOL_NAME, {})
    assert out.startswith('Error: missing required argument')


def test_non_integer_block_returns_error_string():
    tool = ExtractCondensed(Chunks(chunks=[_condensed('cmp', original='orig')]))
    for bad in ('abc', [], {}, None):
        out = tool(TOOL_NAME, {'block': bad})
        assert out.startswith('Error:'), (bad, out)


def test_bool_block_is_rejected_not_coerced_to_int():
    # ``bool`` is a subclass of ``int`` so ``int(True) == 1``. Without
    # an explicit guard, ``{'block': True}`` would silently retrieve
    # block 1 -- a nasty footgun if an LLM stringifies a truthy flag.
    tool = ExtractCondensed(Chunks(chunks=[_condensed('cmp', original='orig1')]))
    out_true = tool(TOOL_NAME, {'block': True})
    assert out_true.startswith('Error:') and 'bool' in out_true
    out_false = tool(TOOL_NAME, {'block': False})
    assert out_false.startswith('Error:') and 'bool' in out_false
    # Sanity: the real integer 1 still works.
    assert tool(TOOL_NAME, {'block': 1}) == 'orig1'


def test_float_block_is_rejected_not_silently_truncated():
    # ``int(1.9) == 1`` would silently round a float down; reject it.
    tool = ExtractCondensed(Chunks(chunks=[_condensed('cmp', original='orig1')]))
    out = tool(TOOL_NAME, {'block': 1.9})
    assert out.startswith('Error:') and 'float' in out
    # And floats that happen to be integer-valued are also rejected to
    # keep the contract simple.
    out2 = tool(TOOL_NAME, {'block': 1.0})
    assert out2.startswith('Error:')


def test_non_dict_arguments_returns_error_not_attribute_error():
    tool = ExtractCondensed(Chunks(chunks=[_condensed('cmp', original='orig')]))
    # Bypass ToolManager and feed a non-dict directly; must not raise.
    out = tool(TOOL_NAME, 'not a dict')  # type: ignore[arg-type]
    assert out.startswith('Error:')


def test_out_of_range_block_returns_short_range_error():
    # Short existence error -- we must NOT enumerate every valid id, or
    # a hallucinated ``blocks=[1..200]`` storm would multiply the error
    # into thousands of tokens in the non-trainable bridge.
    tool = ExtractCondensed(
        Chunks(chunks=[
            _condensed('cmp1', original='orig1'),
            _condensed('cmp2', original='orig2'),
        ]))
    out = tool(TOOL_NAME, {'block': 99})
    assert out.startswith('Error:')
    assert 'block 99 not found' in out
    assert '1..2' in out
    # Defensive: the verbose legacy listing must not leak back.
    assert 'Available blocks: 1, 2' not in out


def test_empty_tool_reports_no_blocks_available():
    tool = ExtractCondensed(Chunks(chunks=[_plain('nothing condensed')]))
    out = tool(TOOL_NAME, {'block': 1})
    assert out.startswith('Error:')
    assert 'no blocks available' in out


def test_integer_strings_are_accepted():
    tool = ExtractCondensed(Chunks(chunks=[_condensed('cmp', original='orig')]))
    assert tool(TOOL_NAME, {'block': '1'}) == 'orig'


# ---------------------------------------------------------------------------
# single-block-per-call contract + trajectory-bound idempotency
#
# Lists were previously accepted; they are now rejected so a hallucinated
# ``blocks=[1..200]`` cannot flood the non-trainable bridge. Re-requesting
# the same block returns a short "already expanded" reply instead of the
# raw text (which is already sitting in an earlier tool message).
# ---------------------------------------------------------------------------
def test_blocks_int_equivalent_to_legacy_block_arg():
    # Passing ``{'blocks': N}`` (single int under the new name) must
    # behave identically to the legacy ``{'block': N}`` path: bare text,
    # no <block_N> wrapper.
    tool = ExtractCondensed(Chunks(chunks=[_condensed('cmp1', original='orig one')]))
    assert tool(TOOL_NAME, {'blocks': 1}) == 'orig one'
    # Re-create the tool so the second call is not deduped against the
    # first (which is covered separately below).
    tool2 = ExtractCondensed(Chunks(chunks=[_condensed('cmp1', original='orig one')]))
    assert tool2(TOOL_NAME, {'block': 1}) == 'orig one'


def test_blocks_list_is_rejected_with_short_error():
    # Single-block-per-call contract: the only way a list reaches this
    # path is if the policy hallucinated a bulk id enumeration, which is
    # exactly what we want to stop. Reject loudly with a brief message.
    tool = ExtractCondensed(
        Chunks(chunks=[
            _condensed('c1', original='a'),
            _condensed('c2', original='b'),
            _condensed('c3', original='c'),
        ]))
    for bad in ([1, 2, 3], (1, 2), [1], []):
        out = tool(TOOL_NAME, {'blocks': bad})
        assert out.startswith('Error:'), (bad, out)
        assert 'single integer' in out or 'one block' in out, (bad, out)


def test_second_call_on_same_block_returns_already_expanded_notice():
    # Trajectory-bound idempotency. The raw text has already been handed
    # to the model as a prior tool response, so returning it again only
    # doubles the non-trainable footprint. The second call gets a short
    # notice instead -- no "Error:" prefix (it's not a failure) and
    # crucially the raw text must NOT be repeated.
    tool = ExtractCondensed(
        Chunks(chunks=[
            _condensed('cmp1', original='ORIGINAL TEXT FOR ONE'),
            _condensed('cmp2', original='ORIGINAL TEXT FOR TWO'),
        ]))
    first = tool(TOOL_NAME, {'block': 1})
    assert first == 'ORIGINAL TEXT FOR ONE'
    second = tool(TOOL_NAME, {'block': 1})
    assert 'already expanded' in second
    assert 'ORIGINAL TEXT FOR ONE' not in second
    # Dedup is per-id: a different block is still expandable once.
    third = tool(TOOL_NAME, {'block': 2})
    assert third == 'ORIGINAL TEXT FOR TWO'
    # And then that one also becomes deduped.
    fourth = tool(TOOL_NAME, {'block': 2})
    assert 'already expanded' in fourth


def test_already_expanded_is_trajectory_bound_fresh_instance_resets():
    # ``MultiTurnCondenseRollout`` builds a new ExtractCondensed per
    # trajectory, so a fresh instance must start with an empty dedup set
    # even if a sibling trajectory just expanded block 1.
    chunks = Chunks(chunks=[_condensed('c1', original='raw text')])
    t1 = ExtractCondensed(chunks)
    assert t1(TOOL_NAME, {'block': 1}) == 'raw text'
    assert 'already expanded' in t1(TOOL_NAME, {'block': 1})
    t2 = ExtractCondensed(chunks)  # independent trajectory
    assert t2(TOOL_NAME, {'block': 1}) == 'raw text'


def test_prefers_blocks_over_legacy_block_when_both_present():
    # Undefined which wins in theory; we declare ``blocks`` takes
    # precedence so callers can migrate incrementally.
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('c1', original='NEW'),
        _condensed('c2', original='LEGACY'),
    ]))
    out = tool(TOOL_NAME, {'blocks': 1, 'block': 2})
    assert out == 'NEW'


# ---------------------------------------------------------------------------
# tool_info metadata
# ---------------------------------------------------------------------------
def test_tool_info_shape_and_serializability():
    tool = ExtractCondensed(Chunks(chunks=[]))
    info = tool.tool_info()
    # OpenAI-shape: {type: 'function', function: {name, description, parameters}}
    assert info['type'] == 'function'
    fn = info['function']
    assert fn['name'] == TOOL_NAME == 'extract_condensed'
    assert 'description' in fn and fn['description']
    # parameters is a plain mapping (not a JSON string): the jinja chat
    # template consumes it directly.
    params = fn['parameters']
    assert isinstance(params, dict)
    # The whole info dict must still be JSON-serializable so it can be
    # embedded inside a trace / logged safely.
    json.dumps(info)
    # Preferred parameter name is ``blocks`` (single int per call; no list).
    assert 'blocks' in params
    assert 'int' in params['blocks']
    # The old ``int OR list[int]`` signature must be gone: no list-form
    # type annotation leaks through. (The sentence may still say the
    # phrase "lists are rejected", which is fine.)
    assert 'list[' not in params['blocks']
    assert 'OR list' not in params['blocks']


# ---------------------------------------------------------------------------
# ToolManager integration
# ---------------------------------------------------------------------------
def test_register_with_tool_manager_and_dispatch():
    tool = ExtractCondensed(
        Chunks(chunks=[
            _condensed('cmp1', original='orig one'),
            _condensed('cmp2', original='orig two'),
        ]))
    mgr = ToolManager({})
    mgr.register(tool)
    assert TOOL_NAME in mgr.names()

    # dict-form arguments
    out = mgr({'type': 'function', 'function': {'name': TOOL_NAME, 'arguments': {'block': 2}}})
    assert out == 'orig two'

    # JSON-string-form arguments (OpenAI-style)
    out = mgr({'type': 'function', 'function': {'name': TOOL_NAME, 'arguments': '{"block": 1}'}})
    assert out == 'orig one'


def test_manager_reports_error_on_unknown_block_without_raising():
    tool = ExtractCondensed(Chunks(chunks=[_condensed('cmp1', original='orig one')]))
    mgr = ToolManager({})
    mgr.register(tool)
    out = mgr({'type': 'function', 'function': {'name': TOOL_NAME, 'arguments': '{"block": 999}'}})
    assert out.startswith('Error:')


# ---------------------------------------------------------------------------
# end-to-end: round-trip with KeywordCondenser (uses raw.original)
# ---------------------------------------------------------------------------
_SPACY_OK = True
try:
    import spacy  # noqa: F401
    spacy.load('en_core_web_sm')
except Exception:
    _SPACY_OK = False

LONG_PASSAGE = ('Christopher Nolan was born on 30 July 1970 in London. '
                'He is a British-American film director, producer and screenwriter. '
                'His film Inception (2010) is a science-fiction heist movie. '
                'Inception grossed over 829 million dollars worldwide.')


@pytest.mark.skipif(not _SPACY_OK, reason='en_core_web_sm not available')
def test_end_to_end_with_keyword_condenser_returns_original():
    from twinkle_agentic.condenser.keyword import KeywordCondenser

    pre = Chunks(chunks=[{'type': 'text', 'role': 'user', 'content': LONG_PASSAGE}])
    post = KeywordCondenser(compression_ratio=4.0, min_chars=50)(pre)

    # The condenser should have left behind an ``original`` snapshot.
    assert post.chunks[0]['raw']['condensed'] is True
    assert post.chunks[0]['raw']['original'] == LONG_PASSAGE
    assert len(post.chunks[0]['content']) < len(LONG_PASSAGE)

    tool = ExtractCondensed(post)
    assert tool.blocks == [1]
    assert tool(TOOL_NAME, {'block': 1}) == LONG_PASSAGE


@pytest.mark.skipif(not _SPACY_OK, reason='en_core_web_sm not available')
def test_end_to_end_block_indices_match_to_trajectory_wrapping():
    from twinkle_agentic.condenser.keyword import KeywordCondenser

    pre = Chunks(chunks=[
        {
            'type': 'text',
            'role': 'user',
            'content': LONG_PASSAGE,
            'round': 1
        },
        {
            'type': 'text',
            'role': 'assistant',
            'content': LONG_PASSAGE + ' Assistant elaboration.',
            'round': 1
        },
    ])
    # skip_roles default excludes assistant → only first chunk condensed.
    post = KeywordCondenser(compression_ratio=4.0, min_chars=50)(pre)
    tool = ExtractCondensed(post)

    # Exactly one wrapped block.
    assert tool.blocks == [1]
    # The trajectory wrapper agrees: block_1 exists, block_2 does not.
    traj = post.to_trajectory()
    rendered = ''.join(m['content'] if isinstance(m.get('content'), str) else '' for m in traj['messages'])
    assert '<block_1>' in rendered and '</block_1>' in rendered
    assert '<block_2>' not in rendered
    # And the tool returns the correct original.
    assert tool(TOOL_NAME, {'block': 1}) == LONG_PASSAGE
