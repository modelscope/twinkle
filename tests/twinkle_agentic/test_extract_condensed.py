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
from twinkle_agentic.tools.extract_condensed import (
    TOOL_NAME, ExtractCondensed)
from twinkle_agentic.tools.tool_manager import ToolManager


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _condensed(content, *, original=None, role='user', round_idx=1):
    raw = {'condensed': True}
    if original is not None:
        raw['original'] = original
    ch = {'type': 'text', 'role': role, 'content': content, 'raw': raw,
          'round': round_idx}
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
        _plain('system prelude', role='system'),     # not condensed
        _condensed('cmp1', original='orig one'),
        _plain('user follow-up'),                    # not condensed
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
        _condensed('', original=''),            # empty, skipped
        _condensed('cmp', original='orig'),
    ])
    tool = ExtractCondensed(chunks)
    assert tool.blocks == [1]
    assert tool(TOOL_NAME, {'block': 1}) == 'orig'


def test_non_text_chunks_ignored():
    chunks = Chunks(chunks=[
        {'type': 'image', 'content': 'image bytes',
         'raw': {'type': 'image', 'image': 'x'}, 'role': 'user'},
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
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp', original='orig')]))
    out = tool(TOOL_NAME, {})
    assert out.startswith('Error: missing required argument')


def test_non_integer_block_returns_error_string():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp', original='orig')]))
    for bad in ('abc', [], {}, None):
        out = tool(TOOL_NAME, {'block': bad})
        assert out.startswith('Error:'), (bad, out)


def test_bool_block_is_rejected_not_coerced_to_int():
    # ``bool`` is a subclass of ``int`` so ``int(True) == 1``. Without
    # an explicit guard, ``{'block': True}`` would silently retrieve
    # block 1 -- a nasty footgun if an LLM stringifies a truthy flag.
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp', original='orig1')]))
    out_true = tool(TOOL_NAME, {'block': True})
    assert out_true.startswith('Error:') and 'bool' in out_true
    out_false = tool(TOOL_NAME, {'block': False})
    assert out_false.startswith('Error:') and 'bool' in out_false
    # Sanity: the real integer 1 still works.
    assert tool(TOOL_NAME, {'block': 1}) == 'orig1'


def test_float_block_is_rejected_not_silently_truncated():
    # ``int(1.9) == 1`` would silently round a float down; reject it.
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp', original='orig1')]))
    out = tool(TOOL_NAME, {'block': 1.9})
    assert out.startswith('Error:') and 'float' in out
    # And floats that happen to be integer-valued are also rejected to
    # keep the contract simple.
    out2 = tool(TOOL_NAME, {'block': 1.0})
    assert out2.startswith('Error:')


def test_non_dict_arguments_returns_error_not_attribute_error():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp', original='orig')]))
    # Bypass ToolManager and feed a non-dict directly; must not raise.
    out = tool(TOOL_NAME, 'not a dict')  # type: ignore[arg-type]
    assert out.startswith('Error:')


def test_out_of_range_block_returns_error_with_available_list():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp1', original='orig1'),
        _condensed('cmp2', original='orig2'),
    ]))
    out = tool(TOOL_NAME, {'block': 99})
    assert 'block 99 not found' in out
    assert 'Available blocks: 1, 2' in out


def test_empty_tool_reports_no_blocks_available():
    tool = ExtractCondensed(Chunks(chunks=[
        _plain('nothing condensed')]))
    out = tool(TOOL_NAME, {'block': 1})
    assert 'Available blocks: (none)' in out


def test_integer_strings_are_accepted():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp', original='orig')]))
    assert tool(TOOL_NAME, {'block': '1'}) == 'orig'


# ---------------------------------------------------------------------------
# multi-block expansion (``blocks`` accepts int OR list[int])
# ---------------------------------------------------------------------------
def test_blocks_int_equivalent_to_legacy_block_arg():
    # Passing ``{'blocks': N}`` (single int under the new name) must
    # behave identically to the legacy ``{'block': N}`` path: bare text,
    # no <block_N> wrapper.
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp1', original='orig one')]))
    assert tool(TOOL_NAME, {'blocks': 1}) == 'orig one'
    assert tool(TOOL_NAME, {'blocks': 1}) == tool(TOOL_NAME, {'block': 1})


def test_blocks_list_wraps_each_result_in_block_tags():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp1', original='orig one'),
        _condensed('cmp2', original='orig two'),
        _condensed('cmp3', original='orig three'),
    ]))
    out = tool(TOOL_NAME, {'blocks': [1, 3]})
    # Both blocks present, each wrapped, separated by a blank line.
    assert '<block_1>\norig one\n</block_1>' in out
    assert '<block_3>\norig three\n</block_3>' in out
    assert '<block_2>' not in out
    # Order respects input order.
    assert out.index('<block_1>') < out.index('<block_3>')


def test_blocks_list_preserves_order_over_sorting():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('c1', original='a'),
        _condensed('c2', original='b'),
        _condensed('c3', original='c'),
    ]))
    out = tool(TOOL_NAME, {'blocks': [3, 1, 2]})
    # Output order must follow the caller's order, not numeric order.
    assert out.index('<block_3>') < out.index('<block_1>') < out.index('<block_2>')


def test_blocks_list_deduplicates_preserving_first_occurrence():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('c1', original='a'),
        _condensed('c2', original='b'),
    ]))
    out = tool(TOOL_NAME, {'blocks': [1, 2, 1, 2, 1]})
    # Each block appears exactly once.
    assert out.count('<block_1>') == 1
    assert out.count('<block_2>') == 1
    # And the first occurrence pins the order.
    assert out.index('<block_1>') < out.index('<block_2>')


def test_blocks_list_with_single_element_still_wraps():
    # Explicit list form is a commitment to multi-block semantics even
    # if only one element is present -- wrap it so the caller (or
    # downstream sanitizer) can treat list-form results uniformly.
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('c1', original='orig a')]))
    out = tool(TOOL_NAME, {'blocks': [1]})
    assert out == '<block_1>\norig a\n</block_1>'


def test_blocks_list_string_integers_accepted():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('c1', original='a'),
        _condensed('c2', original='b'),
    ]))
    out = tool(TOOL_NAME, {'blocks': ['1', '2']})
    assert '<block_1>\na\n</block_1>' in out
    assert '<block_2>\nb\n</block_2>' in out


def test_blocks_list_rejects_bool_and_float_per_element():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('c1', original='a'),
        _condensed('c2', original='b'),
    ]))
    out_bool = tool(TOOL_NAME, {'blocks': [1, True]})
    assert out_bool.startswith('Error:') and 'bool' in out_bool
    out_float = tool(TOOL_NAME, {'blocks': [1, 2.5]})
    assert out_float.startswith('Error:') and 'float' in out_float


def test_blocks_list_missing_blocks_embed_error_inline():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('c1', original='orig one')]))
    out = tool(TOOL_NAME, {'blocks': [1, 99]})
    # Valid block returns its content; missing one returns an error
    # string inside its own <block_99> wrapper so the caller can tell
    # which one failed without the tool itself raising.
    assert '<block_1>\norig one\n</block_1>' in out
    assert '<block_99>' in out
    assert 'block 99 not found' in out


def test_blocks_empty_list_returns_error():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('c1', original='a')]))
    out = tool(TOOL_NAME, {'blocks': []})
    assert out.startswith('Error:')
    assert 'at least one block number' in out


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
    assert info['tool_name'] == TOOL_NAME == 'extract_condensed'
    assert 'description' in info and info['description']
    # parameters must be a JSON string that loads back cleanly.
    params = json.loads(info['parameters'])
    # Preferred parameter name is ``blocks`` (supports int OR list[int]).
    assert 'blocks' in params
    assert 'int' in params['blocks'] and 'list' in params['blocks']


# ---------------------------------------------------------------------------
# ToolManager integration
# ---------------------------------------------------------------------------
def test_register_with_tool_manager_and_dispatch():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp1', original='orig one'),
        _condensed('cmp2', original='orig two'),
    ]))
    mgr = ToolManager({})
    mgr.register(tool)
    assert TOOL_NAME in mgr.names()

    # dict-form arguments
    out = mgr({'tool_name': TOOL_NAME, 'arguments': {'block': 2}})
    assert out == 'orig two'

    # JSON-string-form arguments (OpenAI-style)
    out = mgr({'tool_name': TOOL_NAME, 'arguments': '{"block": 1}'})
    assert out == 'orig one'


def test_manager_reports_error_on_unknown_block_without_raising():
    tool = ExtractCondensed(Chunks(chunks=[
        _condensed('cmp1', original='orig one')]))
    mgr = ToolManager({})
    mgr.register(tool)
    out = mgr({'tool_name': TOOL_NAME, 'arguments': '{"block": 999}'})
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


LONG_PASSAGE = (
    'Christopher Nolan was born on 30 July 1970 in London. '
    'He is a British-American film director, producer and screenwriter. '
    'His film Inception (2010) is a science-fiction heist movie. '
    'Inception grossed over 829 million dollars worldwide.'
)


@pytest.mark.skipif(not _SPACY_OK, reason='en_core_web_sm not available')
def test_end_to_end_with_keyword_condenser_returns_original():
    from twinkle_agentic.condenser.keyword import KeywordCondenser

    pre = Chunks(chunks=[
        {'type': 'text', 'role': 'user', 'content': LONG_PASSAGE}])
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
        {'type': 'text', 'role': 'user',
         'content': LONG_PASSAGE, 'round': 1},
        {'type': 'text', 'role': 'assistant',
         'content': LONG_PASSAGE + ' Assistant elaboration.', 'round': 1},
    ])
    # skip_roles default excludes assistant → only first chunk condensed.
    post = KeywordCondenser(compression_ratio=4.0, min_chars=50)(pre)
    tool = ExtractCondensed(post)

    # Exactly one wrapped block.
    assert tool.blocks == [1]
    # The trajectory wrapper agrees: block_1 exists, block_2 does not.
    traj = post.to_trajectory()
    rendered = ''.join(
        m['content'] if isinstance(m.get('content'), str) else ''
        for m in traj['messages'])
    assert '<block_1>' in rendered and '</block_1>' in rendered
    assert '<block_2>' not in rendered
    # And the tool returns the correct original.
    assert tool(TOOL_NAME, {'block': 1}) == LONG_PASSAGE
