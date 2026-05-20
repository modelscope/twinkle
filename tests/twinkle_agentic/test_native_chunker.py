# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for :class:`twinkle_agentic.chunker.native.NativeChunker`.

Focus: chunk-size boundaries, separator priority, first-user-only scope,
lossless ``''.join`` of split outputs, and edge cases (empty, multimodal,
tool-calls, invalid config).
"""
from __future__ import annotations

import pytest

from twinkle_agentic.chunker.native import NativeChunker, _hard_cut, _split_keep
from twinkle_agentic.data_format import Chunks


def _u(content, role='user'):
    return {'role': role, 'content': content}


def _join(chunks, type_='text'):
    return ''.join(c['content'] for c in chunks if c.get('type') == type_)


# ---------------------------------------------------------------------------
# chunk_size boundaries
# ---------------------------------------------------------------------------
def test_under_chunk_size_returns_single_chunk():
    ch = NativeChunker(chunk_size=100)
    out = ch({'messages': [_u('hello world')]}).chunks
    assert len(out) == 1
    assert out[0]['content'] == 'hello world'
    assert out[0]['role'] == 'user'
    assert out[0]['type'] == 'text'


def test_exact_chunk_size_not_split():
    ch = NativeChunker(chunk_size=10)
    out = ch({'messages': [_u('a' * 10)]}).chunks
    assert [c['content'] for c in out] == ['a' * 10]


def test_one_over_chunk_size_is_split():
    ch = NativeChunker(chunk_size=10)
    out = ch({'messages': [_u('a' * 11)]}).chunks
    # No separator matches → hard cut; merge won't fuse (10+1 > 10)
    assert len(out) == 2
    assert all(len(c['content']) <= 10 for c in out)
    assert _join(out) == 'a' * 11


def test_all_chunks_respect_size_limit_on_realistic_input():
    ch = NativeChunker(chunk_size=20)
    text = ('hello world. ' * 50).strip()
    out = ch({'messages': [_u(text)]}).chunks
    assert all(len(c['content']) <= 20 for c in out)
    assert _join(out) == text


def test_large_text_split_is_lossless_and_bounded():
    ch = NativeChunker(chunk_size=64)
    text = 'The quick brown fox jumps over the lazy dog. ' * 100
    out = ch({'messages': [_u(text)]}).chunks
    assert _join(out) == text
    assert all(len(c['content']) <= 64 for c in out)


# ---------------------------------------------------------------------------
# separator priority (coarsest available wins)
# ---------------------------------------------------------------------------
def test_paragraph_split_preferred_over_sentence():
    ch = NativeChunker(chunk_size=40)
    text = 'P1 sentence one. P1 sentence two.\n\nP2 sentence one. P2 sentence two.'
    out = ch({'messages': [_u(text)]}).chunks
    assert _join(out) == text
    assert all(len(c['content']) <= 40 for c in out)
    # Because paragraph boundary (18 + 2) and (35) both fit in 40, we
    # expect at most 2 chunks (one per paragraph, possibly merged).
    assert len(out) <= 2


def test_newline_split_used_when_no_paragraph():
    ch = NativeChunker(chunk_size=10)
    text = 'line1\nline2\nline3\nline4'
    out = ch({'messages': [_u(text)]}).chunks
    assert _join(out) == text
    assert all(len(c['content']) <= 10 for c in out)


def test_sentence_split_used_when_no_newline():
    ch = NativeChunker(chunk_size=10)
    text = 'foo bar b. qux qa bc. abc d.'
    out = ch({'messages': [_u(text)]}).chunks
    assert _join(out) == text
    assert all(len(c['content']) <= 10 for c in out)


def test_chinese_sentence_separator():
    ch = NativeChunker(chunk_size=8)
    text = '你好世界。这是测试。再见朋友。'
    out = ch({'messages': [_u(text)]}).chunks
    assert _join(out) == text
    assert all(len(c['content']) <= 8 for c in out)


def test_custom_separator_list_only():
    ch = NativeChunker(chunk_size=10, separators=['|'])
    text = 'aaa|bbb|ccccccccc|dd'
    out = ch({'messages': [_u(text)]}).chunks
    assert _join(out) == text
    assert all(len(c['content']) <= 10 for c in out)


def test_empty_string_sentinel_appended_automatically():
    # User omits '' → chunker must still make progress on unsplittable text
    ch = NativeChunker(chunk_size=3, separators=['|'])
    text = 'abcdefghij'  # no '|' at all
    out = ch({'messages': [_u(text)]}).chunks
    assert _join(out) == text
    assert all(len(c['content']) <= 3 for c in out)


# ---------------------------------------------------------------------------
# first-user-only constraint
# ---------------------------------------------------------------------------
def test_only_first_user_message_is_split():
    ch = NativeChunker(chunk_size=10)
    long = 'a' * 100
    traj = {
        'messages': [
            {
                'role': 'system',
                'content': long
            },
            {
                'role': 'user',
                'content': long
            },  # ← split
            {
                'role': 'assistant',
                'content': long
            },
            {
                'role': 'user',
                'content': long
            },  # ← pass-through
            {
                'role': 'tool',
                'content': long,
                'tool_call_id': 'c1'
            },
        ]
    }
    out = ch(traj).chunks

    # Count chunks per message by position.
    system_chunks = [c for c in out if c['role'] == 'system']
    assistant_chunks = [c for c in out if c['role'] == 'assistant']
    tool_chunks = [c for c in out if c['role'] == 'tool']
    user_chunks = [c for c in out if c['role'] == 'user']

    assert len(system_chunks) == 1
    assert len(assistant_chunks) == 1
    assert len(tool_chunks) == 1
    # First user is split into many + second user pass-through (1 chunk).
    assert len(user_chunks) > 2
    # And the second user chunk sits at the end of the user_chunks group
    # only after the first-user splits.
    assert user_chunks[-1]['content'] == long


def test_system_and_assistant_content_not_split():
    ch = NativeChunker(chunk_size=5)
    long = 'abcdefghijklmn'
    traj = {
        'messages': [
            {
                'role': 'system',
                'content': long
            },
            {
                'role': 'assistant',
                'content': long
            },
        ]
    }
    out = ch(traj).chunks
    assert len(out) == 2
    assert out[0]['content'] == long
    assert out[1]['content'] == long


def test_trajectory_without_user_message_produces_no_split():
    ch = NativeChunker(chunk_size=5)
    long = 'abcdefghij'
    traj = {
        'messages': [
            {
                'role': 'system',
                'content': long
            },
            {
                'role': 'assistant',
                'content': long
            },
        ]
    }
    out = ch(traj).chunks
    assert all(len(c['content']) == len(long) for c in out)


# ---------------------------------------------------------------------------
# decomposition of special message parts
# ---------------------------------------------------------------------------
def test_reasoning_content_becomes_own_chunk():
    ch = NativeChunker(chunk_size=100)
    traj = {
        'messages': [
            _u('hi'),
            {
                'role': 'assistant',
                'reasoning_content': 'think step',
                'content': 'answer'
            },
        ]
    }
    out = ch(traj).chunks
    # user(hi) + assistant.reasoning + assistant.content
    assert len(out) == 3
    assert out[1]['raw']['kind'] == 'reasoning_content'
    assert out[1]['content'] == 'think step'
    assert out[2]['content'] == 'answer'
    assert 'raw' not in out[2] or 'kind' not in out[2].get('raw', {})


def test_tool_calls_become_empty_text_chunks_with_kind():
    ch = NativeChunker(chunk_size=100)
    traj = {
        'messages': [
            _u('hi'),
            {
                'role':
                'assistant',
                'content':
                'calling',
                'tool_calls': [
                    {
                        'type': 'function',
                        'function': {
                            'name': 'foo',
                            'arguments': {}
                        }
                    },
                    {
                        'type': 'function',
                        'function': {
                            'name': 'bar',
                            'arguments': {
                                'x': 1
                            }
                        }
                    },
                ]
            },
        ]
    }
    out = ch(traj).chunks
    tc_chunks = [c for c in out if c.get('raw', {}).get('kind') == 'tool_call']
    assert len(tc_chunks) == 2
    assert tc_chunks[0]['raw']['tool_call']['function']['name'] == 'foo'
    assert tc_chunks[1]['raw']['tool_call']['function']['name'] == 'bar'
    # Empty content on tool_call chunks.
    assert all(c['content'] == '' for c in tc_chunks)


def test_tool_message_preserves_tool_call_id():
    ch = NativeChunker(chunk_size=100)
    traj = {
        'messages': [
            _u('hi'),
            {
                'role': 'tool',
                'content': 'result',
                'tool_call_id': 'call-42'
            },
        ]
    }
    out = ch(traj).chunks
    tool_chunk = out[-1]
    assert tool_chunk['role'] == 'tool'
    assert tool_chunk['raw']['tool_call_id'] == 'call-42'


def test_multimodal_content_preserved_on_first_user():
    ch = NativeChunker(chunk_size=5)
    traj = {
        'messages': [{
            'role':
            'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'describe this image'
                },
                {
                    'type': 'image',
                    'image': 'http://x/y.png'
                },
            ],
        }]
    }
    out = ch(traj).chunks
    text_chunks = [c for c in out if c['type'] == 'text']
    image_chunks = [c for c in out if c['type'] == 'image']
    assert len(image_chunks) == 1
    assert image_chunks[0]['content'] == 'http://x/y.png'
    assert image_chunks[0]['raw'] == {'type': 'image', 'image': 'http://x/y.png'}
    # Text part was split; concatenation is lossless.
    assert _join(text_chunks) == 'describe this image'
    assert all(len(c['content']) <= 5 for c in text_chunks)


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_empty_trajectory():
    ch = NativeChunker(chunk_size=10)
    assert ch({'messages': []}).chunks == []
    assert ch({}).chunks == []


def test_empty_content_string_produces_no_chunks():
    ch = NativeChunker(chunk_size=10)
    assert ch({'messages': [_u('')]}).chunks == []


@pytest.mark.parametrize('bad', [0, -1, -999])
def test_invalid_chunk_size_raises(bad):
    with pytest.raises(ValueError):
        NativeChunker(chunk_size=bad)


def test_chunk_size_one_hard_cuts_all_chars():
    ch = NativeChunker(chunk_size=1)
    text = 'abc'
    out = ch({'messages': [_u(text)]}).chunks
    assert [c['content'] for c in out] == ['a', 'b', 'c']


def test_whitespace_only_text_is_preserved_losslessly():
    ch = NativeChunker(chunk_size=3)
    text = '    \n\n   \n'
    out = ch({'messages': [_u(text)]}).chunks
    assert _join(out) == text
    assert all(len(c['content']) <= 3 for c in out)


# ---------------------------------------------------------------------------
# HotpotQA-shaped realistic payload
# ---------------------------------------------------------------------------
def test_hotpotqa_like_passage_layout():
    ch = NativeChunker(chunk_size=80)
    passages = '\n\n'.join(f'[{i}] Title_{i}: ' + 'This is sentence. ' * 6 for i in range(1, 6))
    user_text = f'Question: who wrote it?\n\nContext:\n\n{passages}'
    out = ch({
        'messages': [
            {
                'role': 'system',
                'content': 'sys'
            },
            _u(user_text),
        ]
    }).chunks
    # System message is not split.
    assert out[0]['role'] == 'system' and out[0]['content'] == 'sys'
    # User text reconstructs losslessly.
    user_chunks = [c for c in out if c['role'] == 'user']
    assert _join(user_chunks) == user_text
    assert all(len(c['content']) <= 80 for c in user_chunks)


# ---------------------------------------------------------------------------
# to_trajectory integration (non-split messages round-trip cleanly)
# ---------------------------------------------------------------------------
def test_non_split_messages_roundtrip_through_to_trajectory():
    ch = NativeChunker(chunk_size=1024)
    tc = {'type': 'function', 'function': {'name': 'foo', 'arguments': {}}}
    traj = {
        'messages': [
            {
                'role': 'system',
                'content': 'sys'
            },
            {
                'role': 'user',
                'content': 'short question'
            },
            {
                'role': 'assistant',
                'content': 'answer',
                'tool_calls': [tc]
            },
            {
                'role': 'tool',
                'content': 'result',
                'tool_call_id': 'c1'
            },
        ]
    }
    chunks = ch(traj)
    back = chunks.to_trajectory(block_wrapper=None)
    msgs = back['messages']
    assert msgs[0] == {'role': 'system', 'content': 'sys'}
    assert msgs[1]['role'] == 'user'
    assert msgs[1]['content'] == 'short question'
    assert msgs[2]['role'] == 'assistant'
    assert msgs[2]['content'] == 'answer'
    assert msgs[2]['tool_calls'] == [tc]
    assert msgs[3]['role'] == 'tool'
    assert msgs[3]['content'] == 'result'
    assert msgs[3]['tool_call_id'] == 'c1'


# ---------------------------------------------------------------------------
# helper-level tests (white-box, catches regressions in primitives)
# ---------------------------------------------------------------------------
def test_split_keep_is_lossless():
    cases = [
        ('', '|'),
        ('abc', '|'),
        ('a|b|c', '|'),
        ('|abc|', '|'),
        ('|||', '|'),
        ('aa..bb.', '.'),
        ('hello', ''),  # empty separator → single piece
    ]
    for text, sep in cases:
        parts = _split_keep(text, sep)
        assert ''.join(parts) == text, (text, sep, parts)


def test_hard_cut_bounds_and_lossless():
    for text, size in [('', 3), ('a', 3), ('abcde', 3), ('abcdef', 3)]:
        parts = _hard_cut(text, size)
        assert ''.join(parts) == text
        assert all(len(p) <= size for p in parts)


def test_split_keep_keeps_separator_suffix():
    assert _split_keep('aa.bb.cc', '.') == ['aa.', 'bb.', 'cc']
    assert _split_keep('aa\n\nbb\n\ncc', '\n\n') == ['aa\n\n', 'bb\n\n', 'cc']


# ---------------------------------------------------------------------------
# separator ordering / priority contract
# ---------------------------------------------------------------------------
def test_prefers_paragraph_boundary_over_period_when_both_fit():
    # Two paragraphs. Each fits in 40. The whole thing (47) does not.
    ch = NativeChunker(chunk_size=40)
    text = 'para one sentence. more.\n\npara two sentence.'
    assert len(text) > 40
    out = ch({'messages': [_u(text)]}).chunks
    # Chunker should split at '\n\n', not inside a paragraph.
    assert out[0]['content'].endswith('\n\n')
    assert _join(out) == text


# ---------------------------------------------------------------------------
# round numbering
# ---------------------------------------------------------------------------
def test_round_starts_at_zero_for_pre_user_system():
    ch = NativeChunker(chunk_size=1024)
    out = ch({
        'messages': [
            {
                'role': 'system',
                'content': 'you are helpful'
            },
            _u('hello'),
        ]
    }).chunks
    assert [c['round'] for c in out] == [0, 1]


def test_round_increments_on_each_user_message():
    ch = NativeChunker(chunk_size=1024)
    out = ch({
        'messages': [
            _u('first user'),
            {
                'role': 'assistant',
                'content': 'first reply'
            },
            _u('second user'),
            {
                'role': 'assistant',
                'content': 'second reply'
            },
            _u('third user'),
        ]
    }).chunks
    rounds = [c['round'] for c in out]
    # assistant msgs inherit the round of the preceding user turn.
    assert rounds == [1, 1, 2, 2, 3]


def test_round_covers_tool_responses_between_users():
    ch = NativeChunker(chunk_size=1024)
    out = ch({
        'messages': [
            _u('query'),
            {
                'role': 'assistant',
                'content': 'calling tool'
            },
            {
                'role': 'tool',
                'content': 'tool result',
                'tool_call_id': 'x'
            },
            {
                'role': 'assistant',
                'content': 'final'
            },
        ]
    }).chunks
    assert {c['round'] for c in out} == {1}


def test_round_preserved_when_first_user_is_split():
    ch = NativeChunker(chunk_size=20)
    long_user = 'hello world. ' * 10  # gets split
    out = ch({
        'messages': [
            {
                'role': 'system',
                'content': 'sys'
            },
            _u(long_user),
            {
                'role': 'assistant',
                'content': 'ack'
            },
            _u('again'),
        ]
    }).chunks
    # All pieces of the split first user share round=1, system is round=0,
    # assistant inherits round=1, second user is round=2.
    by_role = {}
    for c in out:
        by_role.setdefault(c.get('role'), []).append(c['round'])
    assert set(by_role.get('system', [])) == {0}
    assert set(by_role.get('assistant', [])) == {1}
    # Multiple user chunks from the split share round=1.
    assert by_role['user'].count(1) >= 2
    assert by_role['user'][-1] == 2
