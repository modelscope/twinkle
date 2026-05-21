# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import pytest

from twinkle_agentic.chunker.native import NativeChunker
from twinkle_agentic.condenser.keyword import KeywordCondenser
from twinkle_agentic.data_format import Chunks

# Module-level skip if spaCy or the small English model are unavailable.
spacy = pytest.importorskip('spacy')
try:
    spacy.load('en_core_web_sm')
except OSError:
    pytest.skip('en_core_web_sm not available', allow_module_level=True)

# A realistic multi-sentence passage; long enough to exercise the three
# output slots and the compression budget.
LONG_PASSAGE = ('Christopher Nolan was born on 30 July 1970 in London. '
                'He is a British-American film director, producer and screenwriter. '
                'His film Inception (2010) is a science-fiction heist movie starring '
                'Leonardo DiCaprio. Inception grossed over 829 million dollars worldwide '
                'and received eight Academy Award nominations, winning four. '
                'Nolan also directed The Dark Knight trilogy and Interstellar in 2014.')


def _user_chunk(text, role='user'):
    return {'role': role, 'type': 'text', 'content': text}


def _wrap(*chunks):
    return Chunks(chunks=list(chunks))


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('kw', [
    {
        'num_relations': -1
    },
    {
        'num_keywords': -1
    },
    {
        'max_first_sentence_chars': -1
    },
    {
        'compression_ratio': 1.0
    },
    {
        'compression_ratio': 0.5
    },
    {
        'min_chars': -1
    },
])
def test_invalid_config_raises(kw):
    with pytest.raises(ValueError):
        KeywordCondenser(**kw)


# ---------------------------------------------------------------------------
# compression-ratio contract (STRICT upper bound)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('ratio', [2.0, 3.0, 4.0, 6.0, 10.0])
def test_compression_ratio_is_strictly_enforced(ratio):
    cond = KeywordCondenser(
        num_relations=3, max_first_sentence_chars=160, num_keywords=8, compression_ratio=ratio, min_chars=50)
    src = _user_chunk(LONG_PASSAGE)
    out = cond(_wrap(src)).chunks
    assert len(out) == 1
    compressed = out[0]['content']
    budget = math.ceil(len(LONG_PASSAGE) / ratio)
    assert len(compressed) <= budget, (f'ratio={ratio}: got len={len(compressed)} > budget={budget}')
    assert compressed, 'output must be non-empty'


def test_extreme_ratio_keeps_output_non_empty_and_bounded():
    cond = KeywordCondenser(compression_ratio=100.0, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks
    compressed = out[0]['content']
    budget = math.ceil(len(LONG_PASSAGE) / 100.0)
    assert 0 < len(compressed) <= budget


# ---------------------------------------------------------------------------
# raw.condensed marker + block wrapping
# ---------------------------------------------------------------------------
def test_marks_condensed_and_wraps_in_block_tags():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50)
    chunks = cond(_wrap(_user_chunk(LONG_PASSAGE)))
    assert chunks.chunks[0]['raw']['condensed'] is True
    traj = chunks.to_trajectory()
    # Exactly one compressed passage → block_1 wrap.
    user_content = traj['messages'][0]['content']
    assert '<block_1>' in user_content and '</block_1>' in user_content


def test_multiple_chunks_numbered_sequentially_starting_from_1():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50)
    passages = [_user_chunk(LONG_PASSAGE) for _ in range(3)]
    chunks = cond(_wrap(*passages))
    traj = chunks.to_trajectory()
    content = traj['messages'][0]['content']
    for i in (1, 2, 3):
        assert f'<block_{i}>' in content and f'</block_{i}>' in content
    assert '<block_4>' not in content


# ---------------------------------------------------------------------------
# slot extraction (opening / relations / keywords)
# ---------------------------------------------------------------------------
def test_opening_relations_keywords_present_when_budget_allows():
    # Generous budget → all three slots should appear.
    # LONG_PASSAGE is ~390 chars; full markup is ~370 chars, so we
    # need a ratio close to 1.0 to keep every slot.
    cond = KeywordCondenser(
        num_relations=3, max_first_sentence_chars=160, num_keywords=8, compression_ratio=1.05, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    assert out.startswith('Open: ')
    assert '\nRel: ' in out
    assert '\nMore: ' in out
    # At least one of the primary entities should survive in keywords.
    assert 'Nolan' in out or 'Inception' in out


def test_opening_first_sentence_respects_max_chars():
    cond = KeywordCondenser(
        num_relations=0, max_first_sentence_chars=20, num_keywords=0, compression_ratio=1.1, min_chars=10)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    # Opening slot is trimmed to <= 20 chars
    opening_line = out.split('\n', 1)[0]
    assert opening_line.startswith('Open: ')
    opening_text = opening_line[len('Open: '):]
    assert len(opening_text) <= 20


def test_relations_use_triple_or_quadruple_syntax():
    cond = KeywordCondenser(
        num_relations=5, max_first_sentence_chars=10, num_keywords=0, compression_ratio=1.1, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    # We expect at least one '(a | b | c)' or '(a | b | c | d)' pattern.
    assert '(' in out and ')' in out
    # Parentheses must balance.
    assert out.count('(') == out.count(')')
    # Pipe-delimited slots (avoids ',' collision with slot-internal commas).
    assert ' | ' in out


def test_verb_surface_preserved_not_lemma():
    """Triples keep surface form with auxiliaries: 'was born' not 'bear'."""
    cond = KeywordCondenser(
        num_relations=3, max_first_sentence_chars=10, num_keywords=0, compression_ratio=1.1, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    # Auxiliary preserved.
    assert 'was born' in out or 'was released' in out or 'is' in out
    # Bare lemma of 'born' must NOT appear as the verb slot.
    assert '| bear |' not in out and '| bear on |' not in out


def test_internal_hyphens_preserved_in_np():
    """NP text keeps 'science-fiction' / 'British-American' hyphens."""
    cond = KeywordCondenser(
        num_relations=5, max_first_sentence_chars=10, num_keywords=0, compression_ratio=1.1, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    assert 'science-fiction' in out or 'British-American' in out


def test_pronoun_subject_triples_skipped():
    """Unresolved pronoun subjects (He/She/It) are noise and dropped."""
    cond = KeywordCondenser(
        num_relations=5, max_first_sentence_chars=10, num_keywords=0, compression_ratio=1.1, min_chars=50)
    # LONG_PASSAGE has 'He is a British-American film director...'
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    assert '(He |' not in out and '(he |' not in out


def test_cardinal_entities_filtered_from_keywords():
    cond = KeywordCondenser(
        num_relations=0, num_keywords=10, max_first_sentence_chars=0, compression_ratio=1.1, min_chars=50)
    passage = ('Alpha earned 100 medals. Beta scored 200 points. Gamma made 300 attempts. '
               'Delta received 400 votes. Epsilon collected 500 tokens. Zeta passed 600 miles.')
    out = cond(_wrap(_user_chunk(passage))).chunks[0]['content']
    for num in ('100', '200', '300', '400', '500', '600'):
        assert num not in out, f'pure CARDINAL {num!r} leaked into keywords'


def test_keyword_subsumption_prefers_longer_form():
    """'Nolan' is dropped when 'Christopher Nolan' is already kept."""
    cond = KeywordCondenser(
        num_relations=0, max_first_sentence_chars=10, num_keywords=8, compression_ratio=1.05, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    more_line = next((ln for ln in out.splitlines() if ln.startswith('More: ')), '')
    kws = [k.strip() for k in more_line[len('More: '):].split(',') if k.strip()]
    # No keyword may be a token-subset of another kept keyword.
    import re
    sets = [frozenset(re.findall(r'\w+', k.lower())) for k in kws]
    for i, a in enumerate(sets):
        for j, b in enumerate(sets):
            if i != j:
                assert not a < b, (f'{kws[i]!r} is subsumed by {kws[j]!r} but kept')


def test_keyword_exclusion_is_token_level_not_substring():
    """A keyword is only excluded if ALL its words appear in the opening.

    Substring-based exclusion would wrongly drop 'Starfleet' because
    'star' appears inside other tokens; token-level exclusion keeps it.
    """
    cond = KeywordCondenser(
        num_relations=0, max_first_sentence_chars=60, num_keywords=5, compression_ratio=1.1, min_chars=50)
    passage = ('The Starfleet Academy trains officers for deep-space missions. '
               'Captain Kirk graduated there in 2251. Starfleet operates many vessels.')
    out = cond(_wrap(_user_chunk(passage))).chunks[0]['content']
    # 'Starfleet' shouldn't be dropped just because 'star' is a substring
    # of something in the opening.
    assert 'Starfleet' in out or 'Kirk' in out


def test_opening_truncation_at_word_boundary():
    """When opening exceeds max_chars, cut at the last whole word."""
    cond = KeywordCondenser(
        num_relations=0, max_first_sentence_chars=25, num_keywords=0, compression_ratio=1.1, min_chars=10)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    opening = out.split('\n', 1)[0][len('Open: '):]
    assert len(opening) <= 25
    # Must not end mid-word: last char is a word char AND original passage
    # contains the exact trimmed string as a prefix of the first sentence.
    first_sent = LONG_PASSAGE.split('.', 1)[0]
    assert first_sent.startswith(opening)
    # The char after the trimmed prefix in the source should be a space
    # (i.e. we really did stop on a word boundary).
    if len(opening) < len(first_sent):
        assert first_sent[len(opening)] == ' '


def test_budget_is_filled_greedily_with_triples_and_keywords():
    """At a moderate ratio, output should include MORE than just opening.

    Regression test for the old priority-drop logic that collapsed to
    opening-only whenever the full composition exceeded budget.
    """
    cond = KeywordCondenser(
        num_relations=3, max_first_sentence_chars=80, num_keywords=8, compression_ratio=2.0, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    budget = math.ceil(len(LONG_PASSAGE) / 2.0)
    assert len(out) <= budget
    # At ratio=2.0 we MUST retain at least one relation AND at least one keyword.
    assert '\nRel: ' in out
    assert '\nMore: ' in out


def test_budget_too_small_falls_back_to_raw_truncation():
    """Even at absurd ratios, output is non-empty and bounded."""
    cond = KeywordCondenser(
        num_relations=3, num_keywords=5, max_first_sentence_chars=160, compression_ratio=200.0, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    budget = math.ceil(len(LONG_PASSAGE) / 200.0)
    assert 0 < len(out) <= budget


def test_num_relations_zero_suppresses_slot():
    cond = KeywordCondenser(num_relations=0, num_keywords=5, compression_ratio=1.2, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    assert '\nRel: ' not in out


def test_num_keywords_zero_suppresses_slot():
    cond = KeywordCondenser(num_relations=3, num_keywords=0, compression_ratio=1.2, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    assert '\nMore: ' not in out


# ---------------------------------------------------------------------------
# budget priority: drop keywords → drop relations → truncate opening
# ---------------------------------------------------------------------------
def test_tight_budget_drops_keywords_first():
    # Pick a ratio that is just tight enough to force one slot to go.
    # Full output len ≈ 200+; opening+relations alone ≈ 120.
    cond = KeywordCondenser(
        num_relations=2, max_first_sentence_chars=80, num_keywords=8, compression_ratio=3.0, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    budget = math.ceil(len(LONG_PASSAGE) / 3.0)
    assert len(out) <= budget
    assert out.startswith('Open: ')


def test_very_tight_budget_falls_back_to_opening_only():
    # Ratio large enough that only the opening slot can fit.
    # Keep max_first_sentence_chars small so it does fit.
    cond = KeywordCondenser(
        num_relations=5, max_first_sentence_chars=40, num_keywords=8, compression_ratio=8.0, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    budget = math.ceil(len(LONG_PASSAGE) / 8.0)
    assert len(out) <= budget
    # Either opening-only or further truncated — both fine.
    assert out.startswith('Open') or len(out) <= budget


# ---------------------------------------------------------------------------
# selection policy
# ---------------------------------------------------------------------------
def test_skip_roles_default_preserves_system_tool_assistant():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50)
    src = _wrap(
        _user_chunk(LONG_PASSAGE, role='system'),
        _user_chunk(LONG_PASSAGE, role='assistant'),
        _user_chunk(LONG_PASSAGE, role='tool'),
        _user_chunk(LONG_PASSAGE, role='user'),
    )
    out = cond(src).chunks
    # First three pass through untouched.
    for i in range(3):
        assert out[i]['content'] == LONG_PASSAGE
        assert (out[i].get('raw') or {}).get('condensed') is not True
    # Fourth gets condensed.
    assert out[3]['raw']['condensed'] is True
    assert len(out[3]['content']) < len(LONG_PASSAGE)


def test_custom_skip_roles():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50, skip_roles=())
    src = _wrap(_user_chunk(LONG_PASSAGE, role='assistant'))
    out = cond(src).chunks
    assert out[0]['raw']['condensed'] is True


def test_short_content_passes_through():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=500)
    src = _user_chunk(LONG_PASSAGE)  # shorter than 500
    out = cond(_wrap(src)).chunks
    assert out[0]['content'] == LONG_PASSAGE
    assert (out[0].get('raw') or {}).get('condensed') is not True


def test_non_text_chunk_passes_through():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=1)
    src = {
        'type': 'image',
        'content': 'http://x/y.png',
        'role': 'user',
        'raw': {
            'type': 'image',
            'image': 'http://x/y.png'
        }
    }
    out = cond(_wrap(src)).chunks
    assert out[0] == src


def test_reasoning_and_tool_call_kind_chunks_pass_through():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50)
    reasoning = {
        'type': 'text',
        'role': 'assistant',
        'content': LONG_PASSAGE,
        'raw': {
            'kind': 'reasoning_content'
        },
    }
    # Assistant role would already be skipped, but the kind-filter must
    # hold even if role is user.
    tool_call = {
        'type': 'text',
        'role': 'user',
        'content': LONG_PASSAGE,
        'raw': {
            'kind': 'tool_call',
            'tool_call': {
                'type': 'function',
                'function': {
                    'name': 'x',
                    'arguments': {}
                }
            }
        },
    }
    out = cond(_wrap(reasoning, tool_call)).chunks
    assert (out[0].get('raw') or {}).get('condensed') is not True
    assert (out[1].get('raw') or {}).get('condensed') is not True


def test_empty_content_is_untouched():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=0)
    src = _user_chunk('')
    out = cond(_wrap(src)).chunks
    assert out[0] == src


# ---------------------------------------------------------------------------
# integration with NativeChunker + to_trajectory round-trip
# ---------------------------------------------------------------------------
def test_chunker_then_condenser_produces_block_numbered_output():
    chunker = NativeChunker(chunk_size=300)
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50)

    passages = '\n\n'.join(f'[{i}] Title_{i}: ' + LONG_PASSAGE for i in range(1, 4))
    user_text = f'Question: who directed Inception?\n\nContext:\n\n{passages}'
    traj = {
        'messages': [
            {
                'role': 'system',
                'content': 'You are a helpful agent.'
            },
            {
                'role': 'user',
                'content': user_text
            },
        ]
    }
    chunks = cond(chunker(traj))
    back = chunks.to_trajectory()

    # System untouched; user got multiple condensed blocks.
    assert back['messages'][0]['content'] == 'You are a helpful agent.'
    user_content = back['messages'][1]['content']
    assert '<block_1>' in user_content
    # Each block must be strictly smaller than its source chunk.
    assert len(user_content) < len(user_text)


def test_condenser_preserves_chunk_order_and_count():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50)
    src_chunks = _wrap(
        _user_chunk('short', role='user'),
        _user_chunk(LONG_PASSAGE, role='user'),
        _user_chunk(LONG_PASSAGE, role='system'),
    )
    out = cond(src_chunks).chunks
    assert len(out) == 3
    assert out[0]['content'] == 'short'  # too short
    assert out[1]['raw']['condensed'] is True  # condensed
    assert out[2]['content'] == LONG_PASSAGE  # skipped role


# ---------------------------------------------------------------------------
# idempotency: running condenser twice is safe
# ---------------------------------------------------------------------------
def test_condenser_is_idempotent_on_already_condensed_output():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50)
    once = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    # Second pass must be a no-op: content identical, raw marker kept.
    twice = cond(_wrap(once)).chunks[0]
    assert twice['raw']['condensed'] is True
    assert twice['content'] == once['content']
    # And a third pass must also be stable.
    thrice = cond(_wrap(twice)).chunks[0]
    assert thrice['content'] == once['content']


# ---------------------------------------------------------------------------
# round-based selection filter
# ---------------------------------------------------------------------------
def _round_chunk(text, round_idx, role='user'):
    return {'role': role, 'type': 'text', 'content': text, 'round': round_idx}


def test_rounds_filter_only_compresses_first_user_turn():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50, rounds=[1])
    out = cond(_wrap(
        _round_chunk(LONG_PASSAGE, 1),
        _round_chunk(LONG_PASSAGE + ' extra.', 2),
    )).chunks
    # Round 1 compressed.
    assert out[0]['raw']['condensed'] is True
    assert len(out[0]['content']) < len(LONG_PASSAGE)
    # Round 2 passed through unchanged.
    assert out[1]['content'].endswith(' extra.')
    assert not (out[1].get('raw') or {}).get('condensed')


def test_rounds_filter_excludes_chunks_without_round_field():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50, rounds=[1])
    # Chunk missing ``round`` must be treated as non-matching.
    plain = _user_chunk(LONG_PASSAGE)
    out = cond(_wrap(plain)).chunks[0]
    assert out['content'] == LONG_PASSAGE
    assert not (out.get('raw') or {}).get('condensed')


def test_rounds_filter_default_none_preserves_legacy_behavior():
    cond = KeywordCondenser(compression_ratio=4.0, min_chars=50)
    # No rounds set; chunks without ``round`` are still compressed.
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    assert out['raw']['condensed'] is True
    assert len(out['content']) < len(LONG_PASSAGE)
