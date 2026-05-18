# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit + integration tests for :class:`twinkle_agentic.condenser.model.ModelCondenser`.

Unit tests use a deterministic mock :class:`Sampler` so the suite runs
without GPUs / vLLM. The final block contains an opt-in integration
test that spins up a real ``Qwen/Qwen2.5-3B-Instruct`` sampler on a
single GPU; enable it with::

    TWINKLE_TEST_REAL_SAMPLER=1 pytest tests/twinkle_agentic/test_model_condenser.py
"""
from __future__ import annotations

import math
import os
import pytest
from typing import Callable, List

# Import directly from the submodule to avoid the (currently broken)
# ``twinkle.sampler.__init__`` import chain in this workspace.
from twinkle.data_format.sampling import SampledSequence, SampleResponse, SamplingParams
from twinkle_agentic.condenser.model import ModelCondenser, _strip_code_fences
from twinkle_agentic.data_format import Chunks

# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
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


class _MockSampler:
    """Deterministic duck-typed sampler. Calls ``responder(passage)`` per input.

    We do NOT subclass :class:`twinkle.sampler.base.Sampler` to avoid
    dragging the workspace's currently-broken template init-chain into
    the test module. ``ModelCondenser`` only touches
    ``sampler.sample(...)``, so duck-typing is sufficient.
    """

    def __init__(self, responder: Callable[[str], str]):
        self._responder = responder
        self.template = object()  # truthy placeholder, never inspected
        self.engine = None
        self.calls: list[dict] = []

    def sample(
        self,
        inputs,
        sampling_params=None,
        adapter_name='',
        *,
        num_samples=1,
        **_kw,
    ) -> list[SampleResponse]:
        inputs_list = inputs if isinstance(inputs, list) else [inputs]
        out: list[SampleResponse] = []
        for traj in inputs_list:
            user_msg = next(m for m in traj['messages'] if m['role'] == 'user')
            prompt = user_msg['content']
            marker = 'Passage:\n'
            idx = prompt.rfind(marker)
            passage = prompt[idx + len(marker):] if idx >= 0 else prompt
            decoded = self._responder(passage)
            self.calls.append({
                'passage': passage,
                'sampling_params': sampling_params,
            })
            out.append(SampleResponse(sequences=[SampledSequence(stop_reason='stop', tokens=[], decoded=decoded)]))
        return out


def _well_formed_markdown(passage: str) -> str:
    """A standard three-section markdown response."""
    return ('## Summary\n'
            'Christopher Nolan is a British-American director born in London in 1970.\n\n'
            '## Key Facts\n'
            '- Nolan directed Inception (2010) starring Leonardo DiCaprio.\n'
            '- Inception grossed over 829 million dollars worldwide.\n'
            '- Nolan also directed The Dark Knight trilogy and Interstellar.\n\n'
            '## More\n'
            'Nolan, Inception, Leonardo DiCaprio, Interstellar, London, 1970')


# ---------------------------------------------------------------------------
# constructor validation
# ---------------------------------------------------------------------------
def test_requires_sampler():
    with pytest.raises(ValueError):
        ModelCondenser(sampler=None)


@pytest.mark.parametrize('kw', [
    {
        'compression_ratio': 1.0
    },
    {
        'compression_ratio': 0.5
    },
    {
        'min_chars': -1
    },
    {
        'batch_size': 0
    },
    {
        'user_prompt_template': 'no placeholders'
    },
    {
        'user_prompt_template': 'only {budget} placeholder'
    },
    {
        'user_prompt_template': 'only {text} placeholder'
    },
])
def test_invalid_config_raises(kw):
    with pytest.raises(ValueError):
        ModelCondenser(_MockSampler(_well_formed_markdown), **kw)


# ---------------------------------------------------------------------------
# pure helper smoke tests
# ---------------------------------------------------------------------------
def test_strip_code_fences():
    wrapped = '```markdown\n## Summary\nhi\n```'
    assert _strip_code_fences(wrapped) == '## Summary\nhi'
    # No fence → returned as-is.
    plain = '## Summary\nhi'
    assert _strip_code_fences(plain) == plain


# ---------------------------------------------------------------------------
# compression-vs-passthrough semantics (no hard clamp anymore)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('ratio', [2.0, 3.0, 4.0, 6.0, 10.0])
def test_compressed_output_is_strictly_shorter_than_original(ratio):
    cond = ModelCondenser(
        _MockSampler(_well_formed_markdown),
        compression_ratio=ratio,
        min_chars=50,
        min_budget_chars=1,
    )
    chunk = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    if chunk.get('raw', {}).get('condensed'):
        # When accepted, output MUST be strictly shorter than the input.
        assert len(
            chunk['content']) < len(LONG_PASSAGE), (f'ratio={ratio}: condensed output len={len(chunk["content"])}'
                                                    f' must be < original len={len(LONG_PASSAGE)}')
    else:
        # Passthrough: chunk must be byte-identical to the input.
        assert chunk['content'] == LONG_PASSAGE


def test_overlong_model_output_falls_back_to_original():
    """When the LLM output is not strictly shorter than the input,
    the original passage is kept verbatim and NOT marked condensed."""
    overflow = lambda _p: _well_formed_markdown('') * 5  # noqa: E731
    cond = ModelCondenser(_MockSampler(overflow), compression_ratio=3.0, min_chars=50, min_budget_chars=1)
    chunk = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    assert chunk['content'] == LONG_PASSAGE
    assert not (chunk.get('raw') or {}).get('condensed')


def test_equal_length_model_output_falls_back_to_original():
    """Output equal in length to the input is treated as non-useful
    compression and triggers passthrough."""
    same_length = lambda p: 'X' * len(p)  # noqa: E731
    cond = ModelCondenser(_MockSampler(same_length), compression_ratio=4.0, min_chars=50, min_budget_chars=1)
    chunk = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    assert chunk['content'] == LONG_PASSAGE
    assert not (chunk.get('raw') or {}).get('condensed')


# ---------------------------------------------------------------------------
# structural output quality
# ---------------------------------------------------------------------------
def test_well_formed_output_keeps_three_sections_at_generous_budget():
    cond = ModelCondenser(_MockSampler(_well_formed_markdown), compression_ratio=1.1, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    assert '## Summary' in out
    assert '## Key Facts' in out
    assert '## More' in out
    # Primary entities survive in some form.
    assert 'Nolan' in out or 'Inception' in out


def test_tight_ratio_still_accepts_shorter_output():
    """At a tight ratio, whatever the LLM produces is accepted as long
    as it is strictly shorter than the input; we no longer clamp it."""

    def responder(_p):
        return ('## Summary\nA short sentence.\n\n'
                '## More\nTopics: x, y, z.\n\n'
                '## Key Facts\n- Fact one here.\n- Fact two here.')

    cond = ModelCondenser(_MockSampler(responder), compression_ratio=3.5, min_chars=50, min_budget_chars=1)
    chunk = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    assert chunk['raw']['condensed'] is True
    assert len(chunk['content']) < len(LONG_PASSAGE)
    assert '## Summary' in chunk['content']


def test_degenerate_output_falls_back_to_original():
    """When model output has NO alphanumerics (pure markdown markers),
    the condenser falls back to the original passage verbatim."""
    markers_only = lambda _p: '## \n- \n##'  # noqa: E731
    cond = ModelCondenser(_MockSampler(markers_only), compression_ratio=4.0, min_chars=50, min_budget_chars=1)
    chunk = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    assert chunk['content'] == LONG_PASSAGE
    assert not (chunk.get('raw') or {}).get('condensed')


def test_garbled_but_shorter_output_is_accepted():
    """If the model emits unstructured but strictly shorter text, we
    take it verbatim — the condenser is not a format validator."""
    garbled = lambda _p: 'this is some unstructured blob'  # noqa: E731
    cond = ModelCondenser(_MockSampler(garbled), compression_ratio=4.0, min_chars=50, min_budget_chars=1)
    chunk = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    assert chunk['raw']['condensed'] is True
    assert 'unstructured' in chunk['content']
    assert len(chunk['content']) < len(LONG_PASSAGE)


def test_code_fenced_output_is_unwrapped():
    wrapped = lambda _p: '```markdown\n' + _well_formed_markdown('') + '\n```'  # noqa: E731
    cond = ModelCondenser(_MockSampler(wrapped), compression_ratio=1.5, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]['content']
    # After unwrapping, header is at the start (no leading ```).
    assert not out.startswith('```')
    assert out.startswith('## Summary')


# ---------------------------------------------------------------------------
# raw.condensed marker + block wrapping
# ---------------------------------------------------------------------------
def test_marks_condensed_and_wraps_in_block_tags():
    cond = ModelCondenser(_MockSampler(_well_formed_markdown), compression_ratio=4.0, min_chars=50)
    chunks = cond(_wrap(_user_chunk(LONG_PASSAGE)))
    assert chunks.chunks[0]['raw']['condensed'] is True
    traj = chunks.to_trajectory()
    user_content = traj['messages'][0]['content']
    assert '<block_1>' in user_content and '</block_1>' in user_content


def test_multiple_chunks_numbered_sequentially():
    cond = ModelCondenser(_MockSampler(_well_formed_markdown), compression_ratio=4.0, min_chars=50, batch_size=2)
    passages = [_user_chunk(LONG_PASSAGE) for _ in range(3)]
    chunks = cond(_wrap(*passages))
    traj = chunks.to_trajectory()
    content = traj['messages'][0]['content']
    for i in (1, 2, 3):
        assert f'<block_{i}>' in content and f'</block_{i}>' in content
    assert '<block_4>' not in content


# ---------------------------------------------------------------------------
# selection policy
# ---------------------------------------------------------------------------
def test_skip_roles_default_preserves_system_tool_assistant():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50)
    src = _wrap(
        _user_chunk(LONG_PASSAGE, role='system'),
        _user_chunk(LONG_PASSAGE, role='assistant'),
        _user_chunk(LONG_PASSAGE, role='tool'),
        _user_chunk(LONG_PASSAGE, role='user'),
    )
    out = cond(src).chunks
    for i in range(3):
        assert out[i]['content'] == LONG_PASSAGE
        assert (out[i].get('raw') or {}).get('condensed') is not True
    assert out[3]['raw']['condensed'] is True
    # Only one real compression job (the user chunk); the batch is padded
    # up to ``batch_size`` with duplicates of that job to keep distributed
    # samplers happy, and the extra responses are then discarded.
    assert len(sampler.calls) == cond.batch_size


def test_custom_skip_roles_empty_tuple():
    cond = ModelCondenser(_MockSampler(_well_formed_markdown), compression_ratio=4.0, min_chars=50, skip_roles=())
    src = _wrap(_user_chunk(LONG_PASSAGE, role='assistant'))
    out = cond(src).chunks
    assert out[0]['raw']['condensed'] is True


def test_short_content_passes_through():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=500)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks
    assert out[0]['content'] == LONG_PASSAGE
    assert (out[0].get('raw') or {}).get('condensed') is not True
    assert sampler.calls == []


def test_non_text_chunk_passes_through():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=1)
    img = {
        'type': 'image',
        'content': 'http://x/y.png',
        'role': 'user',
        'raw': {
            'type': 'image',
            'image': 'http://x/y.png'
        }
    }
    out = cond(_wrap(img)).chunks
    assert out[0] == img
    assert sampler.calls == []


def test_reasoning_kind_chunk_passes_through():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50)
    reasoning = {
        'type': 'text',
        'role': 'user',
        'content': LONG_PASSAGE,
        'raw': {
            'kind': 'reasoning_content'
        },
    }
    out = cond(_wrap(reasoning)).chunks
    assert (out[0].get('raw') or {}).get('condensed') is not True
    assert sampler.calls == []


def test_already_condensed_chunk_is_not_reprocessed():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50)
    once = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    assert once['raw']['condensed'] is True
    sampler.calls.clear()
    twice = cond(_wrap(once)).chunks[0]
    # No second sampler call — idempotent.
    assert sampler.calls == []
    assert twice == once


# ---------------------------------------------------------------------------
# batching & ordering
# ---------------------------------------------------------------------------
def test_batching_respects_batch_size():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50, batch_size=2)
    src = _wrap(*[_user_chunk(LONG_PASSAGE) for _ in range(5)])
    out = cond(src).chunks
    assert len(out) == 5
    for c in out:
        assert c['raw']['condensed'] is True
    # 5 real jobs dispatched in batches of ``batch_size=2`` with the last
    # batch padded to full size: 2 + 2 + 2 = 6 sampler calls, of which
    # only 5 correspond to real work (the 6th is a duplicate discarded).
    assert len(sampler.calls) == 6


def test_order_preserved_with_mixed_chunks():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50, batch_size=2)
    src = _wrap(
        _user_chunk('short', role='user'),  # too short
        _user_chunk(LONG_PASSAGE, role='user'),  # condensed
        _user_chunk(LONG_PASSAGE, role='system'),  # skipped role
        _user_chunk(LONG_PASSAGE, role='user'),  # condensed
    )
    out = cond(src).chunks
    assert len(out) == 4
    assert out[0]['content'] == 'short'
    assert out[1]['raw']['condensed'] is True
    assert out[2]['content'] == LONG_PASSAGE
    assert (out[2].get('raw') or {}).get('condensed') is not True
    assert out[3]['raw']['condensed'] is True


# ---------------------------------------------------------------------------
# prompt robustness
# ---------------------------------------------------------------------------
def test_braces_in_text_do_not_break_prompt_formatting():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50)
    text = ('The JSON config was {"model": "Qwen", "temperature": 0.7}. ' * 5)
    out = cond(_wrap(_user_chunk(text))).chunks[0]
    assert out['raw']['condensed'] is True
    # Prompt contained the raw text verbatim.
    assert sampler.calls[0]['passage'].strip().startswith('The JSON config was {"model":')


def test_prompt_mentions_budget_in_user_message():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=3.0, min_chars=50)
    cond(_wrap(_user_chunk(LONG_PASSAGE)))
    expected_budget = math.ceil(len(LONG_PASSAGE) / 3.0)
    # The mock recorded the prompt passage; we check the sampling_params
    # carries a reasonable max_tokens (derived from budget).
    assert sampler.calls[0]['sampling_params'].max_tokens >= expected_budget // 2


def test_custom_sampling_params_is_forwarded():
    sampler = _MockSampler(_well_formed_markdown)
    custom = SamplingParams(temperature=0.3, max_tokens=256)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50, sampling_params=custom)
    cond(_wrap(_user_chunk(LONG_PASSAGE)))
    assert sampler.calls[0]['sampling_params'] is custom


# ---------------------------------------------------------------------------
# semantic preservation (mock-level sanity)
# ---------------------------------------------------------------------------
def test_semantic_preservation_when_compressed():
    """When the condenser accepts the model output, important entities
    survive in some form."""
    cond = ModelCondenser(_MockSampler(_well_formed_markdown), compression_ratio=2.0, min_chars=50, min_budget_chars=1)
    chunk = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    out = chunk['content']
    if chunk.get('raw', {}).get('condensed'):
        hits = sum(1 for ent in ('Nolan', 'Inception', 'Leonardo DiCaprio', 'London') if ent in out)
        assert hits >= 2
    else:
        # Passthrough branch: the original must be returned verbatim.
        assert out == LONG_PASSAGE


# ---------------------------------------------------------------------------
# integration test (opt-in; requires single GPU + vLLM + Qwen model)
# ---------------------------------------------------------------------------
INTEGRATION_ENABLED = bool(os.environ.get('TWINKLE_TEST_REAL_SAMPLER'))
INTEGRATION_MODEL = os.environ.get('TWINKLE_TEST_MODEL', 'Qwen/Qwen2.5-3B-Instruct')


@pytest.mark.skipif(
    not INTEGRATION_ENABLED,
    reason='Set TWINKLE_TEST_REAL_SAMPLER=1 to run the real-model integration test',
)
def test_integration_real_qwen_sampler_end_to_end():
    """End-to-end test with a real Qwen sampler on a single GPU."""
    vllm = pytest.importorskip('vllm')  # noqa: F841
    from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler

    sampler = vLLMSampler(
        model_id=INTEGRATION_MODEL,
        engine_args={
            'dtype': 'bfloat16',
            'gpu_memory_utilization': 0.7,
            'max_model_len': 4096,
            'enforce_eager': True,
        },
    )
    try:
        sampler.set_template('qwen2_5')
    except Exception:
        # Fall back to 'auto' template detection if the named one
        # isn't registered in this build.
        sampler.set_template('default')

    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50)
    chunk = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    out = chunk['content']

    # Either the model produced a strictly shorter compression (most
    # common), or the chunk is passed through verbatim.
    if chunk.get('raw', {}).get('condensed'):
        assert 0 < len(out) < len(LONG_PASSAGE)
        assert any(ent in out for ent in ('Nolan', 'Inception', 'London', 'Leonardo'))
    else:
        assert out == LONG_PASSAGE


# ---------------------------------------------------------------------------
# round-based selection filter
# ---------------------------------------------------------------------------
def _round_chunk(text, round_idx, role='user'):
    return {'role': role, 'type': 'text', 'content': text, 'round': round_idx}


def test_rounds_filter_only_compresses_first_user_turn():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50, rounds=[1])
    out = cond(_wrap(
        _round_chunk(LONG_PASSAGE, 1),
        _round_chunk(LONG_PASSAGE + ' extra.', 2),
    )).chunks
    # One real compression job (round 1) padded up to ``batch_size``.
    assert len(sampler.calls) == cond.batch_size
    # Round 1 compressed.
    assert out[0]['raw']['condensed'] is True
    # Round 2 untouched.
    assert out[1]['content'].endswith(' extra.')
    assert not (out[1].get('raw') or {}).get('condensed')


def test_rounds_filter_excludes_chunks_without_round_field():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50, rounds=[1])
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    # No call because the chunk had no ``round`` field.
    assert sampler.calls == []
    assert out['content'] == LONG_PASSAGE
    assert not (out.get('raw') or {}).get('condensed')


def test_rounds_filter_default_none_preserves_legacy_behavior():
    sampler = _MockSampler(_well_formed_markdown)
    cond = ModelCondenser(sampler, compression_ratio=4.0, min_chars=50)
    out = cond(_wrap(_user_chunk(LONG_PASSAGE))).chunks[0]
    assert out['raw']['condensed'] is True
    # One real job, padded up to ``batch_size``.
    assert len(sampler.calls) == cond.batch_size
