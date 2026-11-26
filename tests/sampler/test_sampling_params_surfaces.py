# Copyright (c) ModelScope Contributors. All rights reserved.
"""G18 regression pins for SamplingParams conversion surfaces (no engine needed)."""
import pytest

from twinkle.data_format import SamplingParams


def test_from_dict_does_not_mutate_the_callers_dict():
    d = {'max_new_tokens': 8, 'temperature': 0.7, 'not_a_field': 1}
    params = SamplingParams.from_dict(d)
    assert d == {'max_new_tokens': 8, 'temperature': 0.7, 'not_a_field': 1}
    assert params.max_tokens == 8  # the rename still happens
    assert params.temperature == 0.7


def test_from_dict_max_tokens_wins_over_max_new_tokens():
    params = SamplingParams.from_dict({'max_tokens': 4, 'max_new_tokens': 8})
    assert params.max_tokens == 4


def test_top_k_minus_one_disables_top_k_in_every_backend():
    params = SamplingParams(top_k=-1, temperature=1.0)
    # HF spells "disabled" as 0; omitting it lets HF's fallback default (50) or a
    # checkpoint-declared top_k keep filtering.
    assert params.to_transformers()['top_k'] == 0
    # vLLM/sglang accept -1 as "all tokens"; carrying it explicitly overrides any
    # model-declared default the engine would otherwise inject.
    assert params.to_sglang()['top_k'] == -1
    vllm_params = pytest.importorskip('vllm').SamplingParams
    assert params.to_vllm().top_k in (-1, 0)
    # positive values still pass through unchanged
    assert SamplingParams(top_k=7).to_transformers()['top_k'] == 7
    assert SamplingParams(top_k=7).to_sglang()['top_k'] == 7


def test_zero_top_k_is_still_rejected_at_the_twinkle_boundary():
    with pytest.raises(ValueError, match='top_k must be -1 or >= 1'):
        SamplingParams(top_k=0)


def test_string_stops_use_stop_string_criteria_not_eos_folding():
    import os
    transformers = pytest.importorskip('transformers')
    model_id = os.environ.get(
        'G18_TEST_MODEL',
        '/mnt/workspace/.cache/modelscope/hub/models/Qwen--Qwen3.5-0.8B/snapshots/master')
    if not os.path.isdir(model_id):
        pytest.skip(f'local test model not available: {model_id}')
    tok = transformers.AutoTokenizer.from_pretrained(model_id)
    params = SamplingParams(stop=['hello world'])
    gen_kwargs = params.to_transformers(tok)
    # multi-token phrases must not be folded into eos_token_id (that stops at the
    # phrase's first token); they ride a StopStringCriteria instead.
    assert gen_kwargs['eos_token_id'] == tok.eos_token_id
    criteria = gen_kwargs['stopping_criteria']
    from transformers.generation.stopping_criteria import StopStringCriteria
    assert any(isinstance(c, StopStringCriteria) for c in criteria)
    # integer stops still extend eos_token_id
    gen_kwargs = SamplingParams(stop=[123]).to_transformers(tok)
    assert 123 in gen_kwargs['eos_token_id']