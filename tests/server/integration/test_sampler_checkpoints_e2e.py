"""Real training and named-checkpoint sampling against an external server.

Use the same external-server environment as test_tinker_rl_correctness_e2e.
"""
import json
import os

import numpy as np
import pytest

from tests.server.integration.test_tinker_rl_correctness_e2e import _completion_logprobs, _datum

pytestmark = pytest.mark.skipif(os.environ.get('TWINKLE_TEST_GPU_E2E') != '1',
                                reason='Requires a real trainer and vLLM sampler')


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    yield


@pytest.fixture(autouse=True)
def _reset_canonical_state_actor():
    yield


def test_named_save_and_cached_overwrite():
    from transformers import AutoTokenizer
    from twinkle import init_tinker_client

    init_tinker_client()
    from tinker import ServiceClient, types

    service = ServiceClient(base_url=os.environ['TWINKLE_SERVER_URL'], api_key='EMPTY_TOKEN')
    model_id = os.environ['TWINKLE_TEST_MODEL_ID']
    timeout = float(os.environ.get('TWINKLE_TEST_OPERATION_TIMEOUT', '900'))
    tokenizer = AutoTokenizer.from_pretrained(os.environ['TWINKLE_TEST_MODEL_PATH'], local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'What is 2 + 3? Answer in one short sentence.'}],
        tokenize=True, add_generation_prompt=True, enable_thinking=False, return_dict=False)
    client = service.create_lora_training_client(base_model=model_id, rank=16, seed=2026)

    def sample(sampler):
        result = sampler.sample(prompt=types.ModelInput.from_ints(prompt), num_samples=1,
                                sampling_params=types.SamplingParams(max_tokens=24, temperature=0, top_p=1)
                                ).result(timeout=timeout)
        seq = result.sequences[0]
        assert seq.tokens and len(seq.tokens) == len(seq.logprobs)
        assert np.isfinite(seq.logprobs).all()
        return seq

    def same(actual, expected):
        assert actual.tokens == expected.tokens
        np.testing.assert_allclose(actual.logprobs, expected.logprobs, rtol=0, atol=1e-5)

    x = client.save_weights_for_sampler(name='x').result(timeout=timeout).path
    assert x.endswith('/sampler_weights/x')
    sampler_x = service.create_sampling_client(base_model=model_id, model_path=x)
    first = sample(sampler_x)
    same(sample(sampler_x), first)
    datum = _datum(prompt, first, 1.0)
    before = _completion_logprobs(client, datum, timeout)
    losses = []
    for _ in range(3):
        result = client.forward_backward([datum], 'cross_entropy').result(timeout=timeout)
        losses.append(float(result.metrics['loss:mean']))
        assert np.isfinite(losses[-1])
        client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=timeout)
    after = _completion_logprobs(client, datum, timeout)
    assert float(np.mean(after - before)) > 1e-4

    y = client.save_weights_for_sampler(name='y').result(timeout=timeout).path
    assert y.endswith('/sampler_weights/y')
    sampler_y = service.create_sampling_client(base_model=model_id, model_path=y)
    trained = sample(sampler_y)
    same(sample(sampler_x), first)
    live = client.save_weights_and_get_sampling_client()
    same(sample(live), trained)
    same(sample(sampler_x), first)
    same(sample(sampler_y), trained)

    overwritten = client.save_weights_for_sampler(name='x').result(timeout=timeout).path
    assert overwritten == x
    refreshed = sample(sampler_x)
    same(refreshed, trained)
    assert refreshed.tokens != first.tokens or not np.allclose(refreshed.logprobs, first.logprobs, atol=1e-4)
    print('SAMPLER_CHECKPOINT_E2E ' + json.dumps({
        'losses': losses, 'trainer_logprob_before': float(np.mean(before)),
        'trainer_logprob_after': float(np.mean(after)), 'named_paths': [x, y],
        'named_versions_survive_live_save': True, 'cached_overwrite_reloaded': True,
        'tokens_before': first.tokens, 'tokens_after': refreshed.tokens,
        'sampler_logprob_before': float(np.mean(first.logprobs)),
        'sampler_logprob_after': float(np.mean(refreshed.logprobs)),
    }), flush=True)
