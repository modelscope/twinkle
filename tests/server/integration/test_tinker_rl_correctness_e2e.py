# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real Tinker RL correctness checks against a running training + vLLM server.

Enable with TWINKLE_TEST_GPU_E2E=1. Set TWINKLE_SERVER_URL,
TWINKLE_TEST_MODEL_ID (public routing alias), and TWINKLE_TEST_MODEL_PATH
(local tokenizer path). The training deployment must have data_world_size=1;
the sampler must return raw logprobs. No datasets or mocks are used.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Requires a running training + vLLM server on real accelerators',
)


def _report(**values):
    print('TINKER_RL_CORRECTNESS ' + json.dumps(values, sort_keys=True), flush=True)


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    """HTTP clients use the external server; do not start the unit-test Ray cluster."""
    yield


@pytest.fixture(autouse=True)
def _reset_canonical_state_actor():
    """Do not clear state owned by the external server between live tests."""
    yield


@pytest.fixture(scope='module')
def live_clients():
    from transformers import AutoTokenizer

    from twinkle import init_tinker_client

    init_tinker_client()
    from tinker import ServiceClient, types

    model_id = os.environ['TWINKLE_TEST_MODEL_ID']
    tokenizer = AutoTokenizer.from_pretrained(os.environ['TWINKLE_TEST_MODEL_PATH'], local_files_only=True)
    prompt_tokens = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'What is 2 + 3? Answer in one short sentence.'}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=False,
    )
    service = ServiceClient(base_url=os.environ['TWINKLE_SERVER_URL'], api_key='EMPTY_TOKEN')
    sampler = service.create_sampling_client(base_model=model_id)
    timeout = float(os.environ.get('TWINKLE_TEST_OPERATION_TIMEOUT', '900'))
    samples = []
    for _ in range(2):
        result = sampler.sample(
            prompt=types.ModelInput.from_ints(prompt_tokens),
            num_samples=1,
            sampling_params=types.SamplingParams(max_tokens=24, temperature=0, top_p=1),
        ).result(timeout=timeout)
        assert len(result.sequences) == 1
        seq = result.sequences[0]
        assert seq.tokens
        assert seq.logprobs is not None and len(seq.logprobs) == len(seq.tokens)
        assert np.isfinite(seq.logprobs).all()
        samples.append(seq)
    assert samples[0].tokens == samples[1].tokens, 'temperature=0 must produce the same greedy completion'
    _report(
        check='sampling',
        model=model_id,
        tokens=len(samples[0].tokens),
        logprobs=len(samples[0].logprobs),
        greedy_repeat_equal=True,
        completion=tokenizer.decode(samples[0].tokens),
        mean_logprob=float(np.mean(samples[0].logprobs)),
    )
    return service, model_id, prompt_tokens, samples[0], timeout


def _datum(prompt_tokens, sample, advantage):
    from tinker import types

    tokens = prompt_tokens + list(sample.tokens)
    prompt_positions = len(prompt_tokens) - 1
    size = len(tokens) - 1

    def tensor(values, dtype):
        return types.TensorData(data=values, dtype=dtype, shape=[size])

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens[:-1]),
        loss_fn_inputs={
            'target_tokens': tensor(tokens[1:], 'int64'),
            'weights': tensor([0.0] * prompt_positions + [1.0] * len(sample.tokens), 'float32'),
            'logprobs': tensor([0.0] * prompt_positions + list(sample.logprobs), 'float32'),
            'advantages': tensor([0.0] * prompt_positions + [advantage] * len(sample.tokens), 'float32'),
        },
    )


def _completion_logprobs(client, datum, timeout):
    result = client.forward([datum], 'cross_entropy').result(timeout=timeout)
    values = np.asarray(result.loss_fn_outputs[0]['logprobs'].tolist(), dtype=np.float64)
    weights = np.asarray(datum.loss_fn_inputs['weights'].tolist())
    assert len(values) >= len(weights)
    values = values[:len(weights)][weights != 0]
    assert values.size and np.isfinite(values).all()
    return values


def test_live_unsupported_losses_are_rejected_without_changing_policy(live_clients):
    from tinker import BadRequestError

    service, model_id, prompt, sample, timeout = live_clients
    client = service.create_lora_training_client(base_model=model_id, rank=16, seed=2026)
    datum = _datum(prompt, sample, -1.0)
    before = _completion_logprobs(client, datum, timeout)
    for loss_fn in ('ppo', 'cispo', 'dro'):
        with pytest.raises(BadRequestError) as exc:
            client.forward_backward([datum], loss_fn).result(timeout=timeout)
        assert exc.value.status_code == 400
        assert f"Unsupported Tinker loss_fn '{loss_fn}'" in str(exc.value)
        after = _completion_logprobs(client, datum, timeout)
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-5)
        _report(check='unsupported_loss', loss_fn=loss_fn, http_status=400, policy_unchanged=True)


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('advantage', [1.0, -1.0])
def test_live_importance_sampling_updates_policy_in_reward_direction(live_clients, batch_size, advantage):
    from tinker import types

    service, model_id, prompt, sample, timeout = live_clients
    client = service.create_lora_training_client(base_model=model_id, rank=16, seed=2026)
    datum = _datum(prompt, sample, advantage)
    before = _completion_logprobs(client, datum, timeout)
    sampler_difference = float(np.mean(np.abs(before - np.asarray(sample.logprobs))))
    assert sampler_difference < 0.05, f'Sampler/trainer logprob mismatch: {sampler_difference}'
    result = client.forward_backward([datum] * batch_size, 'importance_sampling').result(timeout=timeout)
    assert len(result.loss_fn_outputs) == batch_size
    assert np.isfinite(float(result.metrics['loss:mean']))
    client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=timeout)
    after = _completion_logprobs(client, datum, timeout)
    delta = float(np.mean(after) - np.mean(before))
    _report(
        check='reward_direction',
        batch_size=batch_size,
        advantage=advantage,
        before=float(np.mean(before)),
        after=float(np.mean(after)),
        delta=delta,
        sampler_trainer_mean_abs_difference=sampler_difference,
        loss=float(result.metrics['loss:mean']),
    )
    assert advantage * delta > 1e-4, f'Wrong or negligible policy update: advantage={advantage}, delta={delta}'


def test_live_dpo_still_requires_pairs(live_clients):
    from tinker import types, UnprocessableEntityError

    service, model_id, prompt, sample, timeout = live_clients
    client = service.create_lora_training_client(base_model=model_id, rank=16, seed=2026)
    datum = _datum(prompt, sample, 1.0)
    reference = client.forward([datum], 'cross_entropy').result(timeout=timeout)
    datum.loss_fn_inputs['ref_logps'] = reference.loss_fn_outputs[0]['logprobs']
    with pytest.raises(UnprocessableEntityError) as exc:
        client.forward_backward([datum], 'importance_sampling').result(timeout=timeout)
    assert exc.value.status_code == 422
    assert 'complete chosen/rejected pairs' in str(exc.value)
    result = client.forward_backward([datum, datum], 'importance_sampling').result(timeout=timeout)
    assert len(result.loss_fn_outputs) == 2
    assert np.isfinite(float(result.metrics['loss:mean']))
    client.optim_step(types.AdamParams(learning_rate=1e-4)).result(timeout=timeout)
    _report(check='dpo_pairs', single_status=422, paired_batch_size=2, paired_loss=float(result.metrics['loss:mean']))
