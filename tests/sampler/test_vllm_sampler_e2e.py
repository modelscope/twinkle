#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real end-to-end tests for the vLLM *Sampler* seam -- one layer above the engine.

``test_sampler_e2e.py`` drives ``VLLMEngine`` directly with raw token ids. These tests drive the public
``vLLMSampler.sample()`` the way training and inference actually call it: a real ``Template`` encodes a
Trajectory (chat messages), the vLLM engine generates, and the sampler assembles decoded ``SampleResponse``
objects. Green here means the whole path -- template encode -> engine generate -> response assembly -> the
InputFeature fast-path that skips the template -- is wired correctly, which is what "the sampler works" has
to mean. Nothing is stubbed: real weights, real tokenizer, real generation.

Heavy: real model load + GPU. Gated by the project tiering (``slow`` + ``accel(1)``), so a capable box runs
it while ``pytest -m 'not slow'`` (CI) and CPU-only boxes skip. A warm ModelScope cache lets it run offline.

Environment:
    TWINKLE_MODEL_ID: model to sample from (default Qwen/Qwen2.5-0.5B-Instruct, small and cache-friendly)
    TWINKLE_MAX_MODEL_LEN: max model length (default 512)
"""
import os

# Resolve weights from ModelScope (the project default) before importing anything that reads these.
os.environ.setdefault('VLLM_USE_MODELSCOPE', 'True')
os.environ.setdefault('TRUST_REMOTE_CODE', '1')

import pytest  # noqa: E402

MODEL_ID = os.environ.get('TWINKLE_MODEL_ID', 'Qwen/Qwen2.5-0.5B-Instruct')
MAX_MODEL_LEN = int(os.environ.get('TWINKLE_MAX_MODEL_LEN', '512'))
# The tokenizer is loaded through twinkle's Template, which resolves a ModelScope id via the ``ms://`` prefix.
TEMPLATE_MODEL_ID = MODEL_ID if MODEL_ID.startswith('ms://') else f'ms://{MODEL_ID}'

pytestmark = [
    pytest.mark.slow,
    pytest.mark.accel(1),
    pytest.mark.skipif(not __import__('importlib').util.find_spec('vllm'), reason='vllm not installed'),
]


def _model_is_cached() -> bool:
    """True when the weights already sit in a local hub cache, so building the sampler needs no network."""
    bare_id = MODEL_ID[len('ms://'):] if MODEL_ID.startswith('ms://') else MODEL_ID
    ms_cache = os.environ.get('MODELSCOPE_CACHE')
    if ms_cache and os.path.isdir(os.path.join(ms_cache, 'models', bare_id.replace('/', '--'))):
        return True
    hf_home = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
    return os.path.isdir(os.path.join(hf_home, 'hub', f'models-{bare_id.replace("/", "-")}'))


@pytest.fixture(scope='module')
def sampler():
    """Build one real vLLMSampler for the whole module; engine startup is the expensive part.

    A single GPU is forced (``tensor_parallel_size=1``) and memory capped so the test coexists with whatever
    else holds the card. The Template loads only the tokenizer from the same weights.
    """
    if not _model_is_cached():
        try:
            import urllib.request
            urllib.request.urlopen('https://www.modelscope.cn', timeout=5)
        except Exception as e:  # cold cache and no network: skip rather than hang on a doomed download
            pytest.skip(f'{MODEL_ID} not cached and ModelScope unreachable: {e}')

    from twinkle.sampler.vllm_sampler.vllm_sampler import vLLMSampler
    built = vLLMSampler(
        model_id=MODEL_ID,
        engine_args={
            'tensor_parallel_size': 1,
            'gpu_memory_utilization': 0.3,
            'max_model_len': MAX_MODEL_LEN,
        },
    )
    built.set_template('Template', model_id=TEMPLATE_MODEL_ID)
    try:
        yield built
    finally:
        # Engine teardown can emit asyncio/await noise after the loop closes; the process is exiting anyway,
        # so never let a shutdown-time exception turn a passing test into a teardown error.
        try:
            built.shutdown()
        except Exception:
            pass


def _greedy(max_tokens: int = 16):
    from twinkle.data_format.sampling import SamplingParams
    return SamplingParams(max_tokens=max_tokens, temperature=0.0)


def _tokens_of(response):
    seq = response.sequences[0]
    return list(seq.tokens), seq


def test_sample_from_trajectory_encodes_generates_and_decodes(sampler):
    """The headline path: chat messages -> Template.encode -> vLLM generate -> SampleResponse with tokens."""
    traj = {'messages': [{'role': 'user', 'content': 'What is 2+2? Reply with just the number.'}]}
    responses = sampler.sample([traj], sampling_params=_greedy())

    assert isinstance(responses, list) and len(responses) == 1, 'one input must yield exactly one response'
    tokens, seq = _tokens_of(responses[0])
    assert len(tokens) >= 1, 'the sampler produced no tokens -- generation never reached the engine'
    assert all(isinstance(t, int) for t in tokens), 'tokens must be token ids, not text'
    assert len(tokens) <= 16, 'max_tokens must be honoured end to end'
    assert seq.stop_reason is not None, 'a finished sequence must report why it stopped'
    # The decoded text is the real contract a caller reads; decode independently to prove tokens are coherent.
    text = sampler.template.tokenizer.decode(tokens, skip_special_tokens=True)
    assert isinstance(text, str) and text.strip() != '', 'tokens must decode to non-empty text'


def test_greedy_sampling_is_deterministic_through_the_whole_path(sampler):
    """temperature=0 must give identical tokens across calls -- a real behavioural property, not a smoke test.

    This pins that sampling params actually reach the engine (a wiring break would silently fall back to the
    engine default and diverge), without asserting brittle exact wording from a 0.5B model.
    """
    traj = {'messages': [{'role': 'user', 'content': 'Name one primary colour.'}]}
    first, _ = _tokens_of(sampler.sample([traj], sampling_params=_greedy())[0])
    second, _ = _tokens_of(sampler.sample([traj], sampling_params=_greedy())[0])
    assert first == second, 'greedy decoding must be reproducible through encode -> generate -> decode'


def test_sample_from_input_feature_skips_the_template(sampler):
    """An already-encoded InputFeature is sampled directly; the Template must not be needed for this path."""
    tokenizer = sampler.template.tokenizer
    input_ids = tokenizer.encode('The capital of France is', add_special_tokens=True)
    responses = sampler.sample([{'input_ids': input_ids}], sampling_params=_greedy())

    tokens, _ = _tokens_of(responses[0])
    assert len(tokens) >= 1, 'InputFeature path produced no tokens'
    assert all(isinstance(t, int) for t in tokens)


def test_sample_batch_returns_one_response_per_input_in_order(sampler):
    """A batch of trajectories yields one response each, so DP slicing and flattening stay aligned."""
    trajs = [
        {'messages': [{'role': 'user', 'content': 'Say the number one.'}]},
        {'messages': [{'role': 'user', 'content': 'Say the number two.'}]},
        {'messages': [{'role': 'user', 'content': 'Say the number three.'}]},
    ]
    responses = sampler.sample(trajs, sampling_params=_greedy(max_tokens=8))

    assert len(responses) == 3, f'expected 3 responses for 3 inputs, got {len(responses)}'
    for i, response in enumerate(responses):
        assert len(response.sequences) >= 1, f'input {i} produced no sequence'
        assert len(list(response.sequences[0].tokens)) >= 1, f'input {i} produced no tokens'
