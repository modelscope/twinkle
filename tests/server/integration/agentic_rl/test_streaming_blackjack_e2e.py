# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run the actual streaming cookbook against an existing Model/Sampler server.

Run on the server host so the saved adapter files are accessible. Configure
TWINKLE_SERVER_URL, TWINKLE_SERVER_TOKEN and TWINKLE_TEMPLATE_MODEL_ID, then run:

    TWINKLE_TEST_GPU_E2E=1 pytest --noconftest -q -s tests/server/integration/agentic_rl/test_streaming_blackjack_e2e.py

Use --noconftest: the shared server fixtures start and reset a unit-test Ray
runtime, whereas this test only connects to the existing server over HTTP.

This checks a finite Blackjack training/publish flow, not learning improvement
or AgentENV isolation. A run with no informative group is not a passing E2E.
"""
import importlib.util
import math
import os
import threading
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
    reason='Requires a real Twinkle Model/Sampler server and its checkpoint filesystem',
)


def test_streaming_blackjack_trains_and_samples_published_policy(monkeypatch, record_property):
    import torch
    from safetensors.torch import load_file

    from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel
    from twinkle_client.sampler.vllm_sampler import vLLMSampler

    path = Path(__file__).parents[4] / 'cookbook/client/async_rl/streaming_multi_turn_grpo.py'
    spec = importlib.util.spec_from_file_location('streaming_blackjack_e2e', path)
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    # Leave an update beyond the first bufferful so stopping cannot skip publication.
    recipe.MAX_STEPS = recipe.BUFFER_GROUPS + 1
    saved, requests, training = [], [], []
    steps = 0
    lock = threading.Lock()
    original_save = MultiLoraTransformersModel.save
    original_sample = vLLMSampler.sample
    original_backward = MultiLoraTransformersModel.forward_backward
    original_step = MultiLoraTransformersModel.clip_grad_and_step

    def save(model, *args, **kwargs):
        result = original_save(model, *args, **kwargs)
        saved.append(result)
        return result

    def sample(sampler, *args, **kwargs):
        started = time.monotonic()
        result = original_sample(sampler, *args, **kwargs)
        with lock:
            requests.append((kwargs.get('adapter_uri'), started, time.monotonic()))
        return result

    def backward(model, *args, **kwargs):
        for feature, logps in zip(kwargs['inputs'], kwargs['old_logps'], strict=True):
            labels, mask = feature['labels'], feature['completion_mask']
            assert len(feature['input_ids']) == len(labels) == len(mask)
            assert sum(label != -100 and bool(scored) for label, scored in zip(labels, mask, strict=True)) == len(logps)
            assert logps and all(math.isfinite(value) for value in logps)
        started = time.monotonic()
        result = original_backward(model, *args, **kwargs)
        with lock:
            training.append((started, time.monotonic()))
        return result

    def step(model, *args, **kwargs):
        nonlocal steps
        result = original_step(model, *args, **kwargs)
        steps += 1
        return result

    monkeypatch.setattr(MultiLoraTransformersModel, 'save', save)
    monkeypatch.setattr(vLLMSampler, 'sample', sample)
    monkeypatch.setattr(MultiLoraTransformersModel, 'forward_backward', backward)
    monkeypatch.setattr(MultiLoraTransformersModel, 'clip_grad_and_step', step)
    recipe.train()

    assert steps > 0, 'No nonzero-advantage group completed an optimizer update'
    assert steps <= recipe.MAX_STEPS
    assert len(saved) >= 2, 'No trained policy was published'
    initial_uri = saved[0].twinkle_path
    published = {checkpoint.twinkle_path for checkpoint in saved[1:]}
    assert initial_uri not in published
    assert len(published) == len(saved) - 1
    new_requests = [request for request in requests if request[0] in published]
    assert new_requests, 'No successful sampling call used a published policy'
    assert any(start < end_train and start_train < end
               for _, start, end in requests for start_train, end_train in training), 'No HTTP sampling/training overlap'

    assert all(checkpoint.checkpoint_dir for checkpoint in saved[:2]), 'Server did not return checkpoint directories'
    paths = [Path(checkpoint.checkpoint_dir) / 'adapter_model.safetensors' for checkpoint in saved[:2]]
    assert all(path.is_file() for path in paths), 'Run on the server host with access to its checkpoint files'
    before, after = (load_file(str(path)) for path in paths)
    assert before and before.keys() == after.keys()
    assert all(torch.isfinite(value).all() for weights in (before, after) for value in weights.values())
    changed = sum(not torch.equal(before[key], after[key]) for key in before)
    assert changed > 0, 'Published adapter weights did not change'
    record_property('optimizer_steps', steps)
    record_property('sample_calls', len(requests))
    record_property('new_policy_sample_calls', len(new_requests))
    record_property('changed_tensors', changed)
