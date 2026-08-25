#!/usr/bin/env python
# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for SGLangSampler: sampling in local mode, and weight sync in Ray mode.

The sampling test is the one that pins down SGLangSampler's event loop arrangement -- the engine has
to be built on the loop that drives it, and every sglang call has to stay on that loop. Anything that
regresses there fails here rather than deadlocking in training.

Usage:
    # sampling only (1 GPU)
    CUDA_VISIBLE_DEVICES=0 python tests/sampler/test_sglang_sampler.py --test sampling

    # weight sync, 1 model GPU + 1 sampler GPU (2 GPUs)
    CUDA_VISIBLE_DEVICES=0,1 python tests/sampler/test_sglang_sampler.py --test weight_sync

    # both
    CUDA_VISIBLE_DEVICES=0,1 python tests/sampler/test_sglang_sampler.py

Environment:
    TEST_MODEL_ID: Model to use (default: Qwen/Qwen2-0.5B-Instruct)
"""

import argparse
import importlib.util
import os
import pytest
import sys

os.environ['NCCL_CUMEM_ENABLE'] = '0'

MODEL_ID = os.environ.get('TEST_MODEL_ID', 'Qwen/Qwen2-0.5B-Instruct')

pytestmark = pytest.mark.skip(
    reason='Heavy sglang e2e test (model load + 1-2 GPUs); not viable on dual-V100 CI, run manually.')


def log(msg):
    print(msg, flush=True)


def wait_result(result):
    """Resolve lazy collect / ray object ref to actual value."""
    if hasattr(result, '_is_lazy_collect') and result._is_lazy_collect:
        return result()
    if hasattr(result, 'wait'):
        return result.wait()
    if callable(result) and hasattr(result, '_get_result'):
        return result()
    return result


def get_model_path():
    """Resolve model_id to a local cache path (for offline environments)."""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        cache = snapshot_download(MODEL_ID, local_files_only=True)
        if cache:
            return cache
    except Exception:
        pass
    return MODEL_ID


# Keep the engine small; these tests care about plumbing, not throughput.
ENGINE_ARGS = {
    'mem_fraction_static': 0.35,
    'max_total_tokens': 4096,
    'log_level': 'error',
}


@pytest.mark.skipif(not importlib.util.find_spec('sglang'), reason='sglang not installed')
@pytest.mark.skipif(not __import__('torch').cuda.is_available(), reason='Requires CUDA')
def test_sglang_sampling():
    """Sampling through the full Sampler layer: trajectories, input features, n>1, logprobs."""
    import twinkle
    from twinkle.data_format import InputFeature, Trajectory
    from twinkle.data_format.sampling import SamplingParams
    from twinkle.sampler import SGLangSampler
    from twinkle.template import Template

    twinkle.initialize(mode='local', nproc_per_node=1)
    model_path = get_model_path()

    sampler = SGLangSampler(model_id=model_path, engine_args=dict(ENGINE_ARGS, enable_memory_saver=True))
    sampler.set_template(Template, model_id=model_path)

    try:
        # ── Trajectory in, decoded text out ───────────────────────────────
        traj = Trajectory(messages=[{'role': 'user', 'content': 'What is the capital of France?'}])
        responses = wait_result(sampler.sample(traj, SamplingParams(max_tokens=24, temperature=0.0)))
        assert len(responses) == 1, f'expected 1 response, got {len(responses)}'
        seq = responses[0].sequences[0]
        assert seq.tokens, 'no tokens generated'
        assert seq.decoded, 'sequence was not decoded'
        assert seq.new_input_feature is not None, 'new_input_feature not populated'
        log(f'  trajectory -> {seq.decoded[:80]!r} (stop_reason={seq.stop_reason})')
        assert 'paris' in seq.decoded.lower(), f'expected Paris in a greedy answer, got {seq.decoded!r}'

        # ── Batch of trajectories ─────────────────────────────────────────
        batch = [
            Trajectory(messages=[{'role': 'user', 'content': 'Name a colour.'}]),
            Trajectory(messages=[{'role': 'user', 'content': 'Name an animal.'}]),
        ]
        responses = wait_result(sampler.sample(batch, SamplingParams(max_tokens=8, temperature=0.0)))
        assert len(responses) == 2, f'expected 2 responses, got {len(responses)}'
        log(f'  batch -> {[r.sequences[0].decoded[:30] for r in responses]}')

        # ── InputFeature (pre-encoded) in ─────────────────────────────────
        tokenizer = sampler.template.tokenizer if hasattr(sampler.template, 'tokenizer') else None
        input_ids = tokenizer.encode('The capital of France is') if tokenizer else [785, 6722, 315, 9625, 374]
        responses = wait_result(
            sampler.sample(InputFeature(input_ids=input_ids), SamplingParams(max_tokens=8, temperature=0.0)))
        assert responses[0].sequences[0].tokens, 'no tokens from InputFeature path'
        assert responses[0].prompt_token_ids == list(input_ids), 'prompt_token_ids should echo the input'
        log(f'  input_feature -> {responses[0].sequences[0].decoded[:60]!r}')

        # ── num_samples > 1 ───────────────────────────────────────────────
        responses = wait_result(
            sampler.sample(
                Trajectory(messages=[{'role': 'user', 'content': 'Tell me a word.'}]),
                SamplingParams(max_tokens=8, temperature=1.0, num_samples=3)))
        n_seqs = len(responses[0].sequences)
        assert n_seqs == 3, f'expected 3 sequences for num_samples=3, got {n_seqs}'
        log(f'  num_samples=3 -> {[s.decoded[:20] for s in responses[0].sequences]}')

        # ── logprobs ──────────────────────────────────────────────────────
        responses = wait_result(
            sampler.sample(
                Trajectory(messages=[{'role': 'user', 'content': 'Say hi.'}]),
                SamplingParams(max_tokens=6, temperature=0.0, logprobs=1)))
        seq = responses[0].sequences[0]
        assert seq.logprobs is not None, 'logprobs requested but not returned'
        assert len(seq.logprobs) == len(seq.tokens), \
            f'logprobs/tokens length mismatch: {len(seq.logprobs)} vs {len(seq.tokens)}'
        token_id, logprob = seq.logprobs[0][0]
        assert isinstance(token_id, int) and isinstance(logprob, float), \
            f'expected (token_id:int, logprob:float), got {seq.logprobs[0][0]!r}'
        assert token_id == seq.tokens[0], 'first logprob entry should describe the first sampled token'
        log(f'  logprobs -> first={seq.logprobs[0][0]}')

        # ── prompt logprobs ───────────────────────────────────────────────
        responses = wait_result(
            sampler.sample(
                InputFeature(input_ids=list(input_ids)), SamplingParams(max_tokens=4, temperature=0.0,
                                                                        prompt_logprobs=1)))
        prompt_logprobs = responses[0].prompt_logprobs
        assert prompt_logprobs is not None, 'prompt_logprobs requested but not returned'
        assert len(prompt_logprobs) == len(input_ids), \
            f'prompt_logprobs length {len(prompt_logprobs)} != prompt length {len(input_ids)}'
        assert prompt_logprobs[0] is None, 'the first prompt token has no predecessor, so no logprob'
        log(f'  prompt_logprobs -> len={len(prompt_logprobs)}, head={prompt_logprobs[:3]}')

        # ── max_tokens=0: score the prompt, generate nothing ──────────────
        responses = wait_result(
            sampler.sample(InputFeature(input_ids=list(input_ids)), SamplingParams(max_tokens=0, temperature=0.0)))
        assert responses[0].sequences[0].tokens == [], 'max_tokens=0 should yield no tokens'
        log('  max_tokens=0 -> no tokens, as expected')

        # ── sleep / wake_up ───────────────────────────────────────────────
        wait_result(sampler.sleep(level=1))
        wait_result(sampler.wake_up())
        responses = wait_result(
            sampler.sample(
                Trajectory(messages=[{'role': 'user', 'content': 'What is the capital of France?'}]),
                SamplingParams(max_tokens=24, temperature=0.0)))
        assert 'paris' in responses[0].sequences[0].decoded.lower(), \
            'sampling should still be correct after sleep/wake_up'
        log('  sleep/wake_up -> still coherent')
    finally:
        sampler.shutdown()

    log('  PASS: sglang sampling')
    return True


@pytest.mark.skipif(not importlib.util.find_spec('sglang'), reason='sglang not installed')
@pytest.mark.skipif(
    not os.environ.get('CUDA_VISIBLE_DEVICES') or len(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')) < 2,
    reason='Requires 2+ GPUs',
)
def test_sglang_weight_sync(model_gpus: int = 1, sampler_gpus: int = 1):
    """Weight sync from TransformersModel to SGLangSampler over the checkpoint engine.

    The sampler starts from random weights (``load_format='dummy'``), so a successful NCCL broadcast is
    observable directly: the output goes from garbage to the real model's answer.
    """
    import twinkle
    from twinkle import DeviceGroup, DeviceMesh
    from twinkle.checkpoint_engine import CheckpointEngineManager
    from twinkle.data_format import Trajectory
    from twinkle.data_format.sampling import SamplingParams
    from twinkle.model.transformers import TransformersModel
    from twinkle.sampler import SGLangSampler
    from twinkle.template import Template

    total_gpus = model_gpus + sampler_gpus
    model_path = get_model_path()

    twinkle.initialize(
        mode='ray',
        nproc_per_node=total_gpus,
        groups=[
            DeviceGroup(name='model', ranks=list(range(model_gpus)), device_type='GPU', gpus_per_worker=1),
            DeviceGroup(
                name='sampler', ranks=list(range(model_gpus, total_gpus)), device_type='GPU', gpus_per_worker=1),
        ],
    )

    model = TransformersModel(
        model_id=model_path,
        device_mesh=DeviceMesh.from_sizes(world_size=model_gpus, dp_size=model_gpus),
        remote_group='model',
    )

    sampler = SGLangSampler(
        model_id=model_path,
        engine_args=dict(ENGINE_ARGS, load_format='dummy'),
        device_mesh=DeviceMesh.from_sizes(world_size=sampler_gpus, dp_size=sampler_gpus),
        remote_group='sampler',
    )
    sampler.set_template(Template, model_id=model_path)

    def do_sample(prompt: str, max_tokens: int = 24) -> str:
        traj = Trajectory(messages=[{'role': 'user', 'content': prompt}])
        responses = wait_result(sampler.sample(traj, SamplingParams(max_tokens=max_tokens, temperature=0.0)))
        for response in responses:
            if response and response.sequences:
                return response.sequences[0].decoded or ''
        return ''

    question = 'What is the capital of France?'

    log('\n--- Sampling BEFORE weight sync (dummy weights) ---')
    text_before = do_sample(question)
    log(f'  Output: {text_before[:100]!r}')

    log('\n--- Syncing weights via CheckpointEngineManager ---')
    manager = CheckpointEngineManager(model=model, sampler=sampler)
    manager.sync_weights()
    wait_result(sampler.reset_prefix_cache())

    log('\n--- Sampling AFTER weight sync (real weights) ---')
    text_after = do_sample(question)
    log(f'  Output: {text_after[:100]!r}')

    outputs_differ = text_before != text_after
    log(f'\n  Outputs differ after sync: {outputs_differ}')
    assert outputs_differ, 'weight sync did not change the output — sync likely failed'
    assert 'paris' in text_after.lower(), \
        f'after syncing real weights the model should answer Paris, got {text_after!r}'

    # A second sync must work too: the checkpoint engine is reused across steps in training.
    manager.sync_weights()
    wait_result(sampler.reset_prefix_cache())
    text_again = do_sample(question)
    assert 'paris' in text_again.lower(), f'second sync broke the weights, got {text_again!r}'
    log('  Second sync OK (engine reuse across steps)')

    sampler.shutdown()
    log('  PASS: sglang weight sync')
    return True


TESTS = {
    'sampling': test_sglang_sampling,
    'weight_sync': test_sglang_weight_sync,
}


def main():
    parser = argparse.ArgumentParser(description='Test SGLangSampler')
    parser.add_argument('--test', choices=list(TESTS) + ['all'], default='all')
    args = parser.parse_args()

    names = list(TESTS) if args.test == 'all' else [args.test]
    failed = []
    for name in names:
        log('=' * 70)
        log(f'TEST: {name}')
        log('=' * 70)
        try:
            TESTS[name]()
        except Exception as e:
            log(f'  FAIL: {name}: {type(e).__name__}: {e}')
            import traceback
            traceback.print_exc()
            failed.append(name)

    log('=' * 70)
    log(f"RESULT: {'PASS' if not failed else 'FAIL ' + ','.join(failed)}")
    log('=' * 70)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
