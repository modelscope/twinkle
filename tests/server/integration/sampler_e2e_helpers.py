"""Shared clients and model fixtures for real sampler E2E tests."""
from __future__ import annotations

import concurrent.futures
import json
import math
import os
import threading
import time
from pathlib import Path

import pytest

@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    yield



@pytest.fixture(autouse=True)
def _reset_canonical_state_actor():
    yield



def _report(**values):
    print('SAMPLER_CONCURRENCY ' + json.dumps(values, sort_keys=True), flush=True)



@pytest.fixture(scope='module')
def clients():
    from transformers import AutoTokenizer
    from twinkle import init_tinker_client, init_twinkle_client
    init_tinker_client()
    from tinker import ServiceClient
    model_id = os.environ['TWINKLE_TEST_MODEL_ID']
    model_path = os.environ['TWINKLE_TEST_MODEL_PATH']
    url = os.environ['TWINKLE_SERVER_URL']
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    service = ServiceClient(base_url=url, api_key='EMPTY_TOKEN')
    native = init_twinkle_client(base_url=url, api_key='EMPTY_TOKEN').sampler(model_id)
    return {
        'tokenizer': tokenizer, 'service': service, 'model_id': model_id, 'model_path': model_path,
        'url': url, 'tinker': service.create_sampling_client(base_model=model_id), 'twinkle': native,
        'single_throughput': {},
    }



def _prompt(clients, text):
    return clients['tokenizer'].apply_chat_template(
        [{'role': 'user', 'content': text}], tokenize=True, add_generation_prompt=True,
        enable_thinking=False, return_dict=False)



def _sample(clients, dialect, ids, max_tokens=128, *, adapter=None, num_samples=1, temperature=0):
    if dialect == 'tinker':
        from tinker import types
        result = (adapter or clients['tinker']).sample(
            prompt=types.ModelInput.from_ints(ids), num_samples=num_samples,
            sampling_params=types.SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=1),
        ).result(timeout=300)
        sequences = [(list(seq.tokens), list(seq.logprobs)) for seq in result.sequences]
    else:
        result = clients['twinkle'].sample(
            inputs=[{'input_ids': ids}], num_samples=num_samples, adapter_uri=adapter,
            sampling_params={'max_tokens': max_tokens, 'temperature': temperature, 'top_p': 1, 'logprobs': 1})
        assert result[0].prompt_token_ids == ids
        sequences = [(list(seq.tokens), [position[0][1] for position in seq.logprobs])
                     for seq in result[0].sequences]
    assert len(sequences) == num_samples
    for tokens, logprobs in sequences:
        assert tokens and len(tokens) == len(logprobs)
        assert all(math.isfinite(value) for value in logprobs)
    return sequences



def _parallel(jobs):
    barrier = threading.Barrier(len(jobs))
    def run(job):
        barrier.wait(timeout=30)
        return job()
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        results = list(pool.map(run, jobs))
    return results, time.perf_counter() - started



@pytest.fixture(scope='module')
def adapters(clients):
    import torch
    from peft import LoraConfig
    from safetensors import safe_open
    from safetensors.torch import save_file
    projections = {}
    for shard in Path(clients['model_path']).glob('*.safetensors'):
        with safe_open(str(shard), framework='pt', device='cpu') as weights:
            for name in weights.keys():
                if name.endswith('.self_attn.q_proj.weight'):
                    projections[name[:-len('.weight')]] = weights.get_slice(name).get_shape()
    if not projections:
        pytest.skip('The cached model has no q_proj layers for this LoRA fixture')
    root = Path(os.environ['TWINKLE_TEST_ARTIFACT_DIR']) / 'lora-isolation'
    paths = []
    rank = 8
    for index in range(2):
        path = root / str(index)
        path.mkdir(parents=True, exist_ok=True)
        LoraConfig(r=rank, lora_alpha=rank, target_modules=['q_proj'], bias='none',
                   task_type='CAUSAL_LM', inference_mode=True,
                   base_model_name_or_path=clients['model_id']).save_pretrained(path)
        generator = torch.Generator().manual_seed(100 + index)
        weights = {}
        for name, (output_size, input_size) in projections.items():
            prefix = 'base_model.model.' + name
            weights[prefix + '.lora_A.weight'] = torch.randn(rank, input_size, generator=generator) * 0.05
            weights[prefix + '.lora_B.weight'] = torch.randn(output_size, rank, generator=generator) * 0.05
        save_file(weights, str(path / 'adapter_model.safetensors'))
        paths.append(str(path))
    return paths



@pytest.fixture(scope='module')
def full_checkpoints(clients):
    """Two complete HF checkpoints with different, observable model weights."""
    import torch
    from safetensors.torch import load_file, save_file
    source = Path(clients['model_path'])
    target = Path(os.environ['TWINKLE_TEST_ARTIFACT_DIR']) / 'full-checkpoint-B'
    target.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file() and not (target / file.name).exists():
            (target / file.name).symlink_to(file)
    index = json.loads((source / 'model.safetensors.index.json').read_text())
    shard = index['weight_map']['model.embed_tokens.weight']
    output = target / shard
    if output.is_symlink():
        weights = load_file(str(source / shard), device='cpu')
        weights['model.embed_tokens.weight'] = torch.roll(weights['model.embed_tokens.weight'], 257, 0).contiguous()
        output.unlink()
        save_file(weights, str(output), metadata={'format': 'pt'})
    return [str(source), str(target)]



def _checkpoint_handles(clients, dialect, paths):
    if dialect == 'tinker':
        return [clients['service'].create_sampling_client(base_model=clients['model_id'], model_path=path)
                for path in paths]
    return paths



def _assert_adapter_isolation(clients, adapters, dialect):
    handles = ([None] + [clients['service'].create_sampling_client(base_model=clients['model_id'], model_path=path)
                        for path in adapters] if dialect == 'tinker' else [None, *adapters])
    ids = _prompt(clients, 'Choose one word: apple, banana, orange, purple.')
    solo = [_sample(clients, dialect, ids, 8, adapter=handle)[0] for handle in handles]
    # Default vLLM kernels do not guarantee batch-invariant continuations or
    # logprobs. Homogeneous PPU batches also diverge after this stable prefix.
    # Require distinct adapter fingerprints, then check routing against them;
    # _sample still validates every generated token's finite, aligned logprob.
    fingerprints = [tuple(tokens[:2]) for tokens, _logprobs in solo]
    assert all(len(fingerprint) == 2 for fingerprint in fingerprints)
    assert len(set(fingerprints)) == 3, 'the adapters must change the result, rather than silently falling back to base'
    results, elapsed = _parallel([
        lambda i=i: _sample(clients, dialect, ids, 8, adapter=handles[i % 3]) for i in range(24)])
    for i, result in enumerate(results):
        tokens, _logprobs = result[0]
        assert tuple(tokens[:2]) == fingerprints[i % 3]
    _report(check='lora_isolation', dialect=dialect, requests=24, variants=3, seconds=elapsed,
            fingerprints=fingerprints)


def _real_sampler_actors():
    import ray
    from ray.util.state import list_actors
    ray.init(address=os.environ['RAY_ADDRESS'], namespace='default', ignore_reinit_error=True,
             log_to_driver=False)
    actors = list_actors(address=os.environ['TWINKLE_TEST_RAY_DASHBOARD_URL'],
                         filters=[('class_name', '=', os.environ.get('TWINKLE_TEST_SAMPLER_CLASS', 'VLLMSamplerTQ')),
                                  ('state', '=', 'ALIVE')],
                         detail=True)
    assert len(actors) == int(os.environ.get('TWINKLE_TEST_DP', '1'))
    assert all(actor.state == 'ALIVE' for actor in actors)
    return [ray.get_actor(actor.name, namespace='default') for actor in actors]
