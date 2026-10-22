"""Real Tinker and Twinkle concurrency, weight isolation and recovery."""
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
    """Use the external server; do not start the parent conftest's local Ray."""
    yield


@pytest.fixture(autouse=True)
def _reset_canonical_state_actor():
    """Keep external server state intact across E2E requests."""
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
                         filters=[('class_name', '=', os.environ.get('TWINKLE_TEST_SAMPLER_CLASS', 'ServerVLLMSampler')),
                                  ('state', '=', 'ALIVE')],
                         detail=True)
    assert len(actors) == int(os.environ.get('TWINKLE_TEST_DP', '1'))
    assert all(actor.state == 'ALIVE' for actor in actors)
    return [ray.get_actor(actor.name, namespace='default') for actor in actors]


pytestmark = pytest.mark.skipif(os.environ.get('TWINKLE_TEST_GPU_E2E', '0') != '1',
                                reason='Requires a running real vLLM sampler server')


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
def test_both_dialects_batch_independent_requests(clients, count, dialect):
    ids = _prompt(clients, 'Write a detailed story of at least 300 words about an astronaut. Keep writing.')
    _sample(clients, dialect, ids, 8)
    if count != 1 and dialect not in clients['single_throughput']:
        started = time.perf_counter()
        single = _sample(clients, dialect, ids)
        clients['single_throughput'][dialect] = len(single[0][0]) / (time.perf_counter() - started)
    results, elapsed = _parallel([lambda: _sample(clients, dialect, ids)] * count)
    tokens = sum(len(result[0][0]) for result in results)
    throughput = tokens / elapsed
    _report(check='benchmark', dialect=dialect, count=count, tokens=tokens, seconds=elapsed,
            tokens_per_second=throughput)
    if count == 1:
        clients['single_throughput'][dialect] = throughput
    else:
        assert throughput > 1.5 * clients['single_throughput'][dialect], 'independent requests still appear serial'


@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
def test_concurrent_request_mapping(clients, dialect):
    words = ['apple', 'banana', 'orange', 'purple', 'yellow', 'green', 'blue', 'red']
    prompts = [_prompt(clients, 'Reply with exactly this word and nothing else: ' + word) for word in words]
    solo = [_sample(clients, dialect, ids, 16)[0][0] for ids in prompts]
    jobs = [lambda i=i: _sample(clients, dialect, prompts[i % len(prompts)], 16) for i in range(32)]
    results, elapsed = _parallel(jobs)
    assert all(result[0][0] == solo[i % len(solo)] for i, result in enumerate(results))
    _report(check='mapping', dialect=dialect, count=32, unique_prompts=8, seconds=elapsed, solo_equal=True)


def test_mixed_dialects_and_multiple_samples(clients):
    ids = _prompt(clients, 'Reply with exactly this word and nothing else: apple')
    solo = _sample(clients, 'tinker', ids, 16)[0][0]
    jobs = [lambda i=i: _sample(clients, 'tinker' if i % 2 else 'twinkle', ids, 16)
            for i in range(16)]
    results, elapsed = _parallel(jobs)
    assert all(tokens == solo for result in results for tokens, _logprobs in result)
    _report(check='mixed', requests=16, sequences=16, seconds=elapsed, solo_equal=True)
    # vLLM requires n=1 for greedy sampling. Multiple completions exercise the
    # stochastic contract: count, token/logprob alignment and finite logprobs.
    results, elapsed = _parallel([
        lambda i=i: _sample(clients, 'tinker' if i % 2 else 'twinkle', ids, 16,
                            num_samples=4, temperature=0.7) for i in range(16)])
    assert sum(len(result) for result in results) == 64
    _report(check='mixed_multiple_samples', requests=16, sequences=64, seconds=elapsed)


@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
def test_base_and_two_lora_adapters_are_isolated(clients, adapters, dialect):
    _assert_adapter_isolation(clients, adapters, dialect)


def test_full_checkpoints_are_isolated_across_dialects(clients, full_checkpoints):
    ids = _prompt(clients, 'Reply with exactly this word and nothing else: apple')
    tinker = _checkpoint_handles(clients, 'tinker', full_checkpoints)
    solo = [_sample(clients, 'tinker', ids, 8, adapter=handle)[0][0] for handle in tinker]
    assert solo[0] != solo[1]
    try:
        jobs = [lambda i=i: _sample(clients, 'tinker' if i % 2 else 'twinkle', ids, 8,
                                   adapter=tinker[i % 2] if i % 2 else full_checkpoints[i % 2]) for i in range(8)]
        results, elapsed = _parallel(jobs)
        assert all(result[0][0] == solo[i % 2] for i, result in enumerate(results))
        _report(check='full_checkpoint_mixed_isolation', requests=8, seconds=elapsed, a=solo[0], b=solo[1])
    finally:
        _sample(clients, 'twinkle', ids, 2, adapter=full_checkpoints[0])


@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
def test_invalid_generation_does_not_disable_sampler(clients, dialect):
    ids = _prompt(clients, 'Reply with exactly this word and nothing else: apple')
    with pytest.raises(Exception):
        _sample(clients, dialect, [10**9], 4)
    assert _sample(clients, dialect, ids, 4)[0][0]
    _report(check='failure_recovery', dialect=dialect)


@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
def test_invalid_checkpoint_does_not_disable_sampler(clients, tmp_path, dialect):
    ids = _prompt(clients, 'Reply with exactly this word and nothing else: apple')
    corrupt = tmp_path / 'corrupt'
    corrupt.mkdir()
    (corrupt / 'model.safetensors').write_text('invalid checkpoint')
    for path in (str(tmp_path), str(tmp_path / 'missing'), str(corrupt)):
        handle = _checkpoint_handles(clients, dialect, [path])[0]
        with pytest.raises(Exception):
            _sample(clients, dialect, ids, 2, adapter=handle)
        assert _sample(clients, dialect, ids, 2)[0][0]
    _report(check='invalid_checkpoint_recovery', dialect=dialect)


def test_tinker_tokens_do_not_use_native_template(clients):
    import ray
    actors = _real_sampler_actors()
    ids = _prompt(clients, 'Reply with exactly this word and nothing else: apple')
    expected = _sample(clients, 'tinker', ids, 2)[0][0]

    def install(actor):
        class OtherClientTemplate:
            def get_vllm_input_ids(self, *args):
                raise AssertionError('Tinker used the native template')
            def concat_input_feature(self, *args):
                raise AssertionError('Tinker built unused training features')
            def decode(self, *args):
                raise AssertionError('Tinker decoded unused text')
        actor._e2e_template = actor.template
        actor.template = OtherClientTemplate()

    def restore(actor):
        actor.template = actor._e2e_template
        del actor._e2e_template

    ray.get([actor.__ray_call__.remote(install) for actor in actors], timeout=10)
    try:
        assert _sample(clients, 'tinker', ids, 2)[0][0] == expected
    finally:
        ray.get([actor.__ray_call__.remote(restore) for actor in actors], timeout=10)
        ray.shutdown()


@pytest.mark.parametrize('dialect', ['tinker', 'twinkle'])
def test_sampling_timeout_keeps_backend_owned_without_blocking_readers(clients, dialect):
    import ray
    from ray.util.state import list_actors
    _real_sampler_actors()
    replicas = [a for a in list_actors(address=os.environ['TWINKLE_TEST_RAY_DASHBOARD_URL'], detail=True)
                if a.state == 'ALIVE' and a.class_name.startswith('ServeReplica') and 'SamplerManagement' in a.class_name]
    assert len(replicas) == 1
    replica = ray.get_actor(replicas[0].name, namespace='serve')

    def install(actor):
        service = actor._replica_impl._user_callable_wrapper._callable
        service._e2e_timeout = service.task_queue_config.execution_timeout
        service._e2e_sample = service.sampler.sample
        service._e2e_release = threading.Event()
        service._e2e_entered = threading.Event()
        service.task_queue_config.execution_timeout = .1
        original, entered, release = service._e2e_sample, service._e2e_entered, service._e2e_release
        def held(*args, **kwargs):
            result = original(*args, **kwargs)
            entered.set()
            assert release.wait(60)
            return result
        service.sampler.sample = held

    def allow_next(actor):
        service = actor._replica_impl._user_callable_wrapper._callable
        service.task_queue_config.execution_timeout = service._e2e_timeout
        service.sampler.sample = service._e2e_sample
        return service._e2e_entered.is_set(), service._compute_worker._detached_reserved

    def restore(actor):
        service = actor._replica_impl._user_callable_wrapper._callable
        service._e2e_release.set()
        service.task_queue_config.execution_timeout = service._e2e_timeout
        service.sampler.sample = service._e2e_sample
        for name in ('_e2e_timeout', '_e2e_sample', '_e2e_release', '_e2e_entered'):
            delattr(service, name)

    ids = _prompt(clients, 'Reply with exactly this word and nothing else: apple')
    ray.get(replica.__ray_call__.remote(install), timeout=10)
    try:
        with pytest.raises(Exception, match='timed out|timeout'):
            _sample(clients, dialect, ids, 2)
        deadline = time.monotonic() + 30
        while True:
            entered, reserved = ray.get(replica.__ray_call__.remote(allow_next), timeout=10)
            if entered:
                break
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert reserved == 1
        assert _sample(clients, dialect, ids, 2)[0][0]
        _report(check='sampling_timeout_recovery', dialect=dialect, retained_capacity=reserved)
    finally:
        ray.get(replica.__ray_call__.remote(restore), timeout=10)
        ray.shutdown()


def test_data_plane_http_generation_and_failure_recovery(clients):
    from twinkle_client.data_plane import DataPlaneClient
    store = DataPlaneClient(transport=clients['twinkle']._transport)
    ids = _prompt(clients, 'Reply with exactly this word and nothing else: apple')
    expected = _sample(clients, 'twinkle', ids, 2)[0][0]
    with pytest.raises(Exception):
        clients['twinkle'].sample_to_data_plane(
            inputs=[{'input_ids': [10**9]}], sampling_params={'max_tokens': 2, 'logprobs': 1})

    def generate_and_fetch():
        ref = clients['twinkle'].sample_to_data_plane(
            inputs=[{'input_ids': ids}], num_samples=2, group_ids=['e2e-group'], policy_version=7,
            sampling_params={'max_tokens': 2, 'temperature': .7, 'logprobs': 1})
        try:
            batch = store.get_batch(ref)
            assert ref.size == 2 and len(batch.rows) == len(batch.tags) == 2
            assert all(0 < len(row['tokens']) <= 2 for row in batch.rows)
            assert all(len(row['sampled_logprobs']) == len(row['tokens'])
                       and all(math.isfinite(value) for value in row['sampled_logprobs']) for row in batch.rows)
            assert [tag['generation_idx'] for tag in batch.tags] == [0, 1]
            assert all(tag['group_id'] == 'e2e-group' and tag['rollout_policy_version'] == 7 for tag in batch.tags)
            return len(batch.rows)
        finally:
            store.release(ref)

    results, elapsed = _parallel([
        generate_and_fetch, generate_and_fetch,
        lambda: _sample(clients, 'tinker', ids, 2)[0][0],
        lambda: _sample(clients, 'twinkle', ids, 2)[0][0],
    ])
    assert results == [2, 2, expected, expected]
    _report(check='data_plane_http', requests=4, sequences=6, recovered=True, seconds=elapsed)


def test_backend_self_cancellation_terminates_both_sdk_futures(clients):
    """A backend cancellation terminates futures and allows subsequent sampling."""
    import ray
    from ray.util.state import list_actors
    actors = _real_sampler_actors()
    replicas = [actor for actor in list_actors(address=os.environ['TWINKLE_TEST_RAY_DASHBOARD_URL'], detail=True)
                if actor.state == 'ALIVE' and 'SamplerManagement' in actor.class_name
                and actor.class_name.startswith('ServeReplica')]
    assert len(replicas) == 1, [(actor.class_name, actor.name) for actor in replicas]
    replica = ray.get_actor(replicas[0].name, namespace='serve')

    def install(actor):
        from concurrent.futures import CancelledError
        service = actor._replica_impl._user_callable_wrapper._callable
        original = service.sampler.sample
        barrier = threading.Barrier(2)
        def cancelled(*args, **kwargs):
            result = original(*args, **kwargs)
            assert result[0].sequences[0].tokens
            barrier.wait(timeout=15)
            raise CancelledError('test: cancelled after real engine cleanup')
        service.sampler._e2e_original_sample = original
        service.sampler.sample = cancelled

    def restore(actor):
        service = actor._replica_impl._user_callable_wrapper._callable
        if hasattr(service.sampler, '_e2e_original_sample'):
            service.sampler.sample = service.sampler._e2e_original_sample
            del service.sampler._e2e_original_sample

    def status(actor):
        service = actor._replica_impl._user_callable_wrapper._callable
        return {'jobs': len(service.sampling_guard._jobs),
                'detached': len(service._compute_worker._detached_tasks),
                'reserved': service._compute_worker._detached_reserved,
                'weights_invalid': service.sampling_guard._weights_error is not None}

    ids = _prompt(clients, 'Reply with exactly this word and nothing else: apple')
    ray.get(replica.__ray_call__.remote(install), timeout=10)
    try:
        def cancelled_sample(dialect):
            started = time.monotonic()
            with pytest.raises(Exception, match='Backend generation was cancelled') as error:
                _sample(clients, dialect, ids, 2)
            assert time.monotonic() - started < 15, 'SDK future did not receive a terminal failure promptly'
            return {'dialect': dialect, 'error': str(error.value)}
        results, elapsed = _parallel([lambda: cancelled_sample('tinker'), lambda: cancelled_sample('twinkle')])
        deadline = time.monotonic() + 5
        while True:
            state = ray.get(replica.__ray_call__.remote(status), timeout=5)
            if state['jobs'] == state['detached'] == state['reserved'] == 0:
                break
            assert time.monotonic() < deadline, state
            time.sleep(.01)
        assert not state['weights_invalid']
        engine_requests = ray.get([actor.__ray_call__.remote(
            lambda sampler: len(sampler.engine.engine.output_processor.request_states)) for actor in actors], timeout=10)
        assert engine_requests == [0] * len(actors)
        ray.get(replica.__ray_call__.remote(restore), timeout=10)
        for dialect in ('tinker', 'twinkle'):
            assert _sample(clients, dialect, ids, 2)[0][0]
        _report(check='backend_self_cancellation', results=results, state=state,
                engine_requests=engine_requests, seconds=elapsed)
    finally:
        ray.get(replica.__ray_call__.remote(restore), timeout=10)
        ray.shutdown()
