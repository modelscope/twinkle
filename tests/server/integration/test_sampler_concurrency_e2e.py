"""Real Tinker/Twinkle concurrency, weight isolation and lifecycle checks."""
from __future__ import annotations

import math
import os
import threading
import time

import pytest

from tests.server.integration.sampler_e2e_helpers import (
    _assert_adapter_isolation, _checkpoint_handles, _parallel, _prompt, _ray_runtime,
    _report, _reset_canonical_state_actor, _sample, _real_sampler_actors, adapters, clients, full_checkpoints,
)

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


def test_async_generation_cleanup_and_recovery_on_real_engine(clients):
    import ray
    from ray.util.state import list_actors
    ray.init(address=os.environ['RAY_ADDRESS'], namespace='default', ignore_reinit_error=True,
             log_to_driver=False)
    actors = list_actors(address=os.environ.get('TWINKLE_TEST_RAY_DASHBOARD_URL', 'http://127.0.0.1:8395'),
                         filters=[('class_name', '=', 'VLLMSamplerTQ'), ('state', '=', 'ALIVE')], detail=True)
    assert len(actors) == int(os.environ.get('TWINKLE_TEST_DP', '1'))
    assert all(actor.state == 'ALIVE' for actor in actors)
    actor = ray.get_actor(actors[0].name, namespace='default')
    ids = _prompt(clients, 'Write a very long story of at least 2000 words about an astronaut.')
    def call(method, *args):
        return ray.get(getattr(actor, method).remote(*args), timeout=30)
    def wait_status(key, terminal):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            state = call('get_generation_status', key)
            if state['status'] in terminal:
                return state
            time.sleep(.01)
        raise AssertionError(f'{key} did not terminate: {state}')
    try:
        call('submit_generation', 'e2e-bad', [{'input_ids': [10**9]}],
             {'max_tokens': 8, 'temperature': 0})
        failed = wait_status('e2e-bad', {'failed'})
        assert call('cancel_generation', 'e2e-bad')['status'] == 'completed'
        # A failed prompt in ordinary sample() must wait for its live sibling.
        batch = actor.sample.remote(
            [{'input_ids': ids}, {'input_ids': [10**9]}],
            {'max_tokens': 1024, 'temperature': 0})
        ready, _ = ray.wait([batch], timeout=.2)
        assert not ready, 'ordinary batch returned before its sibling generation exited'
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(batch, timeout=30)

        call('submit_generation', 'e2e-cancel', [{'input_ids': ids}],
             {'max_tokens': 1024, 'temperature': 0})
        time.sleep(.1)
        assert call('get_generation_status', 'e2e-cancel')['status'] == 'running'
        assert call('cancel_generation', 'e2e-cancel')['status'] == 'cancellation_requested'
        wait_status('e2e-cancel', {'cancelled'})
        assert call('cancel_generation', 'e2e-cancel')['status'] == 'completed'
        call('submit_generation', 'e2e-after', [{'input_ids': ids}],
             {'max_tokens': 8, 'temperature': 0})
        wait_status('e2e-after', {'completed'})
        responses = call('collect_generation', 'e2e-after')
        assert responses[0].sequences[0].tokens
        _report(check='async_cleanup_recovery', failure=failed, recovered=True)
    finally:
        for key in ['e2e-bad', 'e2e-cancel', 'e2e-after']:
            call('cancel_generation', key)
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
    """Cancel at the driver boundary after both requests performed real PPU generation.

    Submit both requests before cancellation quarantines the replica. This test
    runs last because uncertain backend cancellation requires a replica restart.
    """
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
        service.sampler.sample = service.sampler._e2e_original_sample
        del service.sampler._e2e_original_sample

    def status(actor):
        service = actor._replica_impl._user_callable_wrapper._callable
        return {'jobs': len(service.sampling_guard._jobs),
                'detached': len(service._compute_worker._detached_tasks),
                'reserved': service._compute_worker._detached_reserved,
                'quarantined': service.sampling_guard._uncertain}

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
        assert state['quarantined']
        engine_requests = ray.get([actor.__ray_call__.remote(
            lambda sampler: len(sampler.engine.engine.output_processor.request_states)) for actor in actors], timeout=10)
        assert engine_requests == [0] * len(actors)
        for dialect in ('tinker', 'twinkle'):
            with pytest.raises(Exception):
                _sample(clients, dialect, ids, 2)
        _report(check='backend_self_cancellation', results=results, state=state,
                engine_requests=engine_requests, seconds=elapsed)
    finally:
        ray.get(replica.__ray_call__.remote(restore), timeout=10)
        ray.shutdown()
