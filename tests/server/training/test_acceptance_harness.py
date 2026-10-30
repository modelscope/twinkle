# Copyright (c) ModelScope Contributors. All rights reserved.
"""Acceptance harness budgets and Redis ownership are independently testable."""
import shutil

import pytest

from tests.server.integration.on_demand_training import stress_validation
from tests.server.integration.on_demand_training.managed_redis import ManagedRedis


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    # These tests deliberately use no Ray runtime.
    yield


class FakeHandle:
    calls = []

    def __init__(self, base, model, protocol, label):
        self.calls.append(('create', protocol))

    def step(self):
        self.calls.append(('step', None))

    def save(self, name):
        return 'checkpoint'

    def resume(self, path):
        assert path == 'checkpoint'

    def close(self):
        self.calls.append(('close', None))


def test_iteration_budget_does_not_round_up(monkeypatch):
    FakeHandle.calls = []
    monkeypatch.setattr(stress_validation, 'TrainingHandle', FakeHandle)
    evidence = stress_validation.run_stress('unused', 'unused', concurrency=2, iterations=3)
    assert evidence['completed'] == 3
    assert len([call for call in FakeHandle.calls if call[0] == 'close']) == 4  # Includes the warm peer.


def test_duration_is_not_truncated_by_iterations(monkeypatch):
    now = [0]

    def clock():
        now[0] += 1
        return now[0]

    monkeypatch.setattr(stress_validation.time, 'monotonic', clock)
    monkeypatch.setattr(stress_validation, 'TrainingHandle', FakeHandle)
    evidence = stress_validation.run_stress('unused', 'unused', concurrency=1, iterations=1, duration=10)
    assert evidence['completed'] > 1 and evidence['seconds'] >= 10


def test_failed_training_never_reports_success(monkeypatch):

    class Broken(FakeHandle):

        def step(self):
            raise RuntimeError('injected failure')

    monkeypatch.setattr(stress_validation, 'TrainingHandle', Broken)
    with pytest.raises(RuntimeError, match='injected failure'):
        stress_validation.run_stress('unused', 'unused', iterations=1)


@pytest.mark.skipif(not shutil.which('redis-server'), reason='redis-server executable required')
def test_owned_redis_restart_preserves_state(tmp_path):
    redis = pytest.importorskip('redis')
    managed = ManagedRedis(tmp_path)
    managed.start()
    client = redis.Redis.from_url(managed.url, socket_timeout=1)
    try:
        client.set('checkpoint', 'durable')

        def unavailable():
            with pytest.raises(redis.exceptions.ConnectionError):
                client.get('checkpoint')

        managed.fault(unavailable)
        assert client.get('checkpoint') == b'durable'
    finally:
        client.close()
        managed.stop()


def test_training_release_assertion_excludes_but_does_not_remove_sampling(monkeypatch):
    import importlib
    import ray
    from ray import serve
    from types import SimpleNamespace
    from tests.server.integration.on_demand_training.acceptance_support import assert_released
    namespace = 'owned'
    sampling_name = 'resources-owned-sampling_pool_epoch-sampler_id'
    applications = {sampling_name: object()}
    groups = {'sampler': {'name': sampling_name, 'state': 'CREATED'}}
    controller = SimpleNamespace(diagnostics=SimpleNamespace(remote=lambda: {'instances': [{'state': 'STOPPED'}]}))
    monkeypatch.setattr(ray, 'get', lambda value, **kwargs: value)
    monkeypatch.setattr(serve, 'status', lambda: SimpleNamespace(applications=applications))
    monkeypatch.setattr(importlib.import_module('ray.util.placement_group'), 'placement_group_table', lambda: groups)
    assert assert_released(controller, namespace)
    assert sampling_name in applications and groups['sampler']['state'] == 'CREATED'
    training_name = 'resources-owned-controller_epoch-instance_id'
    applications[training_name] = object()
    with pytest.raises(AssertionError):
        assert_released(controller, namespace)
    applications.pop(training_name)
    groups['training'] = {'name': training_name, 'state': 'CREATED'}
    with pytest.raises(AssertionError):
        assert_released(controller, namespace)


@pytest.mark.parametrize('values', [[], [0.0, 0.0], [float('nan')], [float('-inf')]])
def test_checkpoint_probe_rejects_vacuous_comparisons(values):
    from tests.server.integration.on_demand_training.acceptance_support import checked_logps
    with pytest.raises(AssertionError):
        checked_logps(values)


def test_checkpoint_probe_accepts_variable_length_sequences():
    from tests.server.integration.on_demand_training.acceptance_support import checked_logps
    assert checked_logps([[-0.25, -0.5], [-0.75]]).tolist() == [-0.25, -0.5, -0.75]
