# Copyright (c) ModelScope Contributors. All rights reserved.
"""Manager state machine for delta synchronization, driven by mock trainer and sampler actors."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from torch import nn

from twinkle.checkpoint_engine import CheckpointEngineManager
from twinkle.checkpoint_engine.delta import DeltaSyncConfig


def manager_pair():
    model = Mock(_actors=[1, 2], device_mesh=SimpleNamespace(world_size=2))
    sampler = Mock(_actors=[3], device_mesh=SimpleNamespace(data_world_size=1))
    model.prepare_delta_sync.return_value = dict(device='cuda', master_metadata=object())
    sampler.validate_delta_sampler.return_value = dict(device='cuda', tp=2)
    sampler.delta_sampler_metadata.return_value = [
        dict(node_ip='10.0.0.1', device_uuid=f'GPU-{rank}', zmq_handle=f'ipc:///rank{rank}')
        for rank in range(2)]
    model.send_delta_weights.return_value = lambda: {'payload_bytes': 6}
    sampler.receive_delta_weights.return_value = lambda: {'changed': True}
    manager = CheckpointEngineManager(model, sampler, sync_backend='delta_sharded')
    return manager, model, sampler


@pytest.fixture
def driver_proxy_manager():
    manager = CheckpointEngineManager.__new__(CheckpointEngineManager)
    manager.model = nn.Module.__new__(nn.Module)
    manager.sampler = SimpleNamespace()
    manager.delta_config = SimpleNamespace(timeout_seconds=30)
    return manager


@pytest.mark.parametrize('bucket_bytes', [0, 1, 5])
def test_bucket_must_hold_one_patch_element(bucket_bytes):
    with pytest.raises(ValueError):
        DeltaSyncConfig(bucket_bytes=bucket_bytes)


@pytest.mark.parametrize('timeout', [0, -1, float('inf'), float('nan')])
def test_timeout_is_finite_positive(timeout):
    with pytest.raises(ValueError):
        DeltaSyncConfig(timeout_seconds=timeout)


def test_delta_timeout_is_applied_to_driver_proxy(driver_proxy_manager):
    with driver_proxy_manager._delta_rpc_timeouts():
        assert driver_proxy_manager.model._ray_get_timeout == 30


def test_delta_timeout_cleanup_supports_uninitialized_module_proxy(driver_proxy_manager):
    with driver_proxy_manager._delta_rpc_timeouts():
        pass
    assert '_ray_get_timeout' not in vars(driver_proxy_manager.model)


@pytest.mark.parametrize('previous_timeout', [None, 10])
def test_delta_timeout_restores_existing_proxy_value(driver_proxy_manager, previous_timeout):
    driver_proxy_manager.model._ray_get_timeout = previous_timeout
    with driver_proxy_manager._delta_rpc_timeouts():
        pass
    assert driver_proxy_manager.model._ray_get_timeout == previous_timeout


def test_delta_rejects_local_deployment():
    with pytest.raises(ValueError, match='standalone'):
        CheckpointEngineManager(SimpleNamespace(), SimpleNamespace(), sync_backend='delta_sharded')


def test_seed_then_sparse_reuses_transport():
    manager, model, sampler = manager_pair()
    manager.sync_weights()
    manager.sync_weights()
    assert model.init_delta_transport.call_count == 1
    assert manager._delta_round == 2
    assert model.init_delta_transport.call_args.args[0]['world_size'] == 3
    sampler.validate_delta_sampler.assert_called_once()
    sampler.delta_sampler_metadata.assert_called_once()
    model.prepare_delta_sync.assert_called_once()


@pytest.mark.parametrize('stage', ['send_delta_weights', 'receive_delta_weights', 'publish_delta_sampler'])
def test_failure_does_not_publish_version(stage):
    manager, model, sampler = manager_pair()
    manager.sync_weights()
    getattr(model if stage == 'send_delta_weights' else sampler, stage).side_effect = TimeoutError('injected')
    with pytest.raises(TimeoutError):
        manager.sync_weights()
    assert manager._delta_round == 1
    sampler.fail_delta_sampler.assert_called_once()
    with pytest.raises(RuntimeError, match='failed'):
        manager.sync_weights()


def test_failed_round_requires_full_job_restart_even_after_close():
    manager, model, sampler = manager_pair()
    manager.sync_weights()
    sampler.receive_delta_weights.side_effect = ValueError('worker write failed')
    with pytest.raises(ValueError):
        manager.sync_weights()
    sampler.receive_delta_weights.side_effect = None
    manager.close_sync_session()
    with pytest.raises(RuntimeError, match='restart the entire'):
        manager.sync_weights()
    assert manager._delta_failed


def test_cleanup_failure_does_not_clear_failed_state():
    manager, model, sampler = manager_pair()
    manager.sync_weights()
    model.close_delta_trainer.side_effect = RuntimeError('communicator hung')
    with pytest.raises(RuntimeError, match='cleanup'):
        manager.close_sync_session()
    assert manager._delta_failed


def test_zero_change_still_resumes_manager_lifecycle():
    manager, model, sampler = manager_pair()
    manager.sync_weights()
    sampler.publish_delta_sampler.reset_mock()
    sampler.receive_delta_weights.return_value = lambda: {'changed': False}
    manager.sync_weights()
    sampler.publish_delta_sampler.assert_called_once()
    assert sampler.begin_delta_sync.call_count == 2


def test_multiple_rollout_replicas_share_one_broadcast_group():
    manager, model, sampler = manager_pair()
    sampler.validate_delta_sampler.return_value = [dict(device='cuda', tp=2)] * 2
    sampler.delta_sampler_metadata.return_value = [
        dict(node_ip=f'10.0.0.{rank // 2 + 1}', device_uuid=f'GPU-{rank}', zmq_handle=f'ipc:///rank{rank}')
        for rank in range(4)]
    sampler.receive_delta_weights.return_value = lambda: [{'changed': True}] * 2
    manager.sync_weights()
    manager.sync_weights()
    endpoint = model.init_delta_transport.call_args.args[0]
    assert endpoint['world_size'] == 5
    assert endpoint['rollout_workers'] == sampler.delta_sampler_metadata.return_value
    model.init_delta_transport.assert_called_once()
    sampler.publish_delta_sampler.assert_called()


def test_replica_completion_disagreement_keeps_rollout_paused():
    manager, model, sampler = manager_pair()
    sampler.validate_delta_sampler.return_value = [dict(device='cuda', tp=2)] * 2
    sampler.delta_sampler_metadata.return_value = [
        dict(node_ip='10.0.0.1', device_uuid=f'GPU-{rank}', zmq_handle=f'ipc:///rank{rank}')
        for rank in range(4)]
    sampler.receive_delta_weights.return_value = lambda: [{'changed': True}] * 2
    manager.sync_weights()
    sampler.publish_delta_sampler.reset_mock()
    sampler.receive_delta_weights.return_value = lambda: [{'changed': True}, {'changed': False}]
    with pytest.raises(ValueError, match='replicas disagree'):
        manager.sync_weights()
    sampler.publish_delta_sampler.assert_not_called()
    sampler.fail_delta_sampler.assert_called_once()


def test_replica_metrics_may_differ_when_completion_agrees():
    manager, model, sampler = manager_pair()
    sampler.validate_delta_sampler.return_value = [dict(device='cuda', tp=2)] * 2
    sampler.delta_sampler_metadata.return_value = [
        dict(node_ip='10.0.0.1', device_uuid=f'GPU-{rank}', zmq_handle=f'ipc:///rank{rank}')
        for rank in range(4)]
    sampler.receive_delta_weights.return_value = lambda: [
        dict(changed=True, seconds=1), dict(changed=True, seconds=2)]
    manager.sync_weights()
    sampler.publish_delta_sampler.assert_called_once()


@pytest.mark.parametrize('invalid_topology', ['missing_worker', 'duplicate_gpu'])
def test_invalid_rollout_topology_fails_before_group_init(invalid_topology):
    manager, model, sampler = manager_pair()
    workers = sampler.delta_sampler_metadata.return_value
    sampler.delta_sampler_metadata.return_value = workers[:1] if invalid_topology == 'missing_worker' else [workers[0]] * 2
    with pytest.raises(ValueError, match='topology|distinct'):
        manager.sync_weights()
    model.init_delta_transport.assert_not_called()


@pytest.mark.parametrize('stage', ['validate_delta_sampler', 'delta_sampler_metadata', 'prepare_delta_sync'])
def test_preflight_failure_can_be_corrected_and_retried(stage):
    manager, model, sampler = manager_pair()
    owner = model if stage == 'prepare_delta_sync' else sampler
    getattr(owner, stage).side_effect = ValueError('invalid configuration')
    with pytest.raises(ValueError, match='invalid configuration'):
        manager.sync_weights()
    sampler.fail_delta_sampler.assert_not_called()
    sampler.begin_delta_sync.assert_not_called()
    getattr(owner, stage).side_effect = None
    manager.sync_weights()
    assert manager.last_delta_metrics['seed'] is True


def test_manager_begin_pauses_before_receivers_or_sender_run():
    from twinkle.checkpoint_engine import CheckpointEngineManager
    events = []
    model = Mock(_actors=[1])
    sampler = Mock(_actors=[2])
    model.prepare_delta_sync.return_value = dict(device='cuda')
    sampler.validate_delta_sampler.return_value = {'tp': 2}
    sampler.delta_sampler_metadata.return_value = [
        dict(node_ip='node', device_uuid=str(rank), zmq_handle=str(rank)) for rank in range(2)]
    sampler.begin_delta_sync.side_effect = lambda: events.append('pause-release')
    model.init_delta_transport.side_effect = lambda endpoint: (events.append('init'), lambda: None)[1]
    sampler.init_delta_sampler_transport.return_value = lambda: None
    model.send_delta_weights.side_effect = lambda: (events.append('send'), lambda: {})[1]
    sampler.receive_delta_weights.return_value = lambda: {'changed': True}
    sampler.publish_delta_sampler.side_effect = lambda: events.append('restore-resume')
    manager = CheckpointEngineManager(model, sampler, sync_backend='delta_sharded')
    manager.sync_weights()
    assert events == ['pause-release', 'init', 'send', 'restore-resume']
