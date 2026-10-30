# Copyright (c) ModelScope Contributors. All rights reserved.
import socket
import asyncio
import threading
import uuid
from unittest.mock import Mock

import pytest
import ray

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher
from twinkle.server.model.pool.actor import controller_name


def test_unresponsive_controller_does_not_block_other_cleanup():
    @ray.remote(num_cpus=0)
    class Controller:
        def __init__(self):
            self.entered = asyncio.Event()
            self.resume = asyncio.Event()

        async def shutdown(self):
            self.entered.set()
            await self.resume.wait()

        async def wait_until_entered(self):
            await self.entered.wait()

        async def unblock(self):
            self.resume.set()

    controller = Controller.remote()
    launcher = ServerLauncher(ServerConfig(resources={'shutdown_timeout_s': 0.3}))
    launcher._training_controller = controller
    finished = threading.Event()
    errors, cleaned = [], []

    def stop():
        try:
            launcher._run_cleanup([
                ('blocked controller', launcher._stop_training_controller),
                ('other owner', lambda: cleaned.append(True))], raise_errors=True)
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=stop)
    # Ensure actor startup is outside the deadline under test.
    ray.get(controller.__ray_ready__.remote(), timeout=10)
    entered = controller.wait_until_entered.remote()
    worker.start()
    try:
        ray.get(entered, timeout=10)
        assert finished.wait(2), 'Controller shutdown exceeded its budget and blocked the next owner'
        assert cleaned == [True]
        assert len(errors) == 1 and isinstance(errors[0].__cause__, ray.exceptions.GetTimeoutError)
        assert launcher._training_controller is controller
        ray.get(controller.unblock.remote(), timeout=5)
        launcher._stop_training_controller()
        assert launcher._training_controller is None
    finally:
        if not finished.is_set():
            ray.get(controller.unblock.remote(), timeout=5)
        worker.join(5)
        ray.kill(controller, no_restart=True)


def test_failed_initialization_does_not_leave_an_actor_that_blocks_a_retry():
    namespace = 'launcher-retry-' + uuid.uuid4().hex
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    devices = int(ray.cluster_resources()['CPU']) + 1
    payload = {
        'ray_namespace': namespace, 'http_options': {'host': '127.0.0.1', 'port': port},
        'applications': [{'name': 'retry', 'route_prefix': '/api/v1/model/retry', 'import_path': 'model',
                          'args': {'model_id': 'mock', 'backend': 'mock', 'train_mode': 'full',
                                   'device_group': {'ranks': devices, 'device_type': 'cpu'},
                                   'device_mesh': {'dp_size': devices}, 'min_replicas': 1}}],
    }
    launcher = ServerLauncher(ServerConfig.model_validate(payload))
    launcher._start_serve()
    try:
        with pytest.raises(ray.exceptions.RayTaskError, match='No single-node allocation'):
            launcher._deploy_training()
        with pytest.raises(ValueError):
            ray.get_actor(controller_name(namespace), namespace=namespace)
        payload['applications'][0]['args']['min_replicas'] = 0
        launcher.config = ServerConfig.model_validate(payload)
        launcher._deploy_training()
        diagnostic = ray.get(launcher._training_controller.diagnostics.remote())
        assert diagnostic['ready'] and diagnostic['instances'] == []
    finally:
        launcher._shutdown()


def test_cleanup_attempts_all_owners_and_reports_the_original_failure():
    calls = []
    error = ConnectionError('shutdown ACK unknown')

    def failed():
        calls.append('failed')
        raise error

    def healthy():
        calls.append('healthy')

    with pytest.raises(RuntimeError, match='cleanup is incomplete') as caught:
        ServerLauncher._run_cleanup([('first', failed), ('second', healthy)], raise_errors=True)
    assert calls == ['failed', 'healthy']
    assert caught.value.__cause__ is error


def test_startup_rollback_preserves_startup_error_and_cleans_all_owners(monkeypatch):
    config = ServerConfig.model_validate({'applications': [
        {'name': 'gateway', 'import_path': 'server', 'route_prefix': '/api/v1'}]})
    launcher = ServerLauncher(config)
    calls = []
    error = ValueError('invalid application')
    monkeypatch.setattr('twinkle.server.launcher.server_launcher.apply_ray_serve_patches', lambda: None)
    monkeypatch.setattr(launcher, '_configure_environment', lambda: None)
    monkeypatch.setattr(launcher, '_init_ray', lambda: None)
    monkeypatch.setattr(launcher, '_start_serve', lambda: None)

    def deploy(app):
        raise error

    def failed_cleanup():
        calls.append('training')
        raise ConnectionError('cleanup pending')

    monkeypatch.setattr(launcher, '_deploy_application', deploy)
    monkeypatch.setattr(launcher, '_stop_training_controller', failed_cleanup)
    monkeypatch.setattr('twinkle.server.telemetry.flush_telemetry_safely', lambda: calls.append('telemetry'))
    with pytest.raises(ValueError) as caught:
        launcher.launch()
    assert caught.value is error
    assert calls == ['training', 'telemetry']


@pytest.mark.parametrize('blocking', [False, True])
def test_launch_preserves_blocking_policy_and_deploys_on_demand_training(monkeypatch, blocking):
    config = ServerConfig.model_validate({'applications': [
        {'name': 'gateway', 'import_path': 'server', 'route_prefix': '/api/v1'},
        {'name': 'train', 'import_path': 'model', 'route_prefix': '/api/v1/model/train',
         'args': {'model_id': 'mock', 'backend': 'mock', 'train_mode': 'full',
                  'device_group': {'ranks': 1, 'device_type': 'cpu'},
                  'device_mesh': {'dp_size': 1}, 'min_replicas': 0}},
    ]})
    launcher = ServerLauncher(config)
    calls = []
    monkeypatch.setattr('twinkle.server.launcher.server_launcher.apply_ray_serve_patches', lambda: None)
    monkeypatch.setattr(launcher, '_configure_environment', lambda: calls.append('environment'))
    monkeypatch.setattr(launcher, '_init_ray', lambda: calls.append('ray'))
    monkeypatch.setattr(launcher, '_start_serve', lambda: calls.append('serve'))
    monkeypatch.setattr(launcher, '_deploy_application', lambda app: calls.append(app.name))
    monkeypatch.setattr(launcher, '_deploy_training', lambda: calls.append('training'))
    monkeypatch.setattr(launcher, '_wait_for_shutdown', lambda: calls.append('wait'))
    monkeypatch.setattr(launcher, '_shutdown', lambda **kwargs: calls.append('shutdown'))
    launcher.launch(blocking=blocking)
    expected = ['environment', 'ray', 'serve', 'gateway', 'train', 'training']
    assert calls == expected + (['wait', 'shutdown'] if blocking else [])


def test_start_serve_reuses_existing_instance_without_restart(monkeypatch):
    launcher = ServerLauncher(ServerConfig())
    client = Mock(return_value=object())
    start, shutdown = Mock(), Mock()
    monkeypatch.setattr('ray.serve.context._get_global_client', client)
    monkeypatch.setattr('ray.serve.start', start)
    monkeypatch.setattr('ray.serve.shutdown', shutdown)
    launcher._start_serve()
    launcher._start_serve()
    client.assert_called_once_with()
    start.assert_not_called()
    shutdown.assert_not_called()
    assert launcher._serve_started


def test_start_serve_starts_missing_instance_with_configured_listener(monkeypatch):
    launcher = ServerLauncher(ServerConfig(proxy_location='HeadOnly'))
    client = Mock(side_effect=RuntimeError('Serve is not running'))
    start, shutdown = Mock(), Mock()
    monkeypatch.setattr('ray.serve.context._get_global_client', client)
    monkeypatch.setattr('ray.serve.start', start)
    monkeypatch.setattr('ray.serve.shutdown', shutdown)
    launcher._start_serve()
    start.assert_called_once_with(http_options=launcher.config.http_options.model_dump(), proxy_location='HeadOnly')
    shutdown.assert_not_called()
    assert launcher._serve_started


@pytest.mark.parametrize('blocking', [False, True])
def test_launch_server_forwards_blocking_policy(monkeypatch, blocking):
    from twinkle.server.launcher.server_launcher import launch_server

    config = ServerConfig()
    launcher = Mock()
    factory = Mock(return_value=launcher)
    monkeypatch.setattr('twinkle.server.launcher.server_launcher.ServerLauncher', factory)
    launch_server(config=config, blocking=blocking)
    factory.assert_called_once_with(config=config, ray_namespace=config.ray_namespace)
    launcher.launch.assert_called_once_with(blocking=blocking)
