# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real resident reset ACK racing worker death; needs two dedicated accelerators."""
import asyncio
import json
import os
import socket
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TWINKLE_TEST_RESIDENT_RELEASE') != '1',
                                reason='Requires two dedicated devices and a real cached model')


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    yield


@pytest.fixture(autouse=True)
def _reset_canonical_state_actor():
    yield


def release_gate(actor, action):
    """Delay only the ACK after the real model reset; never substitute computation."""
    if action == 'install':
        entered, resume = asyncio.Event(), asyncio.Event()
        original = actor.driver.release_training

        async def delayed(instance, training):
            await original(instance, training)
            entered.set()
            await resume.wait()

        actor.driver.release_training = delayed
        actor._release_test_gate = (entered, resume, original)
    elif action == 'entered':
        return actor._release_test_gate[0].is_set()
    elif action == 'resume':
        actor._release_test_gate[1].set()
        actor.driver.release_training = actor._release_test_gate[2]


def worker_origin(actor):
    import twinkle
    from twinkle.server.model.pool import controller
    return {'python': sys.executable, 'twinkle': twinkle.__file__, 'controller': controller.__file__,
            'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}


def test_resident_release_ack_after_worker_failure(tmp_path):
    import ray
    from ray import serve
    from ray.util import list_named_actors
    from twinkle.server.config import ServerConfig
    from twinkle.server.launcher.server_launcher import ServerLauncher
    from tests.server.integration.on_demand_training.acceptance_support import TrainingHandle, wait_until, assert_released
    from tests.server.integration.on_demand_training.managed_redis import ManagedRedis

    root = Path(__file__).resolve().parents[4]
    directory = Path(os.environ.get('TWINKLE_TEST_RUN_DIR', str(tmp_path)))
    directory.mkdir(parents=True, exist_ok=True)
    backend = os.environ['TWINKLE_TEST_BACKEND']
    model = os.environ['TWINKLE_TEST_MODEL']
    namespace = 'resident_' + uuid.uuid4().hex[:8]
    persistence = 'redis' if backend == 'transformers' else 'memory'
    managed = ManagedRedis(directory / 'redis') if persistence == 'redis' else None
    if managed:
        managed.start()
    prefix = namespace + '::'
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    model_name = 'Qwen/Qwen3.5-4B'
    raw = {
        'ray_namespace': namespace, 'http_options': {'host': '127.0.0.1', 'port': port},
        'resources': {'startup_timeout_s': 1800, 'shutdown_timeout_s': 180, 'sweep_interval_s': 1},
        'persistence': ({'mode': persistence, 'redis_url': managed.url, 'key_prefix': prefix}
                        if managed else {'mode': persistence}),
        'applications': [
            {'name': 'server', 'route_prefix': '/api/v1', 'import_path': 'server',
             'args': {'supported_models': [model_name]}},
            {'name': 'model', 'route_prefix': '/api/v1/model/' + model_name, 'import_path': 'model',
             'args': {'model_id': model, 'backend': backend, 'train_mode': 'full',
                      'device_group': {'ranks': 2, 'device_type': 'cuda'},
                      'min_replicas': 1, 'max_replicas': 1,
                      'device_mesh': {'dp_size': 2 if backend == 'transformers' else 1,
                                      'pp_size': 1 if backend == 'transformers' else 2},
                      'max_length': 512, 'idle_timeout_s': 3600,
                      'queue_config': {'execution_timeout': 600},
                      'env_vars': {'TWINKLE_TRUST_REMOTE_CODE': '1'}}},
        ],
    }
    (directory / 'server.json').write_text(json.dumps(raw, indent=2))
    os.environ.update(TWINKLE_STATE_RAY_NAMESPACE=namespace, TWINKLE_PERSISTENCE_MODE=persistence,
                      TWINKLE_DEFAULT_SAVE_DIR=str(directory / 'checkpoints'), TWINKLE_TRUST_REMOTE_CODE='1',
                      TWINKLE_MODEL_ID_ALIASES=json.dumps({model_name: model}),
                      PYTHONPATH=os.pathsep.join((str(root / 'src'), str(root))))
    if managed:
        os.environ.update(TWINKLE_PERSISTENCE_REDIS_URL=managed.url, TWINKLE_PERSISTENCE_KEY_PREFIX=prefix)
    else:
        os.environ.pop('TWINKLE_PERSISTENCE_REDIS_URL', None)
        os.environ.pop('TWINKLE_PERSISTENCE_KEY_PREFIX', None)
    evidence = {'backend': backend, 'namespace': namespace,
                'sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()}
    ray.init(address='local', namespace=namespace, num_cpus=24, num_gpus=2,
             _temp_dir='/tmp/twr-' + namespace[-8:], include_dashboard=False)
    launcher = ServerLauncher(ServerConfig.model_validate(raw))
    handles, controller = [], None
    try:
        launcher._start_serve()
        for app in launcher.config.applications:
            launcher._deploy_application(app)
        launcher._deploy_training()
        controller = launcher._training_controller
        url = f'http://127.0.0.1:{port}'
        old = TrainingHandle(url, model_name, 'twinkle', 'before-fault', mode='full')
        handles.append(old)
        baseline = old.evaluate()
        old.step()
        old.step()
        trained = old.evaluate()
        assert np.max(np.abs(trained - baseline)) > 2e-3, 'Training did not update real model logprobs'
        diagnostic = ray.get(controller.diagnostics.remote())
        instance = next(row for row in diagnostic['instances'] if old.identity in row['training_ids'])
        owned = [row for row in list_named_actors(all_namespaces=True)
                 if row['namespace'] == namespace and instance['instance_id'] in row['name']
                 and row['name'].startswith('model-') and not row['name'].endswith('-registry')]
        assert len(owned) == 2
        evidence['worker_origins'] = [ray.get(ray.get_actor(row['name'], namespace=namespace).__ray_call__.remote(
            worker_origin)) for row in owned]
        for origin in evidence['worker_origins']:
            assert Path(origin['twinkle']).is_relative_to(root / 'src')
            assert origin['python'] == sys.executable
        ray.get(controller.__ray_call__.remote(release_gate, 'install'))
        with ThreadPoolExecutor(max_workers=1) as executor:
            releasing = executor.submit(old.close)
            try:
                wait_until(lambda: ray.get(controller.__ray_call__.remote(release_gate, 'entered')),
                           600, 'Real reset did not finish')
                ray.kill(ray.get_actor(owned[0]['name'], namespace=namespace), no_restart=True)

                def stopped():
                    rows = ray.get(controller.diagnostics.remote())['instances']
                    return next((row for row in rows if row['instance_id'] == instance['instance_id']
                                 and row['state'] == 'STOPPED'), None)

                wait_until(stopped, 600, 'Health monitor did not retire the failed resident')
            finally:
                ray.get(controller.__ray_call__.remote(release_gate, 'resume'))
            releasing.result(timeout=600)
        assert stopped(), 'A late reset ACK resurrected the dead resident'
        fresh = TrainingHandle(url, model_name, 'tinker', 'after-fault', mode='full')
        handles.append(fresh)
        restored = fresh.evaluate()
        np.testing.assert_allclose(restored, baseline, rtol=0, atol=2e-3)
        fresh.step()
        continued = fresh.evaluate()
        assert np.max(np.abs(continued - restored)) > 2e-3
        fresh.close()
        assert stopped()
        rows = ray.get(controller.diagnostics.remote())['instances']
        assert sum(row['state'] == 'READY' for row in rows) == 1
        evidence.update(late_ack_fenced=True, recovered=True, baseline_logps=baseline.tolist(),
                        trained_logps=trained.tolist(), restored_logps=restored.tolist(),
                        continued_logps=continued.tolist(), instances=rows)
    finally:
        if controller is not None:
            ray.get(controller.__ray_call__.remote(
                lambda actor: release_gate(actor, 'resume') if hasattr(actor, '_release_test_gate') else None))
        for handle in handles:
            handle.close(stale=True)
        try:
            if controller is not None:
                ray.get(controller.shutdown.remote(), timeout=240)
                assert_released(controller, namespace, managed.url if managed else None, prefix)
                evidence['released'] = True
            launcher._shutdown()
        finally:
            serve.shutdown()
            ray.shutdown()
            if managed:
                managed.stop()
            (directory / 'result.json').write_text(json.dumps(evidence, indent=2))
