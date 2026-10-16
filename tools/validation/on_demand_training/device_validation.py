# Copyright (c) ModelScope Contributors. All rights reserved.
"""Explicit real-device acceptance; run this file only after local mock checks.

Required: TWINKLE_DEVICE_BACKEND, TWINKLE_DEVICE_MODEL, TWINKLE_DEVICE_OTHER_MODEL,
TWINKLE_DEVICE_RUN_DIR and four exclusively available visible devices.
Tests cover DP resource lifecycle without checkpoint I/O.
This is intentionally outside pytest's default test filename pattern.
"""
import json
import numpy as np
import os
import pytest
import ray
import requests
import socket
import time
import uuid
from on_demand_sdk import unload_tinker_model
from pathlib import Path
from ray import serve

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher
from twinkle_client import init_tinker_client
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import ClientContext
from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel


@pytest.fixture(scope='module')
def server():
    backend = os.environ['TWINKLE_DEVICE_BACKEND']
    assert backend in ('transformers', 'megatron')
    run_dir = Path(os.environ['TWINKLE_DEVICE_RUN_DIR'])
    run_dir.mkdir(parents=True, exist_ok=True)
    namespace = 'device_' + uuid.uuid4().hex[:8]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    os.environ['TWINKLE_STATE_RAY_NAMESPACE'] = namespace
    os.environ['TWINKLE_DEFAULT_SAVE_DIR'] = str(run_dir / 'checkpoints')
    os.environ['TWINKLE_PERSISTENCE_MODE'] = 'memory'
    os.environ['TWINKLE_LONG_POLL_TIMEOUT'] = '2'
    mesh = {'dp_size': 2}
    apps = []
    for alias, variable in [('small', 'TWINKLE_DEVICE_MODEL'), ('other', 'TWINKLE_DEVICE_OTHER_MODEL')]:
        source = os.environ[variable]
        assert Path(source, 'config.json').is_file()
        mode = {'num_devices': 2, 'device_mesh': mesh, 'idle_timeout_s': 600 if alias == 'small' else 30}
        if backend == 'transformers':
            mode['strategy'] = 'accelerate'
        apps.append({
            'name': alias,
            'route_prefix': '/api/v1/model/' + alias,
            'import_path': 'model',
            'args': {
                'model_id': source,
                'backend': backend,
                'device_type': 'cuda',
                'modes': {
                    'lora': {
                        **mode, 'max_loras': 2,
                        'max_r': 8
                    },
                    'full': mode
                },
            }
        })
    apps.append({
        'name': 'gateway',
        'route_prefix': '/api/v1',
        'import_path': 'server',
        'args': {
            'server_config': {
                'per_token_model_limit': 10
            }
        },
        'deployments': [{
            'ray_actor_options': {
                'num_cpus': 0.1
            }
        }]
    })
    os.environ['TWINKLE_MODEL_ID_ALIASES'] = json.dumps(
        {app['name']: app['args']['model_id']
         for app in apps if app['import_path'] == 'model'})
    config = ServerConfig(ray_namespace=namespace, http_options={'host': '127.0.0.1', 'port': port}, applications=apps)
    import yaml
    (run_dir / 'server.yaml').write_text(yaml.safe_dump(config.to_yaml_dict(), sort_keys=False))
    ray.init(
        address='local',
        namespace=namespace,
        num_cpus=16,
        num_gpus=4,
        _temp_dir='/tmp/twd-' + namespace[-8:],
        include_dashboard=False,
        log_to_driver=True)
    launcher = ServerLauncher(config)
    launcher._start_serve()
    evidence = {
        'backend': backend,
        'namespace': namespace,
        'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'cases': []
    }
    try:
        for app in config.applications:
            launcher._deploy_application(app)
        launcher._deploy_training()
        controller = launcher._training_controller
        base = f'http://127.0.0.1:{port}'
        assert ray.get(controller.diagnostics.remote())['instances'] == []
        evidence['ray_address'] = ray.get_runtime_context().gcs_address
        yield base, controller, evidence, launcher
    finally:
        if getattr(launcher, '_training_controller', None) is not None:
            ray.get(launcher._training_controller.shutdown.remote(), timeout=180)
            ray.kill(launcher._training_controller, no_restart=True)
        (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
        serve.shutdown()
        ray.shutdown()


class Handle:

    def __init__(self, base, protocol, mode, alias='small'):
        self.base, self.protocol = base, protocol
        self.token = 'device-validation-owner'
        session = requests.post(base + '/api/v1/twinkle/create_session', json={}, timeout=20).json()['session_id']
        self.transport = ClientTransport(ClientContext(base_url=base, api_key=self.token, session_id=session))
        self.rate = 1e-3 if mode == 'lora' else 5e-5
        if protocol == 'native':
            from peft import LoraConfig
            self.client = MultiLoraTransformersModel(alias, transport=self.transport)
            self.client.add_adapter_to_model('same',
                                             LoraConfig(r=8, target_modules='all-linear') if mode == 'lora' else None)
            self.identity = self.client._training_ids['same']
            self.client.set_loss('CrossEntropyLoss')
            self.client.set_optimizer('Adam', lr=self.rate)
            self.client.set_processor('InputProcessor', padding_side='right')
        else:
            init_tinker_client()
            from tinker import ServiceClient
            self.service = ServiceClient(base_url=base, api_key=self.token)
            self.client = (
                self.service.create_lora_training_client(base_model=alias, rank=8)
                if mode == 'lora' else self.service.create_full_training_client(base_model=alias))
            self.identity = self.client.model_id
        # Fixed token IDs need no external dataset, and every position has a target.
        self.inputs = [{
            'input_ids': [100, 200, 300, 400, 500, 600, 700, 800],
            'labels': [200, 300, 400, 500, 600, 700, 800, 900],
            'position_ids': list(range(8)),
            'attention_mask': [1] * 8
        }] * 2

    def datum(self):
        from tinker import types
        feature = self.inputs[0]
        return [
            types.Datum(
                model_input=types.ModelInput.from_ints(feature['input_ids']),
                loss_fn_inputs={
                    'target_tokens': types.TensorData(data=feature['labels'], dtype='int64', shape=[8]),
                    'weights': types.TensorData(data=[1.] * 8, dtype='float32', shape=[8]),
                }) for _ in self.inputs
        ]

    def evaluate(self):
        if self.protocol == 'native':
            result = self.client.forward_only(self.inputs).result
            values = result['logps']
        else:
            result = self.client.forward(self.datum(), 'cross_entropy').result()
            values = result.loss_fn_outputs[0]['logprobs'].tolist()
        values = np.asarray(values, dtype=float).reshape(-1)
        assert values.size and np.isfinite(values).all(), values
        return values

    def step(self):
        if self.protocol == 'native':
            self.client.forward_backward(self.inputs)
            self.client.clip_grad_and_step()
        else:
            from tinker import types
            self.client.forward_backward(self.datum(), 'cross_entropy').result()
            self.client.optim_step(types.AdamParams(learning_rate=self.rate)).result()

    def close(self):
        if self.protocol == 'native':
            self.client.remove_adapter()
        else:
            assert unload_tinker_model(self.service, self.identity).model_id == self.identity
        self.transport.close()


def active_instances(controller):
    return [row for row in ray.get(controller.diagnostics.remote())['instances'] if row['state'] != 'STOPPED']


@pytest.mark.parametrize('protocol', ['native', 'tinker'])
@pytest.mark.parametrize('mode', ['lora', 'full'])
def test_dp_resource_lifecycle(server, protocol, mode):
    base, controller, evidence, _ = server
    assert not active_instances(controller)
    assert ray.available_resources().get('GPU', 0) == 4
    first, second = Handle(base, protocol, mode), Handle(base, protocol, mode)
    instances = active_instances(controller)
    assert len(instances) == (1 if mode == 'lora' else 2)
    assert all(len(row['devices']) == 2 for row in instances)
    first_instance_ids = {row['instance_id'] for row in instances}
    baseline, untouched = first.evaluate(), second.evaluate()
    for _ in range(4):
        first.step()
    trained = first.evaluate()
    assert float(trained.mean()) > float(baseline.mean()) + 1e-4
    np.testing.assert_allclose(second.evaluate(), untouched, rtol=0, atol=2e-3)
    second.step()
    peer_trained = second.evaluate()
    assert float(peer_trained.mean()) > float(untouched.mean()) + 1e-4
    np.testing.assert_allclose(first.evaluate(), trained, rtol=0, atol=2e-3)
    if mode == 'lora':
        third, fourth = Handle(base, protocol, mode), Handle(base, protocol, mode)
        third.evaluate()
        instances = active_instances(controller)
        assert len(instances) == 2
        assert ray.available_resources().get('GPU', 0) == 0
        with pytest.raises(Exception, match='(?i)allocation'):
            Handle(base, protocol, mode)
    allocated = [device for row in instances for device in row['devices']]
    assert len(allocated) == 4 and len(set(allocated)) == 4
    if mode == 'lora':
        fourth.close()
        third.close()
    first.close()
    assert len(active_instances(controller)) == 1
    assert ray.available_resources().get('GPU', 0) == 2
    np.testing.assert_allclose(second.evaluate(), peer_trained, rtol=0, atol=2e-3)
    second.step()
    peer_continued = second.evaluate()
    assert float(peer_continued.mean()) > float(peer_trained.mean()) + 1e-4
    second.close()
    assert not active_instances(controller)
    assert ray.available_resources().get('GPU', 0) == 4

    alternate = Handle(base, protocol, mode, alias='other')
    alternate_instances = active_instances(controller)
    assert len(alternate_instances) == 1
    reused = alternate_instances[0]['devices']
    assert len(reused) == 2 and set(reused).issubset(allocated)
    other_initial = alternate.evaluate()
    for _ in range(4):
        alternate.step()
    other_trained = alternate.evaluate()
    assert float(other_trained.mean()) > float(other_initial.mean()) + 1e-4
    health = requests.get(base + '/api/v1/twinkle/healthz/deep', timeout=30)
    assert health.status_code == 200 and health.json()['healthy']
    deadline = time.monotonic() + 90
    while active_instances(controller):
        assert time.monotonic() < deadline, 'Business-idle instance did not release DP devices'
        response = alternate.transport.post(
            base + '/api/v1/twinkle/session_heartbeat',
            json_data={'session_id': alternate.transport.context.session_id})
        response.raise_for_status()
        time.sleep(2)
    assert ray.available_resources().get('GPU', 0) == 4
    with pytest.raises(Exception) as expired:
        alternate.evaluate()
    alternate.transport.close()

    fresh = Handle(base, protocol, mode)
    fresh_instances = active_instances(controller)
    assert len(fresh_instances) == 1
    assert fresh_instances[0]['instance_id'] not in first_instance_ids
    np.testing.assert_allclose(fresh.evaluate(), baseline, rtol=0, atol=2e-3)
    fresh.close()
    assert not active_instances(controller)
    assert ray.available_resources().get('GPU', 0) == 4
    evidence['cases'].append({
        'protocol': protocol,
        'mode': mode,
        'parallelism': {
            'dp_size': 2,
            'tp_size': 1,
            'fsdp_size': 1
        },
        'initial_logps': baseline.tolist(),
        'trained_logps': trained.tolist(),
        'peer_initial_logps': untouched.tolist(),
        'peer_trained_logps': peer_trained.tolist(),
        'peer_continued_logps': peer_continued.tolist(),
        'other_initial_logps': other_initial.tolist(),
        'other_trained_logps': other_trained.tolist(),
        'original_devices': allocated,
        'cross_model_reused_devices': reused,
        'health': health.json(),
        'idle_expired_error': str(expired.value)[:400],
        'lazy_start': True,
        'shared_or_exclusive_isolation': True,
        'idle_reclaimed': True,
        'fresh_recreated': True,
        'passed': True
    })
    (Path(os.environ['TWINKLE_DEVICE_RUN_DIR']) / 'progress.json').write_text(json.dumps(evidence, indent=2))
