# Copyright (c) ModelScope Contributors. All rights reserved.
"""HTTP integration with both installed SDKs and a CPU mock model."""
import json
import os
import ray
import requests
import socket
import tempfile
import time
import uuid
from on_demand_sdk import unload_tinker_model
from ray import serve

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher
from twinkle_client import init_tinker_client
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import ClientContext
from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel


def main():
    namespace = 'http_' + uuid.uuid4().hex[:8]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    os.environ['TWINKLE_STATE_RAY_NAMESPACE'] = namespace
    os.environ['TWINKLE_DEFAULT_SAVE_DIR'] = tempfile.mkdtemp(prefix='twinkle-training-http-')
    os.environ['TWINKLE_LONG_POLL_TIMEOUT'] = '2'
    os.environ['TWINKLE_PERSISTENCE_MODE'] = 'memory'
    os.environ['PYTHONPATH'] = str(__import__('pathlib').Path.cwd() / 'src')
    ray.init(address='local', namespace=namespace, num_cpus=16, include_dashboard=False, log_to_driver=False)
    config = ServerConfig.model_validate({
        'ray_namespace':
        namespace,
        'http_options': {
            'host': '127.0.0.1',
            'port': port
        },
        'training': {
            'startup_timeout_s': 120,
            'shutdown_timeout_s': 30
        },
        'applications': [{
            'name': 'small',
            'import_path': 'model',
            'route_prefix': '/api/v1/model/small',
            'args': {
                'model_id': 'small',
                'backend': 'mock',
                'device_type': 'cpu',
                'modes': {
                    'lora': {
                        'num_devices': 1,
                        'device_mesh': {
                            'dp_size': 1
                        },
                        'max_loras': 2,
                        'max_replicas': 1
                    },
                    'full': {
                        'num_devices': 1,
                        'device_mesh': {
                            'dp_size': 1
                        }
                    }
                }
            },
        }, {
            'name': 'other',
            'import_path': 'model',
            'route_prefix': '/api/v1/model/other',
            'args': {
                'model_id': 'other',
                'backend': 'mock',
                'device_type': 'cpu',
                'modes': {
                    'full': {
                        'num_devices': 1,
                        'device_mesh': {
                            'dp_size': 1
                        }
                    }
                }
            },
        }, {
            'name': 'gateway',
            'import_path': 'server',
            'route_prefix': '/api/v1',
            'deployments': [{
                'ray_actor_options': {
                    'num_cpus': 0.1
                }
            }]
        }],
    })
    launcher = ServerLauncher(config)
    launcher._start_serve()
    base = f'http://127.0.0.1:{port}'

    def wait_epoch(controller, old_epoch):
        deadline = time.monotonic() + 90
        last_error = None
        while time.monotonic() < deadline:
            try:
                diagnostic = ray.get(controller.diagnostics.remote(), timeout=2)
                if diagnostic['epoch'] != old_epoch:
                    return diagnostic
            except ray.exceptions.RayError as exc:
                last_error = exc
            time.sleep(0.1)
        raise AssertionError(f'Controller did not recover: {last_error}')

    try:
        for spec in config.applications:
            if spec.import_path != 'model':
                launcher._deploy_application(spec)
        launcher._deploy_training()
        controller = launcher._training_controller
        assert ray.get(controller.diagnostics.remote())['instances'] == []
        session = requests.post(base + '/api/v1/twinkle/create_session', json={}, timeout=10).json()['session_id']
        transport = ClientTransport(ClientContext(base_url=base, api_key='probe-owner', session_id=session))
        first = MultiLoraTransformersModel('small', transport=transport)
        second = MultiLoraTransformersModel('small', transport=transport)
        first.add_adapter_to_model('same', None)
        instance = next(
            item for item in ray.get(controller.diagnostics.remote())['instances']
            if first._training_ids['same'] in item['training_ids'])
        bypass = requests.post(
            base + f"/api/v1/training-runtime/{instance['instance_id']}/twinkle/add_adapter_to_model",
            json={
                'adapter_name': 'bypass',
                'config': None
            },
            headers={'Authorization': 'Bearer probe-owner'},
            timeout=10)
        assert bypass.status_code == 400 and 'catalog endpoint' in bypass.text, bypass.text
        second.add_adapter_to_model('same', None)
        old_id = first._training_ids['same']
        assert old_id != second._training_ids['same']
        assert json.loads(first.get_train_configs().result) == {}
        first.zero_grad()
        first.remove_adapter()
        first.add_adapter_to_model('same', None)
        assert old_id != first._training_ids['same']
        first.remove_adapter()
        second.remove_adapter()
        from peft import LoraConfig
        native_lora = MultiLoraTransformersModel('small', transport=transport)
        native_lora.add_adapter_to_model('native-lora', LoraConfig(r=8, target_modules='all-linear'))
        native_lora.zero_grad()
        native_lora.remove_adapter()
        init_tinker_client()
        from tinker import ServiceClient
        service = ServiceClient(base_url=base, api_key='probe-tinker')
        shared1 = service.create_lora_training_client(base_model='small', rank=8)
        shared2 = service.create_lora_training_client(base_model='small', rank=8)
        assert shared1.model_id != shared2.model_id
        assert shared1.get_info().is_lora
        assert shared2.get_info().model_name == 'small'
        from tinker import types
        datum = types.Datum(
            model_input=types.ModelInput.from_ints([1, 2, 3]),
            loss_fn_inputs={
                'target_tokens': types.TensorData(data=[2, 3, 4], dtype='int64', shape=[3]),
                'weights': types.TensorData(data=[1.0, 1.0, 1.0], dtype='float32', shape=[3]),
            })
        assert shared1.forward_backward([datum], 'cross_entropy').result().metrics
        shared1.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        exclusive = service.create_full_training_client(base_model='small')
        assert not exclusive.get_info().is_lora
        assert exclusive.forward_backward([datum], 'cross_entropy').result().metrics
        exclusive.optim_step(types.AdamParams(learning_rate=1e-4)).result()
        assert unload_tinker_model(service, exclusive.model_id).model_id == exclusive.model_id
        assert shared1.get_info().is_lora and shared2.get_info().is_lora
        try:
            service.create_lora_training_client(base_model='small', rank=8)
            raise AssertionError('A third LoRA user bypassed configured capacity')
        except Exception as exc:
            assert 'replica limit' in str(exc).lower(), exc
        assert unload_tinker_model(service, shared2.model_id).model_id == shared2.model_id
        replacement = service.create_lora_training_client(base_model='small', rank=8)
        assert replacement.model_id != shared2.model_id
        assert shared1.get_info().is_lora
        old_epoch = ray.get(controller.diagnostics.remote())['epoch']
        ray.kill(controller, no_restart=False)
        diagnostic = wait_epoch(controller, old_epoch)
        assert not [item for item in diagnostic['instances'] if item['state'] != 'STOPPED']
        stale = requests.post(
            base + '/api/v1/get_info',
            json={
                'model_id': shared1.model_id,
                'type': 'get_info'
            },
            headers={
                'Authorization': 'Bearer probe-tinker',
                'x-request-id': uuid.uuid4().hex
            },
            timeout=10)
        assert stale.status_code == 404, stale.text
        before = ray.available_resources().get('CPU', 0)
        for index in range(10):
            model = MultiLoraTransformersModel('small' if index % 2 == 0 else 'other', transport=transport)
            model.add_adapter_to_model('cycle', None)
            model.zero_grad()
            model.remove_adapter()
            print(json.dumps({'phase': 'cycles', 'completed': index + 1}), flush=True)
        assert ray.available_resources().get('CPU', 0) >= before - 0.02
        final = ray.get(controller.diagnostics.remote())
        assert all(item['state'] == 'STOPPED' for item in final['instances'])
        live = MultiLoraTransformersModel('other', transport=transport)
        live.add_adapter_to_model('state-loss', None)
        ray.kill(ray.get_actor('twinkle_state_actor', namespace=namespace), no_restart=False)
        after_loss = wait_epoch(controller, final['epoch'])
        assert not [item for item in after_loss['instances'] if item['state'] != 'STOPPED']
        session = requests.post(base + '/api/v1/twinkle/create_session', json={}, timeout=10).json()['session_id']
        fresh_transport = ClientTransport(ClientContext(base_url=base, api_key='probe-owner', session_id=session))
        fresh = MultiLoraTransformersModel('other', transport=fresh_transport)
        fresh.add_adapter_to_model('state-loss', None)
        assert fresh._training_ids['state-loss'] != live._training_ids['state-loss']
        fresh.remove_adapter()
        fresh_transport.close()
        print(
            json.dumps({
                'passed': True,
                'native_recreation': True,
                'native_lora': True,
                'tinker_shared': True,
                'tinker_full': True,
                'restart': True,
                'state_loss': True,
                'runtime_lifecycle_rejected': True,
                'cycles': 10,
                'instances': len(final['instances'])
            }))
        transport.close()
    finally:
        if getattr(launcher, '_training_controller', None) is not None:
            ray.get(launcher._training_controller.shutdown.remote(), timeout=60)
            ray.kill(launcher._training_controller, no_restart=True)
        serve.shutdown()
        ray.shutdown()


if __name__ == '__main__':
    main()
