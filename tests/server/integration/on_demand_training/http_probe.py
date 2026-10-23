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
from pathlib import Path
from on_demand_sdk import unload_tinker_model
from ray import serve

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher
from twinkle_client import init_tinker_client, init_twinkle_client
from twinkle_client.model.multi_lora_transformers import MultiLoraTransformersModel


def main():
    started = time.monotonic()

    def phase(name, **details):
        print(json.dumps({'phase': name, 'elapsed_s': round(time.monotonic() - started, 2), **details}), flush=True)

    namespace = 'http_' + uuid.uuid4().hex[:8]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    os.environ['TWINKLE_STATE_RAY_NAMESPACE'] = namespace
    os.environ['TWINKLE_DEFAULT_SAVE_DIR'] = tempfile.mkdtemp(prefix='twinkle-training-http-')
    os.environ['TWINKLE_LONG_POLL_TIMEOUT'] = '2'
    os.environ['TWINKLE_PERSISTENCE_MODE'] = 'memory'
    os.environ['PYTHONPATH'] = str(__import__('pathlib').Path.cwd() / 'src')
    ray.init(
        address='local',
        namespace=namespace,
        num_cpus=8,
        _temp_dir='/tmp/twh-' + namespace[-8:],
        include_dashboard=False,
        log_to_driver=False)
    config = ServerConfig.model_validate({
        'ray_namespace':
        namespace,
        'http_options': {
            'host': '127.0.0.1',
            'port': port
        },
        'resources': {
            'startup_timeout_s': 600,
            'shutdown_timeout_s': 180
        },
        'applications': [{
            'name': 'small',
            'import_path': 'model',
            'route_prefix': '/api/v1/model/small',
            'args': {
                'model_id': 'small',
                'backend': 'mock',
                'train_mode': 'lora',
                'device_group': {'ranks': 1, 'device_type': 'cpu'},
                'device_mesh': {'dp_size': 1},
                'max_loras': 2,
                'max_replicas': 1,
                'adapter_config': {'adapter_timeout': 30},
            },
        }, {
            'name': 'other',
            'import_path': 'model',
            'route_prefix': '/api/v1/model/other',
            'args': {
                'model_id': 'small',
                'backend': 'mock',
                'train_mode': 'full',
                'device_group': {'ranks': 1, 'device_type': 'cpu'},
                'device_mesh': {'dp_size': 1},
                'min_replicas': 0,
                'max_replicas': 1,
                'adapter_config': {'adapter_timeout': 30},
            },
            'deployments': [{
                'max_ongoing_requests': 8,
                'autoscaling_config': {'min_replicas': 1, 'max_replicas': 1, 'target_ongoing_requests': 4},
                'ray_actor_options': {'num_cpus': 0.1, 'runtime_env': {'env_vars': {'LEGACY_CONFIG_PROBE': '1'}}},
            }],
        }, {
            'name': 'routed', 'import_path': 'model', 'route_prefix': '/api/v1/model/routed',
            'args': {
                'model_id': 'small', 'backend': 'mock', 'train_mode': 'lora',
                'device_group': {'ranks': 1, 'device_type': 'cpu'}, 'device_mesh': {'dp_size': 1},
                'max_loras': 1, 'max_replicas': 2,
                'adapter_config': {'adapter_timeout': 30},
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
    native = fresh_native = service = None

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
        phase('server_ready')
        native = init_twinkle_client(base_url=base, api_key='probe-owner')
        transport = native.transport
        first = MultiLoraTransformersModel('small', transport=transport)
        second = MultiLoraTransformersModel('small', transport=transport)
        from peft import LoraConfig

        def assert_separate_instances(identities):
            diagnostic = ray.get(controller.diagnostics.remote())
            owners = {identity: item['instance_id'] for item in diagnostic['instances']
                      for identity in item['training_ids'] if identity in identities}
            assert set(owners) == set(identities) and len(set(owners.values())) == 2, owners
            return owners

        routed = [MultiLoraTransformersModel('routed', transport=transport) for _ in range(2)]
        for client in routed:
            client.add_adapter_to_model('same', LoraConfig(r=8, target_modules='all-linear'))
        native_owners = assert_separate_instances([client._training_ids['same'] for client in routed])
        for _ in range(3):
            for client in reversed(routed):
                assert json.loads(client.get_train_configs().result) == {}
                client.zero_grad()
        for client in routed:
            client.remove_adapter()
        phase('native_routing_verified')
        first.add_adapter_to_model('same', LoraConfig(r=8, target_modules='all-linear'))
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
        second.add_adapter_to_model('same', LoraConfig(r=8, target_modules='all-linear'))
        old_id = first._training_ids['same']
        assert old_id != second._training_ids['same']
        assert json.loads(first.get_train_configs().result) == {}
        first.zero_grad()
        first.remove_adapter()
        first.add_adapter_to_model('same', LoraConfig(r=8, target_modules='all-linear'))
        assert old_id != first._training_ids['same']
        first.remove_adapter()
        second.remove_adapter()
        native_lora = MultiLoraTransformersModel('small', transport=transport)
        native_lora.add_adapter_to_model('native-lora', LoraConfig(r=8, target_modules='all-linear'))
        native_lora.zero_grad()
        native_lora.remove_adapter()
        phase('native_lifecycle_verified')
        init_tinker_client()
        from tinker import ServiceClient
        service = ServiceClient(base_url=base, api_key='probe-tinker')
        routed_tinker = [service.create_lora_training_client(base_model='routed', rank=8) for _ in range(2)]
        tinker_owners = assert_separate_instances([client.model_id for client in routed_tinker])
        for _ in range(3):
            for client in reversed(routed_tinker):
                assert client.get_info().model_id == client.model_id
        for client in routed_tinker:
            assert unload_tinker_model(service, client.model_id).model_id == client.model_id
        phase('tinker_routing_verified')
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
        exclusive = service.create_full_training_client(base_model='other')
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
        phase('controller_recovered')
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
        for index in range(3):
            model = MultiLoraTransformersModel('small' if index % 2 == 0 else 'other', transport=transport)
            model.add_adapter_to_model('cycle', LoraConfig(r=8, target_modules='all-linear')
                                       if index % 2 == 0 else None)
            model.zero_grad()
            model.remove_adapter()
            phase('cycles', completed=index + 1)
        assert ray.available_resources().get('CPU', 0) >= before - 0.02
        final = ray.get(controller.diagnostics.remote())
        assert all(item['state'] == 'STOPPED' for item in final['instances'])
        live = MultiLoraTransformersModel('other', transport=transport)
        live.add_adapter_to_model('state-loss', None)
        native.close()
        ray.kill(ray.get_actor('twinkle_state_actor', namespace=namespace), no_restart=False)
        after_loss = wait_epoch(controller, final['epoch'])
        assert not [item for item in after_loss['instances'] if item['state'] != 'STOPPED']
        fresh_native = init_twinkle_client(base_url=base, api_key='probe-owner')
        fresh_transport = fresh_native.transport
        fresh = MultiLoraTransformersModel('other', transport=fresh_transport)
        fresh.add_adapter_to_model('state-loss', None)
        assert fresh._training_ids['state-loss'] != live._training_ids['state-loss']
        fresh.remove_adapter()
        phase('state_loss_recovered')
        evidence = {
            'passed': True,
            'native_recreation': True,
            'native_stable_routing': native_owners,
            'tinker_stable_routing': tinker_owners,
            'native_lora': True,
            'tinker_shared': True,
            'tinker_full': True,
            'restart': True,
            'state_loss': True,
            'runtime_lifecycle_rejected': True,
            'cycles': 3,
            'instances': len(final['instances'])
        }
        if result_path := os.environ.get('TWINKLE_TEST_HTTP_RESULT'):
            Path(result_path).write_text(json.dumps(evidence))
        print(json.dumps(evidence), flush=True)
    finally:
        for client in (native, fresh_native):
            if client is not None:
                client.close()
        if service is not None:
            service.holder.close()
        if getattr(launcher, '_training_controller', None) is not None:
            ray.get(launcher._training_controller.shutdown.remote(), timeout=180)
            ray.kill(launcher._training_controller, no_restart=True)
        serve.shutdown()
        ray.shutdown()


if __name__ == '__main__':
    main()
