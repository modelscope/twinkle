# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import copy
import math

import pytest
from pydantic import ValidationError

from twinkle.server.config import ServerConfig
from twinkle.server.config.training import TrainingConfig
from twinkle.server.exceptions import RequestRejectedError


def model_config(mode='lora'):
    devices = 2 if mode == 'lora' else 4
    return {
        'name': 'Qwen/small' if mode == 'lora' else 'Qwen/small-full',
        'model_id': 'ms://Qwen/small', 'backend': 'transformers', 'device_type': 'cuda',
        'mode': mode, 'config': {'num_devices': devices, 'device_mesh': {'fsdp_size': devices}},
    }


def model_application():
    return {'name': 'model-small', 'route_prefix': '/api/v1/model/Qwen/small',
            'import_path': 'model', 'args': {
                'model_id': 'ms://Qwen/small', 'backend': 'transformers',
                'device_group': {'ranks': 2, 'device_type': 'cuda'},
                'device_mesh': {'fsdp_size': 2},
            }}


def test_catalog_defaults_and_mode_selection():
    config = TrainingConfig(models=[model_config(), model_config('full')])
    catalog = config
    shared = catalog.get('Qwen/small', 'lora')
    exclusive = catalog.get('Qwen/small-full', 'full')
    assert shared.capacity == 5
    assert exclusive.capacity == 1
    assert shared.config.min_replicas == exclusive.config.min_replicas == 0
    assert shared.config.idle_timeout_s == 1800
    assert config.allocation_timeout_s == 3
    with pytest.raises(RequestRejectedError, match='Unknown training model'):
        catalog.get('not-in-catalog', 'lora')


def test_existing_yaml_uses_explicit_instance_limits_and_business_idle():
    from pathlib import Path
    paths = sorted(Path('cookbook/client/server').glob('**/server_config*.yaml'))
    assert paths
    for path in paths:
        config = ServerConfig.from_yaml(path)
        applications = [app for app in config.applications if app.import_path == 'model']
        for app, model in zip(applications, config.training_config.models):
            assert model.config.idle_timeout_s == app.args.idle_timeout_s
            assert model.config.min_replicas == app.args.min_replicas
            assert model.config.max_replicas == app.args.max_replicas
            assert model.config.adapter_config == app.args.adapter_config


@pytest.mark.parametrize('field,value', [
    ('num_devices', 0), ('num_devices', True), ('num_devices', 3),
    ('min_replicas', -1), ('idle_timeout_s', math.inf), ('idle_timeout_s', math.nan),
    ('idle_timeout_s', 0), ('ranks', [0, 1]), ('max_loras', 0),
])
def test_invalid_mode_cannot_reach_allocation(field, value):
    model = model_config()
    model['config'][field] = value
    with pytest.raises(ValidationError):
        TrainingConfig(models=[model])


def test_ep_and_vpp_do_not_multiply_physical_device_count():
    model = model_config()
    model['backend'] = 'megatron'
    model['config'] = {
        'num_devices': 8,
        'device_mesh': {'tp_size': 2, 'dp_size': 2, 'pp_size': 2, 'ep_size': 2, 'vpp_size': 2},
    }
    assert TrainingConfig(models=[model]).models[0].config.num_devices == 8


@pytest.mark.parametrize('mutation', ['duplicate', 'missing_mode', 'full_lora_field', 'hybrid_megatron', 'mesh_type', 'min_max'])
def test_cross_field_constraints(mutation):
    model = model_config()
    models = [model]
    if mutation == 'duplicate':
        models.append(copy.deepcopy(model))
    elif mutation == 'missing_mode':
        del model['mode']
    elif mutation == 'full_lora_field':
        model = model_config('full')
        model['config']['max_loras'] = 2
        models = [model]
    elif mutation == 'hybrid_megatron':
        model['backend'] = 'megatron'
        model['config']['hybrid'] = {'allocation_path': '/tmp/allocation.json'}
    elif mutation == 'mesh_type':
        model['config']['device_mesh']['device_type'] = 'npu'
    elif mutation == 'min_max':
        model['config'].update(min_replicas=2, max_replicas=1)
    with pytest.raises(ValidationError):
        TrainingConfig(models=models)


def test_full_only_is_explicit_and_lora_is_unavailable():
    model = model_config('full')
    catalog = TrainingConfig(models=[model])
    assert catalog.get('Qwen/small-full', 'full').capacity == 1
    with pytest.raises(RequestRejectedError, match='does not provide lora'):
        catalog.get('Qwen/small-full', 'lora')


def test_training_namespace_conflicts_are_rejected():
    with pytest.raises(ValidationError, match='unique'):
        ServerConfig(applications=[model_application(), model_application()])
    with pytest.raises(ValidationError, match='route conflicts'):
        ServerConfig(applications=[model_application(), {
            'name': 'processor', 'import_path': 'processor', 'route_prefix': '/api/v1/model/Qwen/small',
        }])
    with pytest.raises(ValidationError, match='models'):
        ServerConfig(training={'models': [model_config()]})


def test_existing_yaml_examples_keep_their_model_entries(monkeypatch):
    from pathlib import Path
    monkeypatch.setenv('TWINKLE_LOCAL_MODEL_PATH', '/tmp/model-cache')
    paths = list(Path('cookbook/client/server').rglob('server_config*.yaml'))
    paths += list(Path('cookbook/client/async_rl').glob('server_config.yaml'))
    paths += list(Path('tests/server/config').glob('server_config*e2e*.yaml'))
    paths += list(Path('tests/server/fixtures').glob('server_config_mock*.yaml'))
    assert len(paths) == 9
    for path in paths:
        config = ServerConfig.from_yaml(path)
        models = [app for app in config.applications if app.import_path == 'model']
        assert len(models) == len(config.training_config.models) == 1
        model = config.training_config.models[0]
        assert model.model_id == models[0].args.model_id
        assert model.route_prefix == models[0].route_prefix
        mode = model.config
        original = models[0]
        expected_min = original.args.min_replicas
        assert mode.min_replicas == expected_min
        assert ServerConfig.model_validate(config.to_yaml_dict()).training_config == config.training_config


def test_configured_lifecycle_overrides_and_full_instances():
    app = {'name': 'model', 'route_prefix': '/api/v1/model/small', 'import_path': 'model', 'args': {
        'model_id': 'weights', 'backend': 'mock', 'train_mode': 'full',
        'device_group': {'ranks': 2, 'device_type': 'CPU'}, 'device_mesh': {'dp_size': 2},
        'min_replicas': 0, 'max_replicas': 3, 'idle_timeout_s': 17,
    }, 'deployments': [{'num_replicas': 1}]}
    config = ServerConfig(applications=[app])
    mode = config.training_config.models[0].config
    assert (mode.num_devices, mode.min_replicas, mode.max_replicas, mode.idle_timeout_s) == (2, 0, 3, 17)
    assert config.training_config.models[0].mode == 'full'


def test_separate_model_applications_select_lora_and_full():
    shared = model_application()
    exclusive = copy.deepcopy(shared)
    exclusive.update(name='model-small-full', route_prefix='/api/v1/model/Qwen/small-full')
    exclusive['args'].update(train_mode='full', min_replicas=0, max_replicas=2, idle_timeout_s=60)
    config = ServerConfig(applications=[shared, exclusive])
    catalog = config.training_config
    assert catalog.get('Qwen/small', 'lora').capacity == 5
    full = catalog.get('Qwen/small-full', 'full')
    assert full.capacity == 1
    assert (full.config.min_replicas, full.config.max_replicas, full.config.idle_timeout_s) == (0, 2, 60)
    with pytest.raises(RequestRejectedError, match='does not provide full'):
        catalog.get('Qwen/small', 'full')
    assert ServerConfig.model_validate(config.to_yaml_dict()).training_config == config.training_config


def test_launcher_keeps_training_and_sampling_capabilities_separate(monkeypatch):
    from ray import serve
    from twinkle.server.launcher.server_launcher import ServerLauncher
    config = ServerConfig(applications=[model_application(), {
        'name': 'gateway', 'import_path': 'server', 'route_prefix': '/api/v1',
    }, {
        'name': 'sampler', 'import_path': 'sampler', 'route_prefix': '/api/v1/sampler/teacher',
        'args': {'model_id': 'teacher', 'device_group': {}, 'device_mesh': {}, 'sampler_type': 'mock'},
    }])
    captured = {}
    def builder(**kwargs):
        captured.update(kwargs)
        return 'app'
    launcher = ServerLauncher(config)
    monkeypatch.setattr(launcher, '_resolve_builder', lambda path: builder)
    monkeypatch.setattr(serve, 'run', lambda *args, **kwargs: None)
    launcher._deploy_application(config.applications[1])
    assert captured['training_models'] == ['Qwen/small']
    assert captured['supported_models'] == ['teacher']


def configured_application():
    return {'name': 'model', 'route_prefix': '/api/v1/model/small', 'import_path': 'model', 'args': {
        'model_id': 'weights', 'backend': 'mock', 'train_mode': 'lora', 'max_length': 128,
        'device_group': {'ranks': 2, 'device_type': 'cpu'}, 'device_mesh': {'dp_size': 2},
        'env_vars': {'USER_OPTION': 'args-wins'}, 'min_replicas': 1, 'max_replicas': 2,
    }, 'deployments': [{
        'name': 'ModelManagement', 'max_ongoing_requests': 17,
        'autoscaling_config': {'min_replicas': 1, 'max_replicas': 2, 'target_ongoing_requests': 16},
        'ray_actor_options': {'num_cpus': 2, 'runtime_env': {
            'env_vars': {'USER_OPTION': 'deployment', 'OTHER_OPTION': 'retained'}, 'pip': ['example==1.0']
        }},
    }]}


@pytest.mark.parametrize('extra', [
    {'modes': None},
    {'modes': {'lora': {'num_devices': 2, 'device_mesh': {'dp_size': 2}}}},
    {'device_type': 'cpu'},
])
def test_model_yaml_rejects_alternative_resource_and_mode_syntax(extra):
    app = configured_application()
    app['args'].update(extra)
    with pytest.raises(ValidationError, match='Extra inputs are not permitted'):
        ServerConfig(applications=[app])


@pytest.mark.parametrize('mode', ['lora', 'full'])
def test_adapter_lifecycle_yaml_reaches_controller_and_round_trips(mode):
    app = model_application()
    policy = {'adapter_timeout': 30, 'adapter_max_lifetime': 36000}
    app['args'].update(train_mode=mode, adapter_config=policy, idle_timeout_s=600)
    config = ServerConfig(applications=[app])
    resolved = config.training_config.models[0].config
    assert resolved.adapter_config.model_dump() == policy
    assert resolved.idle_timeout_s == 600
    restored = ServerConfig.model_validate(config.to_yaml_dict())
    assert restored.training_config == config.training_config


@pytest.mark.parametrize('policy', [
    {'adapter_timeout': 0}, {'adapter_timeout': math.inf}, {'adapter_timeout': math.nan},
    {'adapter_max_lifetime': -1}, {'adapter_max_lifetime': math.inf}, {'unknown': 30},
])
def test_invalid_adapter_lifecycle_is_rejected_before_allocation(policy):
    app = model_application()
    app['args']['adapter_config'] = policy
    with pytest.raises(ValidationError):
        ServerConfig(applications=[app])


def test_configured_options_reach_cpu_router_without_changing_yaml(monkeypatch):
    from types import SimpleNamespace
    from twinkle.server.launcher.server_launcher import ServerLauncher
    from twinkle.server.model.pool.actor import TrainingControllerActor
    import ray
    from ray import serve
    config = ServerConfig(applications=[configured_application()])
    remote = SimpleNamespace(initialize=SimpleNamespace(remote=lambda: None))
    monkeypatch.setattr(TrainingControllerActor, 'options',
                        lambda **kwargs: SimpleNamespace(remote=lambda *args: remote))
    monkeypatch.setattr(ray, 'get', lambda *args, **kwargs: None)
    captured = {}

    def router(*args):
        captured.update(args[-1])
        return 'app'

    monkeypatch.setattr('twinkle.server.model.pool.router.build_training_router', router)
    monkeypatch.setattr(serve, 'run', lambda *args, **kwargs: None)
    ServerLauncher(config)._deploy_training()
    assert captured['max_ongoing_requests'] == 17
    assert captured['autoscaling_config']['target_ongoing_requests'] == 16
    actor = captured['ray_actor_options']
    assert actor['num_cpus'] == 2 and actor['runtime_env']['pip'] == ['example==1.0']
    assert actor['runtime_env']['env_vars']['USER_OPTION'] == 'args-wins'
    assert actor['runtime_env']['env_vars']['OTHER_OPTION'] == 'retained'
    assert config.to_yaml_dict()['applications'][0]['deployments'][0] == configured_application()['deployments'][0]


@pytest.mark.asyncio
async def test_configured_options_reach_runtime_with_one_controller_owned_replica(monkeypatch):
    import importlib
    from twinkle.server.state.training_records import InstanceRecord
    from twinkle.server.model.pool.driver import RayInstanceDriver
    from ray import serve
    config = ServerConfig(applications=[configured_application()])
    model = config.training_config.models[0]
    driver = RayInstanceDriver(config.training_config, None, 'test', 'controller')
    instance = InstanceRecord(instance_id='instance', controller_epoch='epoch', base_model=model.name,
                              mode='lora', startup_deadline=0, devices=['0', '1'])
    captured = {}

    def builder(**kwargs):
        captured.update(kwargs)
        return 'app'

    monkeypatch.setattr('twinkle.server.model.app.build_model_app', builder)
    monkeypatch.setattr(importlib.import_module('ray.util.placement_group'), 'get_placement_group', lambda name: 'pg')
    monkeypatch.setattr(serve, 'run', lambda *args, **kwargs: None)
    await driver.deploy(instance, driver.config.get(model.name, 'lora'))
    options = captured['deploy_options']
    assert options['num_replicas'] == 1 and 'autoscaling_config' not in options
    assert options['max_ongoing_requests'] == 17
    assert options['ray_actor_options']['num_cpus'] == 2
    assert options['ray_actor_options']['runtime_env']['pip'] == ['example==1.0']
    assert captured['max_length'] == 128
    assert captured['nproc_per_node'] == 2
    assert captured['queue_config'] == model.config.queue_config
    assert 'adapter_config' not in captured, 'Controller lifecycle policy must not reach model constructors'
    assert not driver.runtime.deployments


def test_mutated_configuration_does_not_reuse_a_stale_catalog():
    config = ServerConfig(applications=[configured_application()])
    assert config.training_config.models[0].config.min_replicas == 1
    config.applications[0].args.min_replicas = 0
    assert config.training_config.models[0].config.min_replicas == 0


def test_router_replica_limits_do_not_implicitly_set_training_instance_limits():
    app = configured_application()
    app['args'].pop('min_replicas')
    app['args'].pop('max_replicas')
    model = ServerConfig(applications=[app]).training_config.models[0]
    assert model.config.min_replicas == 0 and model.config.max_replicas is None
    assert model.deploy_options['autoscaling_config']['min_replicas'] == 1


def test_l20d_candidate_is_lora_only_and_bounds_total_devices():
    config = ServerConfig.from_yaml('cookbook/client/server/megatron/server_config.yaml')
    model = config.training_config.models[0]
    mode = model.config
    assert model.mode == 'lora'
    assert (mode.num_devices, mode.min_replicas, mode.max_replicas, mode.max_loras) == (2, 0, 2, 3)
    assert mode.device_mesh.world_size() == 2 and mode.device_mesh.pp_size == 2 and mode.idle_timeout_s == 600
    sampler = next(app for app in config.applications if app.import_path == 'sampler')
    assert sampler.args.device_group['ranks'] == 1
    assert sampler.args.device_mesh['dp_size'] == 1 and 'tp_size' not in sampler.args.device_mesh
    assert 'tensor_parallel_size' not in sampler.args.engine_args
    assert (sampler.args.min_replicas, sampler.args.max_replicas, sampler.args.target_ongoing_requests) == (2, 4, 16)
    assert mode.num_devices * mode.max_replicas + sampler.args.device_group['ranks'] * sampler.args.max_replicas == 8
    assert mode.max_loras * mode.max_replicas == 6
    with pytest.raises(RequestRejectedError, match='does not provide full'):
        config.training_config.get(model.name, 'full')


@pytest.mark.parametrize('mode', ['lora', 'full'])
@pytest.mark.parametrize('yaml_entrypoint', [False, True])
def test_single_device_megatron_is_rejected_before_allocation(mode, yaml_entrypoint):
    if yaml_entrypoint:
        app = configured_application()
        app['args'].update(backend='megatron', train_mode=mode, device_group={'ranks': 1, 'device_type': 'cuda'},
                           device_mesh={'dp_size': 1})
    else:
        model = model_config(mode)
        model['backend'] = 'megatron'
        model['config'] = {'num_devices': 1, 'device_mesh': {'dp_size': 1}}
    with pytest.raises(ValidationError, match='at least 2 devices'):
        if yaml_entrypoint:
            ServerConfig(applications=[app])
        else:
            TrainingConfig(models=[model])


def test_transformers_still_accepts_single_device():
    app = model_application()
    app['args'].update(device_group={'ranks': 1, 'device_type': 'cuda'}, device_mesh={'dp_size': 1})
    assert ServerConfig(applications=[app]).training_config.models[0].config.num_devices == 1
