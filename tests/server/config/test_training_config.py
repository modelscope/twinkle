# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import copy
import math

import pytest
from pydantic import ValidationError

from twinkle.server.config import ServerConfig
from twinkle.server.config.training import TrainingConfig
from twinkle.server.exceptions import RequestRejectedError
from twinkle.server.training.catalog import TrainingCatalog


def model_config():
    return {
        'name': 'Qwen/small', 'model_id': 'ms://Qwen/small', 'backend': 'transformers', 'device_type': 'cuda',
        'modes': {
            'lora': {'num_devices': 2, 'device_mesh': {'fsdp_size': 2}},
            'full': {'num_devices': 4, 'device_mesh': {'fsdp_size': 4}},
        },
    }


def model_application(model=None):
    model = dict(model or model_config())
    name = model.pop('name')
    return {'name': 'model-' + name, 'route_prefix': '/api/v1/model/' + name,
            'import_path': 'model', 'args': model}


def test_catalog_defaults_and_mode_selection():
    config = ServerConfig(applications=[model_application()])
    catalog = TrainingCatalog(config.training_config)
    shared = catalog.get('Qwen/small', 'lora')
    exclusive = catalog.get('Qwen/small', 'full')
    assert shared.capacity == 5
    assert exclusive.capacity == 1
    assert shared.config.min_replicas == exclusive.config.min_replicas == 0
    assert shared.config.idle_timeout_s == 1800
    assert config.training.allocation_timeout_s == 3
    assert config.to_yaml_dict()['applications'][0]['args']['model_id'] == 'ms://Qwen/small'
    with pytest.raises(RequestRejectedError, match='Unknown training model'):
        catalog.get('not-in-catalog', 'lora')


def test_existing_yaml_retains_heartbeat_timeout_and_max_lifetime():
    from pathlib import Path
    paths = sorted(Path('cookbook/client/server').glob('**/server_config*.yaml'))
    assert paths
    for path in paths:
        config = ServerConfig.from_yaml(path)
        for app in config.applications:
            if app.import_path != 'model':
                continue
            normalized = config.training_config.models[
                [item.model_id for item in config.training_config.models].index(app.args.model_id)]
            mode = normalized.modes.full if app.args.train_mode == 'full' else normalized.modes.lora
            legacy = app.args.adapter_config or {}
            assert mode.idle_timeout_s == app.args.idle_timeout_s
            assert normalized.legacy_expiration.session_timeout_s == legacy.get('adapter_timeout', 1800)
            assert normalized.legacy_expiration.max_lifetime_s == legacy.get('adapter_max_lifetime', 36000)


@pytest.mark.parametrize('field,value', [
    ('num_devices', 0), ('num_devices', True), ('num_devices', 3),
    ('min_replicas', -1), ('idle_timeout_s', math.inf), ('idle_timeout_s', math.nan),
    ('idle_timeout_s', 0), ('ranks', [0, 1]), ('max_loras', 0),
])
def test_invalid_mode_cannot_reach_allocation(field, value):
    model = model_config()
    model['modes']['lora'][field] = value
    with pytest.raises(ValidationError):
        TrainingConfig(models=[model])


def test_ep_and_vpp_do_not_multiply_physical_device_count():
    model = model_config()
    model['backend'] = 'megatron'
    model['modes']['lora'] = {
        'num_devices': 8,
        'device_mesh': {'tp_size': 2, 'dp_size': 2, 'pp_size': 2, 'ep_size': 2, 'vpp_size': 2},
    }
    assert TrainingConfig(models=[model]).models[0].modes.lora.num_devices == 8


@pytest.mark.parametrize('mutation', ['duplicate', 'no_modes', 'full_lora_field', 'hybrid_megatron', 'mesh_type', 'min_max'])
def test_cross_field_constraints(mutation):
    model = model_config()
    models = [model]
    if mutation == 'duplicate':
        models.append(copy.deepcopy(model))
    elif mutation == 'no_modes':
        model['modes'] = {}
    elif mutation == 'full_lora_field':
        model['modes']['full']['max_loras'] = 2
    elif mutation == 'hybrid_megatron':
        model['backend'] = 'megatron'
        model['modes']['lora']['hybrid'] = {'allocation_path': '/tmp/allocation.json'}
    elif mutation == 'mesh_type':
        model['modes']['lora']['device_mesh']['device_type'] = 'npu'
    elif mutation == 'min_max':
        model['modes']['lora'].update(min_replicas=2, max_replicas=1)
    with pytest.raises(ValidationError):
        TrainingConfig(models=models)


def test_full_only_is_explicit_and_lora_is_unavailable():
    model = model_config()
    del model['modes']['lora']
    catalog = TrainingCatalog(TrainingConfig(models=[model]))
    assert catalog.get('Qwen/small', 'full').capacity == 1
    with pytest.raises(RequestRejectedError, match='does not provide lora'):
        catalog.get('Qwen/small', 'lora')


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
        mode = model.modes.full or model.modes.lora
        assert mode.min_replicas == 1
        assert ServerConfig.model_validate(config.to_yaml_dict()).training_config == config.training_config


def test_legacy_lifecycle_overrides_and_full_instances():
    app = {'name': 'model', 'route_prefix': '/api/v1/model/small', 'import_path': 'model', 'args': {
        'model_id': 'weights', 'backend': 'mock', 'train_mode': 'full',
        'device_group': {'ranks': 2, 'device_type': 'CPU'}, 'device_mesh': {'dp_size': 2},
        'min_replicas': 0, 'max_replicas': 3, 'idle_timeout_s': 17,
        'adapter_config': {'adapter_timeout': 30},
    }, 'deployments': [{'num_replicas': 1}]}
    config = ServerConfig(applications=[app])
    mode = config.training_config.models[0].modes.full
    assert (mode.num_devices, mode.min_replicas, mode.max_replicas, mode.idle_timeout_s) == (2, 0, 3, 17)
    assert config.training_config.models[0].modes.lora is None


def test_extended_modes_do_not_accept_ambiguous_single_mode_settings():
    app = model_application()
    app['args']['train_mode'] = 'full'
    with pytest.raises(ValidationError, match='cannot be mixed'):
        ServerConfig(applications=[app])


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
