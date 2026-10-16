# Copyright (c) ModelScope Contributors. All rights reserved.
import socket
import uuid

import pytest
import ray

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher
from twinkle.server.training.actor import controller_name


def test_failed_initialization_does_not_leave_an_actor_that_blocks_a_retry():
    namespace = 'launcher-retry-' + uuid.uuid4().hex
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    devices = int(ray.cluster_resources()['CPU']) + 1
    payload = {
        'ray_namespace': namespace, 'http_options': {'host': '127.0.0.1', 'port': port},
        'applications': [{'name': 'retry', 'route_prefix': '/api/v1/model/retry', 'import_path': 'model',
                          'args': {'model_id': 'mock', 'backend': 'mock', 'device_type': 'cpu',
                                   'modes': {'full': {'num_devices': devices, 'device_mesh': {'dp_size': devices},
                                                      'min_replicas': 1}}}}],
    }
    launcher = ServerLauncher(ServerConfig.model_validate(payload))
    launcher._start_serve()
    try:
        with pytest.raises(ray.exceptions.RayTaskError, match='No single-node allocation'):
            launcher._deploy_training()
        with pytest.raises(ValueError):
            ray.get_actor(controller_name(namespace), namespace=namespace)
        payload['applications'][0]['args']['modes']['full']['min_replicas'] = 0
        launcher.config = ServerConfig.model_validate(payload)
        launcher._deploy_training()
        diagnostic = ray.get(launcher._training_controller.diagnostics.remote())
        assert diagnostic['ready'] and diagnostic['instances'] == []
    finally:
        launcher._shutdown()
