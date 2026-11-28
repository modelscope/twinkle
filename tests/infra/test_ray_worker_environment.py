# Copyright (c) ModelScope Contributors. All rights reserved.
from unittest.mock import Mock

import pytest

pytest.importorskip('ray')

from twinkle.infra._ray.ray_helper import RayHelper


def test_worker_uses_destination_raylet_pid(monkeypatch):
    import ray

    monkeypatch.setenv('RAY_RAYLET_PID', '12345')
    monkeypatch.setenv('TWINKLE_RAY_REGISTRY_NAME', 'test-registry')
    # 58b2b979: placements carry node_ip (published by ResourceManager._publish_addresses)
    # and create_workers awaits worker._twinkle_ready via ray.get. This test drives the
    # env-var seam with a Mock worker, so supply the address directly and resolve refs
    # by identity -- the assertions below (raylet PID / registry / RANK) are unchanged.
    placements = [{'placement_group': None, 'node_ip': '127.0.0.1'}]
    monkeypatch.setattr(ray, 'get', lambda value, **kwargs: value)
    monkeypatch.setattr(RayHelper, 'group_info', lambda group: {'device_type': 'CPU', 'placements': placements})
    monkeypatch.setattr(RayHelper, 'get_master_id_port', lambda group: ('127.0.0.1', 1234))
    monkeypatch.setattr(RayHelper, '_created_actors', [])
    worker = Mock()
    monkeypatch.setattr(RayHelper, '_get_remote_component', lambda worker_cls: worker)

    RayHelper.create_workers(object, 'test', 'all', instance_id='test-instance')

    env = worker.options.call_args.kwargs['runtime_env']['env_vars']
    assert 'RAY_RAYLET_PID' not in env
    assert env['TWINKLE_RAY_REGISTRY_NAME'] == 'test-registry'
    assert env['RANK'] == '0'
