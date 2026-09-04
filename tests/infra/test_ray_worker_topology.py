from twinkle.infra._ray.ray_helper import _copy_worker_env, _get_node_local_topology


def test_copy_worker_env_drops_ray_process_local_flags(monkeypatch):
    monkeypatch.setenv('RAY_JOB_ID', 'parent-job')
    monkeypatch.setenv('RAY_RAYLET_PID', '12345')
    monkeypatch.setenv('RAY_OVERRIDE_NODE_ID_FOR_TESTING', 'parent-node')
    monkeypatch.setenv('RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES', '1')
    monkeypatch.setenv('TWINKLE_TEST_ENV', 'preserved')

    env = _copy_worker_env()

    assert 'RAY_JOB_ID' not in env
    assert 'RAY_RAYLET_PID' not in env
    assert 'RAY_OVERRIDE_NODE_ID_FOR_TESTING' not in env
    assert env['RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES'] == '1'
    assert env['TWINKLE_TEST_ENV'] == 'preserved'


def test_get_node_local_topology_for_single_node_workers():
    placements = [
        {'node_rank': 0},
        {'node_rank': 0},
    ]

    assert _get_node_local_topology(placements) == [
        (0, [0, 1]),
        (1, [0, 1]),
    ]


def test_get_node_local_topology_does_not_assume_contiguous_global_ranks():
    placements = [
        {'node_rank': 0},
        {'node_rank': 1},
        {'node_rank': 0},
    ]

    assert _get_node_local_topology(placements) == [
        (0, [0, 2]),
        (0, [1]),
        (1, [0, 2]),
    ]
