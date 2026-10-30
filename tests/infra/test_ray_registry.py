# Copyright (c) ModelScope Contributors. All rights reserved.
"""Registry isolation must not reserve a CPU for ordinary non-server use."""
import subprocess
import sys
import textwrap


def test_registry_cpu_reservations_and_instance_isolation():
    # Own a subprocess and cluster so the CPU assertions do not count actors
    # belonging to other tests or reset their process-global RayHelper cache.
    code = textwrap.dedent('''
        import os
        import time
        import uuid
        import ray
        from twinkle.infra._ray.ray_helper import RayHelper

        namespace = 'registry-' + uuid.uuid4().hex[:8]
        ray.init(address='local', num_cpus=2, namespace=namespace,
                 include_dashboard=False, log_to_driver=False,
                 _temp_dir='/tmp/twreg-' + uuid.uuid4().hex[:6])

        def registry(name, managed):
            RayHelper._registry = None
            os.environ['TWINKLE_RAY_REGISTRY_NAME'] = name
            if managed:
                os.environ['TWINKLE_RAY_NAMESPACE'] = namespace
            else:
                os.environ.pop('TWINKLE_RAY_NAMESPACE', None)
            RayHelper.init_registry()
            actor = RayHelper._registry
            ray.get(actor.add_config.remote('owner', name))
            return actor

        def assert_free_cpus(expected):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                available = ray.available_resources().get('CPU', 0)
                if abs(available - expected) < 1e-6:
                    return
                time.sleep(.01)
            raise AssertionError((available, expected))

        try:
            ordinary = registry('ordinary', False)
            assert_free_cpus(2)
            ray.kill(ordinary, no_restart=True)
            left = registry('left', True)
            right = registry('right', True)
            assert_free_cpus(1.98)
            assert ray.get(left.get_config.remote('owner')) == 'left'
            assert ray.get(right.get_config.remote('owner')) == 'right'
            ray.kill(left, no_restart=True)
            assert_free_cpus(1.99)
            assert ray.get(right.get_config.remote('owner')) == 'right'
            ray.kill(right, no_restart=True)
            assert_free_cpus(2)
        finally:
            ray.shutdown()
    ''')
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
