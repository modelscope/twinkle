# Copyright (c) ModelScope Contributors. All rights reserved.
"""Run the full training matrix on an isolated on-demand server."""
import argparse
import atexit
import json
import math
import os
import ray
import socket
import subprocess
import sys
import time
import uuid
import yaml
from pathlib import Path
from ray import serve
from xml.etree import ElementTree

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from twinkle.server.config import ServerConfig
from twinkle.server.launcher.server_launcher import ServerLauncher


def validate_controller_restart(base_url, controller, model_name):
    """Train, restart, restore optimizer checkpoints and recover a failed worker."""
    import numpy as np
    from ray.util import list_named_actors
    from tests.server.integration.on_demand_training.acceptance_support import (
        TrainingHandle, assert_released, wait_until)

    namespace = ray.get_runtime_context().namespace
    old = [TrainingHandle(base_url, model_name, protocol, 'restart-' + protocol)
           for protocol in ('twinkle', 'tinker')]
    snapshots = []
    try:
        for handle in old:
            # Two calls end the native GA=2 workload on an optimizer boundary.
            handle.step()
            handle.step()
            path = handle.save('restart-state')
            values, consumed = handle.evaluate(), handle.consumed_samples
            handle.step()
            handle.step()
            continued = handle.evaluate()
            assert np.max(np.abs(continued - values)) > 2e-3, 'The reference optimizer did not change logprobs'
            snapshots.append((path, values, consumed, continued))
        old_epoch = ray.get(controller.diagnostics.remote())['epoch']
        ray.kill(controller, no_restart=False)

        def restarted():
            try:
                diagnostic = ray.get(controller.diagnostics.remote(), timeout=10)
                return diagnostic if diagnostic['epoch'] != old_epoch else None
            except ray.exceptions.RayError:
                return None

        diagnostic = wait_until(restarted, 600, 'Controller restart timed out')
        assert_released(controller, namespace)
        for handle in old:
            handle.assert_stale()
        restored = []
        try:
            # Keep restored peers on one runtime: checkpoint/optimizer isolation
            # is checked without reloading the base model for each protocol.
            optimizer_errors = {}
            for protocol, (path, values, consumed, continued) in zip(('twinkle', 'tinker'), snapshots):
                fresh = TrainingHandle(base_url, model_name, protocol, 'restored-' + protocol)
                restored.append(fresh)
                progress = fresh.resume(path)
                if protocol == 'twinkle':
                    assert progress['consumed_train_samples'] == consumed
                np.testing.assert_allclose(fresh.evaluate(), values, rtol=0, atol=2e-3)
                fresh.step()
                fresh.step()
                resumed = fresh.evaluate()
                np.testing.assert_allclose(resumed, continued, rtol=0, atol=2e-3)
                optimizer_errors[protocol] = {
                    'max_error': float(np.max(np.abs(resumed - continued))),
                    'checkpoint_logps': values.tolist(), 'continued_logps': continued.tolist(),
                    'resumed_logps': resumed.tolist(),
                }
            active = next(row for row in ray.get(controller.diagnostics.remote())['instances']
                          if restored[0].identity in row['training_ids'])
            assert set(active['training_ids']) == {handle.identity for handle in restored}
            owned = [row for row in list_named_actors(all_namespaces=True)
                     if row['namespace'] == namespace and active['instance_id'] in row['name']
                     and row['name'].startswith('model-') and not row['name'].endswith('-registry')]
            assert owned, 'No model worker available for fault injection'
            ray.kill(ray.get_actor(owned[0]['name'], namespace=namespace), no_restart=True)
            wait_until(lambda: all(row['state'] == 'STOPPED'
                                  for row in ray.get(controller.diagnostics.remote())['instances']),
                       600, 'Worker failure cleanup timed out')
            for handle in restored:
                handle.assert_stale()
            assert_released(controller, namespace)
        finally:
            for handle in restored:
                handle.close(stale=True)
        fresh = TrainingHandle(base_url, model_name, 'twinkle', 'after-worker-fault')
        try:
            fresh.step()
        finally:
            fresh.close()
    finally:
        for handle in old:
            handle.close(stale=True)
    return {'epoch': diagnostic['epoch'], 'worker_fault_recovered': True, 'checkpoint_restored': True,
            'optimizer_continuation': optimizer_errors}


def validate_idempotent_create(base_url, controller, model_name):
    """Concurrent HTTP retries share one receipt, handle and allocation."""
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from peft import LoraConfig
    from twinkle.protocol.serialize import serialize_object
    from twinkle.protocol.types.model import TrainingHandleResponse
    from twinkle_client._future import resolve_response
    from twinkle_client.http import ClientTransport
    from twinkle_client.http.context import ClientContext
    from tests.server.integration.on_demand_training.acceptance_support import assert_released

    session_reply = requests.post(base_url + '/api/v1/twinkle/create_session', json={}, timeout=30)
    session_reply.raise_for_status()
    session = session_reply.json()['session_id']
    transport = ClientTransport(ClientContext(base_url=base_url, api_key='idempotency', session_id=session))
    headers = {'Authorization': 'Bearer idempotency', 'X-Twinkle-Session-Id': session}
    payload = {'client_create_id': uuid.uuid4().hex, 'adapter_name': 'duplicate',
               'config': serialize_object(LoraConfig(r=8, target_modules='all-linear')),
               'gradient_accumulation_steps': 2}
    endpoint = base_url + '/api/v1/model/' + model_name + '/twinkle/add_adapter_to_model'

    def submit(_):
        response = requests.post(endpoint, json=payload,
                                 headers={**headers, 'x-request-id': uuid.uuid4().hex}, timeout=60)
        response.raise_for_status()
        return response

    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            replies = list(executor.map(submit, range(8)))
        assert len({reply.json()['request_id'] for reply in replies}) == 1
        handle = resolve_response(replies[0], TrainingHandleResponse, transport=transport)
        live = [row for row in ray.get(controller.diagnostics.remote())['instances'] if row['state'] != 'STOPPED']
        assert len(live) == 1 and live[0]['training_ids'] == [handle.training_id]
        reply = requests.post(base_url + '/api/v1/model/' + model_name + '/twinkle/remove_adapter',
                              json={'training_id': handle.training_id, 'adapter_name': 'duplicate'},
                              headers={**headers, 'x-request-id': uuid.uuid4().hex}, timeout=60)
        reply.raise_for_status()
        resolve_response(reply, None, transport=transport)
        assert_released(controller, ray.get_runtime_context().namespace)
    finally:
        transport.close()
    return True


def validate_capacity(base_url, controller, model_name, full_model_name, capacity):
    """Assert shared LoRA capacity and full-training exclusivity on real devices."""
    from tests.server.integration.on_demand_training.acceptance_support import TrainingHandle, assert_released
    handles = []
    try:
        for index in range(capacity):
            handles.append(TrainingHandle(base_url, model_name, ('twinkle', 'tinker')[index % 2],
                                          'capacity-' + str(index)))
        live = [row for row in ray.get(controller.diagnostics.remote())['instances'] if row['state'] != 'STOPPED']
        assert len(live) == 1 and len(live[0]['training_ids']) == capacity
        try:
            overflow = TrainingHandle(base_url, model_name, 'tinker', 'overflow')
        except Exception as exc:
            assert 'replica limit' in str(exc).lower(), exc
        else:
            overflow.close()
            raise AssertionError('LoRA capacity was exceeded')
        handles[0].step()
    finally:
        for handle in reversed(handles):
            handle.close()
    assert_released(controller, ray.get_runtime_context().namespace)
    exclusive = TrainingHandle(base_url, full_model_name, 'twinkle', 'exclusive', mode='full')
    try:
        exclusive.step()
        live = [row for row in ray.get(controller.diagnostics.remote())['instances'] if row['state'] != 'STOPPED']
        assert len(live) == 1 and live[0]['mode'] == 'full' and len(live[0]['training_ids']) == 1
        try:
            overflow = TrainingHandle(base_url, full_model_name, 'tinker', 'exclusive-overflow', mode='full')
        except Exception as exc:
            assert 'replica limit' in str(exc).lower(), exc
        else:
            overflow.close()
            raise AssertionError('Full-training instance was shared')
    finally:
        exclusive.close()
    assert_released(controller, ray.get_runtime_context().namespace)
    return True


def validate_redis_restart(base_url, controller, model_name, managed):
    """Only interrupt a Redis process started and owned by this harness."""
    import requests
    from tests.server.integration.on_demand_training.acceptance_support import TrainingHandle
    handle = TrainingHandle(base_url, model_name, 'twinkle', 'redis-fault')
    try:
        handle.step()
        before = ray.get(controller.diagnostics.remote())

        def unavailable():
            response = requests.post(
                base_url + '/api/v1/model/' + model_name + '/twinkle/get_train_configs',
                json={'training_id': handle.identity, 'adapter_name': handle.label},
                headers={'Authorization': 'Bearer ' + handle.token, 'x-request-id': uuid.uuid4().hex,
                         'X-Twinkle-Session-Id': handle.transport.context.session_id}, timeout=15)
            assert response.status_code >= 500, response.text

        managed.fault(unavailable)
        after = ray.get(controller.diagnostics.remote(), timeout=30)
        assert after['epoch'] == before['epoch']
        assert {row['instance_id'] for row in before['instances']} == {
            row['instance_id'] for row in after['instances']}
        handle.step()
    finally:
        handle.close()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True, choices=['transformers', 'megatron'])
    parser.add_argument('--model', required=True)
    parser.add_argument('--training-devices', type=int, default=2)
    parser.add_argument('--training-pp-size', type=int, default=1)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--persistence', choices=['memory', 'redis'], default='memory')
    parser.add_argument('--managed-redis', action='store_true', help='Start an owned Redis for restart injection')
    parser.add_argument('--redis-url', default=os.environ.get('TWINKLE_TEST_REDIS_URL'))
    parser.add_argument('--startup-timeout-s', type=float, default=1800)
    parser.add_argument('--shutdown-timeout-s', type=float, default=180)
    parser.add_argument('--recovery-rounds', type=int, default=1)
    parser.add_argument('--skip-full-cycle', action='store_true')
    parser.add_argument('--recovery-only', action='store_true')
    parser.add_argument('--full-cycle-only', action='store_true')
    parser.add_argument('--stress-iterations', type=int, default=0)
    parser.add_argument('--soak-seconds', type=float, default=0)
    args = parser.parse_args()
    if args.persistence == 'redis' and not args.redis_url and not args.managed_redis:
        parser.error('--redis-url or --managed-redis is required for Redis persistence')
    if args.managed_redis and args.persistence != 'redis':
        parser.error('--managed-redis requires --persistence redis')
    if any(not math.isfinite(value) or value <= 0
           for value in (args.startup_timeout_s, args.shutdown_timeout_s)):
        parser.error('startup and shutdown timeouts must be finite and positive')
    if args.training_devices < 1:
        parser.error('--training-devices must be positive')
    if args.backend == 'megatron' and args.training_devices < 2:
        parser.error('megatron training instances require at least 2 devices')
    if args.training_pp_size < 1 or args.training_devices % args.training_pp_size:
        parser.error('--training-pp-size must be positive and divide --training-devices')
    if args.recovery_rounds < 0 or args.stress_iterations < 0 or args.soak_seconds < 0:
        parser.error('validation budgets must be non-negative')
    root = Path(__file__).resolve().parents[4]
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    managed = None
    if args.managed_redis:
        from tests.server.integration.on_demand_training.managed_redis import ManagedRedis
        managed = ManagedRedis(run_dir / 'redis')
        atexit.register(managed.stop)
        managed.start()
        args.redis_url = managed.url
    namespace = 'matrix_' + uuid.uuid4().hex[:8]
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    source = root / 'tests/server/config' / ('server_config_4b_e2e.yaml' if args.backend == 'transformers' else
                                             'server_config_4b_e2e_megatron.yaml')
    raw = yaml.safe_load(source.read_text())
    raw.setdefault('resources', {}).update({
        'startup_timeout_s': args.startup_timeout_s,
        'shutdown_timeout_s': args.shutdown_timeout_s,
    })
    redis_prefix = f'on-demand-test::{namespace}::'
    raw['persistence'] = ({
        'mode': 'redis',
        'redis_url': args.redis_url,
        'key_prefix': redis_prefix,
    } if args.persistence == 'redis' else {'mode': 'memory'})
    raw['ray_namespace'] = namespace
    raw['http_options'] = {'host': '127.0.0.1', 'port': port}
    full_applications = []
    for app in raw['applications']:
        params = app['args']
        if app['import_path'] == 'model':
            common = {
                'device_group': {'ranks': args.training_devices, 'device_type': 'cuda'},
                'device_mesh': {'device_type': 'cuda', 'dp_size': args.training_devices // args.training_pp_size,
                                                'pp_size': args.training_pp_size},
                'min_replicas': 0, 'max_replicas': 1, 'idle_timeout_s': 3600,
                'max_length': params.get('max_length'),
                'queue_config': {**params.get('queue_config', {}), 'execution_timeout': 600},
            }
            app['args'] = {
                **common, 'model_id': args.model, 'backend': args.backend, 'train_mode': 'lora',
                'max_loras': 10, 'max_r': 32,
                'env_vars': {'TWINKLE_TRUST_REMOTE_CODE': '1'},
            }
            app.pop('deployments', None)
            full_applications.append({
                'name': app['name'] + '-full',
                'route_prefix': app['route_prefix'] + '-full',
                'import_path': 'model',
                'args': {**common, 'model_id': args.model, 'backend': args.backend, 'train_mode': 'full',
                         'env_vars': app['args']['env_vars']},
            })
        elif app['import_path'] == 'sampler':
            params['model_id'] = args.model
            params['engine_args']['attention_config'] = {'flash_attn_version': 3}
    raw['applications'].extend(full_applications)
    if args.recovery_only:
        raw['applications'] = [app for app in raw['applications'] if app['import_path'] in ('server', 'model')]
    config = ServerConfig.model_validate(raw)
    evidence_config = config.to_yaml_dict()
    if args.persistence == 'redis':
        evidence_config['persistence']['redis_url'] = '<redacted>'
    (run_dir / 'server.yaml').write_text(yaml.safe_dump(evidence_config, sort_keys=False))
    os.environ.update({
        'TWINKLE_STATE_RAY_NAMESPACE': namespace,
        'TWINKLE_PERSISTENCE_MODE': args.persistence,
        'TWINKLE_DEFAULT_SAVE_DIR': str(run_dir / 'checkpoints'),
        'TWINKLE_MODEL_ID_ALIASES': json.dumps({model.name: model.model_id
                                              for model in config.training_config.models}),
        'TWINKLE_TRUST_REMOTE_CODE': '1',
        'TWINKLE_TEST_GPU_E2E': '1',
        'TWINKLE_TEST_BACKEND': args.backend,
        'TWINKLE_SERVER_URL': f'http://127.0.0.1:{port}',
        'TWINKLE_TEST_OPERATION_TIMEOUT': '600',
        'PYTHONPATH': os.pathsep.join((str(root / 'src'), str(root))),
    })
    if args.persistence == 'redis':
        os.environ.update({
            'TWINKLE_PERSISTENCE_REDIS_URL': args.redis_url,
            'TWINKLE_PERSISTENCE_KEY_PREFIX': redis_prefix,
        })
    else:
        os.environ.pop('TWINKLE_PERSISTENCE_REDIS_URL', None)
        os.environ.pop('TWINKLE_PERSISTENCE_KEY_PREFIX', None)
    evidence = {
        'backend': args.backend,
        'persistence': args.persistence,
        'namespace': namespace,
        'cases': [],
        'lazy_start': False,
    }
    started = time.monotonic()
    ray.init(
        address='local',
        namespace=namespace,
        num_cpus=24,
        num_gpus=args.training_devices + (0 if args.recovery_only else 1),
        _temp_dir='/tmp/twm-' + namespace[-8:],
        include_dashboard=False)
    evidence['ray_startup_seconds'] = time.monotonic() - started
    launcher = ServerLauncher(config)
    try:
        started = time.monotonic()
        launcher._start_serve()
        for app in config.applications:
            if app.import_path != 'sampler':
                launcher._deploy_application(app)
        launcher._deploy_training()
        assert ray.get(launcher._training_controller.diagnostics.remote())['instances'] == []
        evidence['lazy_start'] = True
        evidence['server_startup_seconds'] = time.monotonic() - started
        print(json.dumps({'phase': 'training_server_ready', 'seconds': evidence['server_startup_seconds'],
                          'sampler_running': False}), flush=True)
        filenames = ['test_sft_e2e.py', 'test_dpo_e2e.py', 'test_grpo_e2e.py']
        if not args.skip_full_cycle:
            filenames.append('test_full_cycle_e2e.py')
        if args.recovery_only:
            filenames = []
        elif args.full_cycle_only:
            filenames = ['test_full_cycle_e2e.py']
        for filename in filenames:
            collection = subprocess.run([
                sys.executable, '-m', 'pytest', '--noconftest', '--collect-only', '-q',
                str(root / 'tests/server/integration' / filename),
            ], capture_output=True, text=True, env=os.environ.copy(), timeout=180)
            (run_dir / (filename + '.collect.log')).write_text(collection.stdout + collection.stderr)
            expected = sum('::test_' in line for line in collection.stdout.splitlines())
            assert collection.returncode == 0 and expected > 0, collection.stdout + collection.stderr
            if filename in ('test_grpo_e2e.py', 'test_full_cycle_e2e.py') and not launcher._sampler_controllers:
                started = time.monotonic()
                print(json.dumps({'phase': 'sampler_starting', 'needed_by': filename}), flush=True)
                for app in config.applications:
                    if app.import_path == 'sampler':
                        launcher._deploy_application(app)
                evidence['sampler_startup_seconds'] = time.monotonic() - started
                evidence['sampler_started_for'] = filename
                print(json.dumps({'phase': 'sampler_ready', 'seconds': evidence['sampler_startup_seconds'],
                                  'needed_by': filename}), flush=True)
            start = time.monotonic()
            print(json.dumps({'phase': 'case_start', 'file': filename,
                              'sampler_running': bool(launcher._sampler_controllers)}), flush=True)
            with (run_dir / (filename + '.log')).open('w') as output:
                result = subprocess.run([
                    sys.executable,
                    '-m',
                    'pytest',
                    '--noconftest',  # HTTP clients do not need a second local Ray fixture.
                    str(root / 'tests/server/integration' / filename),
                    '-vv',
                    '-s',
                    '--junitxml=' + str(run_dir / (filename + '.xml')),
                ],
                                        stdout=output,
                                        stderr=subprocess.STDOUT,
                                        env=os.environ.copy(),
                                        timeout=7200)
            junit = run_dir / (filename + '.xml')
            cases = ElementTree.parse(junit).getroot().findall('.//testcase') if junit.exists() else []
            skipped = sum(case.find('skipped') is not None for case in cases)
            evidence['cases'].append({
                'file': filename,
                'returncode': result.returncode,
                'tests': len(cases),
                'skipped': skipped,
                'sampler_running': bool(launcher._sampler_controllers),
                'seconds': time.monotonic() - start
            })
            (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
            assert result.returncode == 0, f'{filename} failed; see {run_dir / (filename + ".log")}'
            assert len(cases) == expected and skipped == 0
            assert not any(case.find('failure') is not None or case.find('error') is not None for case in cases)
        base_url = f'http://127.0.0.1:{port}'
        model_name = config.training_config.models[0].name
        evidence['recovery'] = []
        for _ in range(args.recovery_rounds):
            recovery = validate_controller_restart(base_url, launcher._training_controller, model_name)
            for sampler in launcher._sampler_controllers:
                assert ray.get(sampler.diagnostics.remote(), timeout=30)['ready'], 'Training restart disturbed sampler'
            if launcher._sampler_controllers:
                recovery['sampler_remained_ready'] = True
            evidence['recovery'].append(recovery)
            evidence['restart_epoch'] = recovery['epoch']
            evidence['worker_fault_recovered'] = recovery['worker_fault_recovered']
            if managed is not None:
                evidence['redis_fault_rounds'] = evidence.get('redis_fault_rounds', 0) + int(
                    validate_redis_restart(base_url, launcher._training_controller, model_name, managed))
            (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
        if args.recovery_rounds:
            evidence['idempotency_checked'] = validate_idempotent_create(
                base_url, launcher._training_controller, model_name)
            evidence['capacity_checked'] = validate_capacity(
                base_url, launcher._training_controller, model_name,
                next(model.name for model in config.training_config.models if model.mode == 'full'),
                config.training_config.models[0].config.max_loras)
        if args.stress_iterations or args.soak_seconds:
            from tests.server.integration.on_demand_training.acceptance_support import assert_released
            controller = launcher._training_controller
            assert_released(controller, namespace,
                            args.redis_url if args.persistence == 'redis' else None, redis_prefix)
            from tests.server.integration.on_demand_training.stress_validation import run_stress
            evidence['stress'] = []
            budgets = ([('short-' + str(index), args.stress_iterations, 0) for index in range(3)]
                       if args.stress_iterations else [])
            if args.soak_seconds:
                budgets.append(('soak', 0, args.soak_seconds))
            for name, iterations, duration in budgets:
                evidence['stress'].append(run_stress(
                    f'http://127.0.0.1:{port}', config.training_config.models[0].name,
                    concurrency=2, iterations=iterations, duration=duration,
                    controller=launcher._training_controller, namespace=namespace,
                    redis_url=args.redis_url if args.persistence == 'redis' else None, prefix=redis_prefix,
                    progress_path=run_dir / (name + '.json')))
                (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
    finally:
        cleanup_started = time.monotonic()
        try:
            controller = getattr(launcher, '_training_controller', None)
            if controller is not None:
                ray.get(controller.shutdown.remote(), timeout=args.shutdown_timeout_s)
                diagnostic = ray.get(controller.diagnostics.remote(), timeout=30)
                assert all(row['state'] == 'STOPPED' for row in diagnostic['instances'])
                from tests.server.integration.on_demand_training.acceptance_support import assert_released
                assert_released(controller, namespace,
                                args.redis_url if args.persistence == 'redis' else None, redis_prefix)
                evidence['training_resources_released'] = True
            launcher._shutdown()
        finally:
            serve.shutdown()
            ray.shutdown()
            if args.persistence == 'redis' and evidence.get('training_resources_released'):
                import redis
                client = redis.Redis.from_url(args.redis_url)
                try:
                    keys = list(client.scan_iter(match=redis_prefix + '*', count=500))
                    if keys:
                        client.delete(*keys)
                finally:
                    client.close()
            if managed is not None:
                managed.stop()
            evidence['cleanup_seconds'] = time.monotonic() - cleanup_started
            (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
    evidence['passed'] = True
    (run_dir / 'results.json').write_text(json.dumps(evidence, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
