# Copyright (c) ModelScope Contributors. All rights reserved.
"""Real vLLM pool acceptance. Run only on two or four dedicated devices with the PPU SDK loaded."""
import asyncio
import json
import math
import os
import shutil
import subprocess
import time
import traceback
import uuid
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TWINKLE_TEST_REAL_POOL') != '1',
                                reason='Requires two or four dedicated devices and a real cached model')


@pytest.fixture(scope='session', autouse=True)
def _ray_runtime():
    yield


@pytest.fixture(autouse=True)
def _reset_canonical_state_actor():
    yield


def payload(endpoint, body):
    return {'endpoint': endpoint, 'method': 'POST', 'body': json.dumps(body).encode(),
            'headers': [(b'content-type', b'application/json'), (b'x-request-id', uuid.uuid4().hex.encode()),
                        (b'authorization', b'Bearer pool-acceptance')], 'query_string': b''}


def checkpoint_alias(model, target):
    target.mkdir()
    for source in model.iterdir():
        if source.is_file():
            (target / source.name).symlink_to(source)
    return str(target)


def make_lora(model, target):
    import torch
    from peft import LoraConfig
    from safetensors import safe_open
    from safetensors.torch import save_file
    rank, weights = 8, {}
    generator = torch.Generator().manual_seed(312)
    for shard in model.glob('*.safetensors'):
        with safe_open(str(shard), framework='pt', device='cpu') as source:
            for name in source.keys():
                if name.endswith('.self_attn.q_proj.weight'):
                    output_size, input_size = source.get_slice(name).get_shape()
                    prefix = 'base_model.model.' + name[:-len('.weight')]
                    weights[prefix + '.lora_A.weight'] = torch.randn(rank, input_size, generator=generator) * .05
                    weights[prefix + '.lora_B.weight'] = torch.randn(output_size, rank, generator=generator) * .05
    assert weights, 'No q_proj weights found; this is not a skipped acceptance case'
    target.mkdir()
    LoraConfig(r=rank, lora_alpha=rank, target_modules=['q_proj'], bias='none', task_type='CAUSAL_LM',
               inference_mode=True, base_model_name_or_path=str(model)).save_pretrained(target)
    save_file(weights, str(target / 'adapter_model.safetensors'))
    return str(target)


async def exercise(model, directory, namespace, evidence, *, extended=True):
    import numpy as np
    import ray
    from ray import serve
    from ray.util import list_named_actors
    from ray.util.placement_group import placement_group_table
    from transformers import AutoTokenizer
    from twinkle.server.config.application_spec import SamplerArgs
    from twinkle.server.sampler.pool.controller import SamplerController
    from twinkle.server.sampler.pool.driver import SamplerDriver
    from twinkle.server.state import get_server_state

    devices_count = int(os.environ.get('TWINKLE_TEST_POOL_DEVICES', '4'))
    assert devices_count in (2, 4)
    minimum = devices_count // 2
    args = SamplerArgs(model_id=str(model), sampler_type='vllm',
                       device_group={'name': 'sampler', 'ranks': 1, 'device_type': 'cuda'},
                       device_mesh={'device_type': 'cuda', 'dp_size': 1},
                       engine_args={'max_model_len': 2048, 'gpu_memory_utilization': .4, 'enable_lora': True,
                                    'enforce_eager': True, 'logprobs_mode': 'processed_logprobs',
                                    'attention_config': {'flash_attn_version': 3}},
                       min_replicas=minimum, max_replicas=devices_count, target_ongoing_requests=1, idle_timeout_s=2,
                       queue_config={'execution_timeout': 600, 'queue_timeout': 600, 'inline_fast_path_timeout': 0.01})
    budgets = {'allocation_timeout_s': 60, 'startup_timeout_s': 900,
               'shutdown_timeout_s': 180, 'sweep_interval_s': 1}
    driver = SamplerDriver(args, namespace, 'real-pool', {}, budgets)
    clock_offset = [0]
    pool = SamplerController(args, driver, budgets, clock=lambda: time.monotonic() + clock_offset[0])
    pools = [pool]
    state = get_server_state()
    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    ids = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Count the integers from one to one hundred thousand.'}],
                                       tokenize=True, add_generation_prompt=True, enable_thinking=False,
                                       return_dict=False)
    calls = []
    futures = []
    invoke = driver.invoke

    async def observed(instance, ticket, body, queue=None):
        calls.append({'instance': instance.identity, 'endpoint': body['endpoint'], 'devices': instance.devices})
        return await invoke(instance, ticket, body, queue)

    driver.invoke = observed

    async def sample(target=pool, path=None, full=None, max_tokens=64):
        body = {'inputs': [{'input_ids': ids}],
                'sampling_params': {'max_tokens': max_tokens, 'temperature': 0, 'logprobs': 1, 'ignore_eos': True}}
        if path:
            body['adapter_uri'] = path
        reply = await target.request(payload('twinkle/sample', body), full_path=full)
        assert reply['status'] == 200, reply
        result = json.loads(reply['body'])
        request_id = result['request_id']
        futures.append(request_id)
        deadline = time.monotonic() + 600
        while result['status'] not in ('completed', 'failed', 'cancelled'):
            assert time.monotonic() < deadline, result
            await asyncio.sleep(.05)
            result = await state.get_future(request_id)
        assert result['status'] == 'completed', result
        seq = result['result']['samples'][0]['sequences'][0]
        assert seq['tokens'] and len(seq['tokens']) == len(seq['logprobs'])
        assert all(math.isfinite(position[0][1]) for position in seq['logprobs'])
        return seq

    async def settled(target=pool):
        await asyncio.gather(*tuple(target.lifecycle.startups.values()))
        await target.lifecycle.join()

    def save():
        snapshots = evidence.setdefault('device_snapshots', [])
        name = f'devices-stage-{len(snapshots):02d}.txt'
        (directory / name).write_text(subprocess.check_output(['ppu-smi'], text=True))
        snapshots.append(name)
        (directory / 'results.json').write_text(json.dumps(evidence, indent=2))

    try:
        await pool.initialize()
        evidence['sizes'] = [len(pool._ready_instances())]
        source_root = str(Path(__file__).resolve().parents[4])

        async def verify_worker_origins():
            workers = [row for row in list_named_actors(all_namespaces=True)
                       if row['namespace'] == namespace and row['name'].startswith('sampler-')
                       and not row['name'].endswith('-registry')]
            assert len(workers) == len(pool._ready_instances())
            origins = {}
            for row in workers:
                worker = ray.get_actor(row['name'], namespace=namespace)
                origin = await worker.__ray_call__.remote(lambda _: __import__('twinkle').__file__)
                assert Path(origin).resolve().is_relative_to(source_root), origin
                origins[row['name']] = origin
            evidence['worker_import_origins'] = origins

        await verify_worker_origins()
        save()
        await asyncio.gather(*(sample(max_tokens=256) for _ in range(2 * devices_count)))
        await settled()
        assert len(pool._ready_instances()) == devices_count
        devices = [device for instance in pool._ready_instances() for device in instance.devices]
        assert len(devices) == len(set(devices)) == devices_count
        calls.clear()
        samples = await asyncio.gather(*(sample(max_tokens=128) for _ in range(2 * devices_count)))
        await settled()
        await verify_worker_origins()
        evidence['sample_result'] = samples[0]
        assert {row['instance'] for row in calls} == {i.identity for i in pool._ready_instances()}
        for instance in pool._ready_instances():
            assert not (await driver.health(instance))['unhealthy']
        evidence.update(device_ids=devices, request_placements=list(calls))
        evidence['sizes'].append(devices_count)
        save()

        # Observe real running compute before moving the idle clock past its deadline.
        futures.clear()
        active = [asyncio.create_task(sample(max_tokens=1024)) for _ in range(devices_count)]
        try:
            deadline = time.monotonic() + 30
            while True:
                records = [await state.get_future(identity) for identity in futures]
                if (len(records) == devices_count and all(row['status'] == 'running' for row in records)
                        and all(instance.requests for instance in pool._ready_instances())):
                    break
                assert time.monotonic() < deadline and not any(task.done() for task in active), records
                await asyncio.sleep(.01)
            clock_offset[0] += 10
            await pool.sweep()
            assert len(pool._ready_instances()) == devices_count
            evidence['active_compute_sweep_safe'] = True
        finally:
            await asyncio.gather(*active)
            await settled()

        # Pause the ACK after a real backend update, then sweep across the idle window.
        entered, resume = asyncio.Event(), asyncio.Event()
        original_wait = driver.wait_operation

        async def delayed_ack(instance, ticket):
            await original_wait(instance, ticket)
            entered.set()
            await resume.wait()

        driver.wait_operation = delayed_ack
        update = asyncio.create_task(pool.request(payload('twinkle/set_template', {'template_cls': 'Qwen3_5Template'})))
        try:
            await asyncio.wait_for(entered.wait(), 120)
            await asyncio.sleep(2.1)
            await pool.sweep()
            assert len(pool._ready_instances()) == devices_count
        finally:
            resume.set()
            await update
            driver.wait_operation = original_wait
        evidence['control_sweep_safe'] = True

        if not extended:
            await asyncio.sleep(2.1)
            await pool.sweep()
            await settled()
            evidence['sizes'].append(len(pool._ready_instances()))
            assert evidence['sizes'] == [minimum, devices_count, minimum]
            evidence['scaling_complete'] = True
            return

        old = checkpoint_alias(model, directory / 'checkpoint-A')
        new = checkpoint_alias(model, directory / 'checkpoint-B')
        await sample(full=old)
        await settled()
        original_load = driver.load_weights
        entered.clear(); resume.clear()

        async def delayed_load(instance, path):
            await original_load(instance, path)
            entered.set()
            await resume.wait()

        driver.load_weights = delayed_load
        update = asyncio.create_task(sample(full=new))
        try:
            await asyncio.wait_for(entered.wait(), 180)
            await asyncio.sleep(2.1)
            await pool.sweep()
            assert len(pool._ready_instances()) == devices_count
        finally:
            resume.set()
            await update
            driver.load_weights = original_load
        await settled()
        shutil.rmtree(old)  # Only the test-owned symlink directory; never the model cache.
        evidence['full_update_sweep_safe'] = True
        await asyncio.sleep(2.1)
        await pool.sweep()
        await settled()
        evidence['sizes'].append(len(pool._ready_instances()))
        assert evidence['sizes'] == [minimum, devices_count, minimum]
        save()
        await asyncio.gather(*(sample(max_tokens=128) for _ in range(2 * devices_count)))
        await settled()
        assert len(pool._ready_instances()) == devices_count
        evidence['superseded_checkpoint_rebuild'] = True

        lora = make_lora(model, directory / 'lora')
        before = await sample(path=lora)
        await settled()
        response = await pool.request(payload('twinkle/unload_adapter_paths', {'adapter_paths': [lora]}))
        assert response['status'] == 200
        after = await sample(path=lora)
        await settled()
        assert before['tokens'] == after['tokens']
        np.testing.assert_allclose([x[0][1] for x in before['logprobs']],
                                   [x[0][1] for x in after['logprobs']], rtol=0, atol=1e-5)
        evidence['lora_results'] = {'before_eviction': before, 'after_reload': after}
        evidence['lora_eviction_reload'] = True
        save()

        # Lose a real control ACK; all four real participants must be stopped.
        entered.clear()
        async def missing_ack(instance, ticket):
            await original_wait(instance, ticket)
            entered.set()
            await asyncio.Event().wait()
        driver.wait_operation = missing_ack
        args.queue_config.execution_timeout = 5
        try:
            with pytest.raises(TimeoutError):
                await pool.request(payload('twinkle/set_template', {'template_cls': 'Qwen3_5Template'}))
            assert entered.is_set()
            assert not pool._ready_instances() and pool._owned_count() == 0
        finally:
            driver.wait_operation = original_wait
            args.queue_config.execution_timeout = 600
        await pool.sweep()
        await settled()
        await sample(full=new)
        await settled()
        evidence['control_timeout_recovered'] = True
        save()

        victim = pool._ready_instances()[0]
        workers = [row for row in list_named_actors(all_namespaces=True)
                   if row['namespace'] == namespace and victim.identity in row['name']
                   and row['name'].startswith('sampler-') and not row['name'].endswith('-registry')]
        assert workers, 'No owned sampler worker for fault injection'
        ray.kill(ray.get_actor(workers[0]['name'], namespace=namespace), no_restart=True)
        deadline = time.monotonic() + 60
        while not (await driver.health(victim))['unhealthy']:
            assert time.monotonic() < deadline
            await asyncio.sleep(.2)
        await pool._close(victim)
        await pool.sweep()
        await settled()
        await sample(full=new)
        await settled()
        evidence['worker_fault_recovered'] = True
        save()

        # The new controller's production recovery finds and retires the old pool's resources.
        old_names = {i.application for i in pool._ready_instances()}
        replacement_driver = SamplerDriver(args, namespace, 'real-pool', {}, budgets)
        replacement = SamplerController(args, replacement_driver, budgets)
        pools.append(replacement)
        await replacement.initialize()
        assert replacement.epoch != pool.epoch
        assert not old_names.intersection(serve.status().applications)
        await sample(target=replacement)
        await settled(replacement)
        evidence['controller_orphan_recovery'] = True
        save()
    except BaseException:
        evidence['failure'] = traceback.format_exc()
        save()
        raise
    finally:
        for target in reversed(pools):
            await target.shutdown()
            await target.driver.runtime.wait_for_cleanup()
        assert not any(name.startswith(driver.prefix) for name in serve.status().applications)
        assert not any(row.get('name', '').startswith(driver.prefix) and row['state'] != 'REMOVED'
                       for row in placement_group_table().values())
        assert not any(row['namespace'] == namespace and 'sampler_' in row['name']
                       for row in list_named_actors(all_namespaces=True))
        evidence['confirmed_release'] = True
        if not extended and evidence.get('scaling_complete'):
            evidence['passed'] = True
        save()
    evidence['passed'] = True
    save()


def _run_real_pool(*, extended):
    import ray
    import twinkle
    from ray import serve
    model = Path(os.environ['TWINKLE_TEST_MODEL_PATH'])
    directory = Path(os.environ['TWINKLE_TEST_ARTIFACT_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    namespace = 'pool_acceptance_' + uuid.uuid4().hex[:8]
    os.environ.update(TWINKLE_STATE_RAY_NAMESPACE=namespace, TWINKLE_PERSISTENCE_MODE='memory',
                      TWINKLE_GPU_PG_CPUS_PER_PROC='1', TWINKLE_TRUST_REMOTE_CODE='1',
                      TWINKLE_DEFAULT_SAVE_DIR=str(directory / 'checkpoints'))
    evidence = {'namespace': namespace, 'model': str(model), 'import': twinkle.__file__,
                'sha': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()}
    ray.init(address='local', namespace=namespace, num_cpus=24,
             num_gpus=int(os.environ.get('TWINKLE_TEST_POOL_DEVICES', '4')),
             _temp_dir='/tmp/twp-' + namespace[-8:], include_dashboard=False, dashboard_port=0)
    try:
        serve.start(http_options={'host': '127.0.0.1', 'port': 0})
        asyncio.run(exercise(model, directory, namespace, evidence, extended=extended))
    finally:
        serve.shutdown()
        ray.shutdown()


def test_real_pool_scaling():
    _run_real_pool(extended=False)


def test_real_pool_lifecycle():
    _run_real_pool(extended=True)
