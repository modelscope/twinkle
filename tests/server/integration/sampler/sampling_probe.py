# Copyright (c) ModelScope Contributors. All rights reserved.
"""Exercise the actual sampler driver and mock compute service on an owned CPU cluster."""
import asyncio
import json
import os
import uuid

import ray
from ray import serve
from ray.util import list_named_actors
from ray.util.placement_group import placement_group_table

from twinkle.server.config.application_spec import SamplerArgs
from twinkle.server.sampler.pool.controller import SamplerController
from twinkle.server.sampler.pool.driver import SamplerDriver


async def run(namespace):
    args = SamplerArgs(
        model_id='mock', sampler_type='mock',
        device_group={'name': 'sampler', 'ranks': 1, 'device_type': 'cpu'},
        device_mesh={'device_type': 'cpu', 'dp_size': 1},
        min_replicas=2, max_replicas=4, target_ongoing_requests=1, idle_timeout_s=1)
    budgets = {'allocation_timeout_s': 10, 'startup_timeout_s': 600,
               'shutdown_timeout_s': 60, 'sweep_interval_s': 1}
    driver = SamplerDriver(args, namespace, 'cpu-allocation', {}, budgets)
    clock = [0]
    pool = SamplerController(args, driver, budgets, clock=lambda: clock[0])
    # Hold replies after the actual compute route runs. The pool must retain
    # their tickets until the caller can confirm backend completion.
    invoke = driver.invoke
    replies = 0
    dispatched, release = asyncio.Event(), asyncio.Event()

    async def delayed_reply(*args, **kwargs):
        nonlocal replies
        response = await invoke(*args, **kwargs)
        replies += 1
        if replies == 8:
            dispatched.set()
        await release.wait()
        return response

    driver.invoke = delayed_reply
    calls = []
    try:
        await pool.initialize()
        sizes = [len(pool._ready_instances())]
        payload = {'endpoint': 'twinkle/sample', 'method': 'POST',
                   'headers': [(b'content-type', b'application/json'),
                               (b'authorization', b'Bearer cpu-probe'),
                               (b'x-request-id', b'cpu-probe')], 'query_string': b'',
                   'body': json.dumps({'inputs': [{'input_ids': [1, 2]}],
                                       'sampling_params': {'max_tokens': 2}}).encode()}
        calls = [asyncio.create_task(pool.request(payload)) for _ in range(8)]
        await asyncio.wait_for(dispatched.wait(), 600)
        await asyncio.gather(*tuple(pool.lifecycle.startups.values()))
        sizes.append(len(pool._ready_instances()))
        clock[0] = 10
        await pool.sweep()
        assert pool._owned_count() == 4
        release.set()
        responses = await asyncio.gather(*calls)
        assert all(response['status'] == 200 for response in responses), responses
        await pool.lifecycle.join()
        from twinkle.server.state import get_server_state
        state = get_server_state()
        for response in responses:
            record = await state.get_future(json.loads(response['body'])['request_id'])
            assert record['status'] == 'completed', record
            assert len(record['result']['samples'][0]['sequences'][0]['tokens']) == 2
        health = await asyncio.gather(*(driver.health(instance) for instance in pool._ready_instances()))
        assert all(not result['unhealthy'] for result in health)
        clock[0] = 12
        await pool.sweep()
        await pool.lifecycle.join()
        sizes.append(len(pool._ready_instances()))
        assert sizes == [2, 4, 2], sizes
        await pool.shutdown()
        await driver.runtime.wait_for_cleanup()
        assert not any(name.startswith(driver.prefix) for name in serve.status().applications)
        assert not any(row.get('name', '').startswith(driver.prefix) and row['state'] != 'REMOVED'
                       for row in placement_group_table().values())
        assert not any(row['namespace'] == namespace and 'sampler_' in row['name']
                       for row in list_named_actors(all_namespaces=True))
        return {'sizes': sizes, 'requests': len(responses), 'confirmed_release': True}
    finally:
        release.set()
        await asyncio.gather(*calls, return_exceptions=True)
        await pool.shutdown()
        await driver.runtime.wait_for_cleanup()


def main():
    namespace = 'sampling_probe_' + uuid.uuid4().hex[:8]
    os.environ.update(TWINKLE_STATE_RAY_NAMESPACE=namespace, TWINKLE_PERSISTENCE_MODE='memory')
    ray.init(address='local', namespace=namespace, num_cpus=24, num_gpus=0,
             _temp_dir='/tmp/tws-' + namespace[-8:], include_dashboard=False)
    try:
        serve.start(http_options={'host': '127.0.0.1', 'port': 0})
        print(json.dumps(asyncio.run(run(namespace))), flush=True)
    finally:
        serve.shutdown()
        ray.shutdown()


if __name__ == '__main__':
    main()
