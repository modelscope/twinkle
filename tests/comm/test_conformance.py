# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend conformance: the same multi-process scenarios must pass on every communicator backend.

Each case spawns ``world_size`` processes that build groups through ``create_communicator`` and
run every scenario against them. A backend joins by adding a ``_BACKEND_DEVICE_TYPES`` entry; an
accelerator backend gets an ``accel(world_size)`` mark, so boxes without enough devices skip it,
and one device per rank (NCCL-style libraries refuse two ranks on one device).
"""
import json
import math
import multiprocessing as mp
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest
import torch

from twinkle.utils.network import find_free_port

_GROUP_TIMEOUT = timedelta(seconds=60)
_BARRIER_DELAY_S = 1.0
_PORT_RETRIES = 3

#: backend name -> device type its ranks bind to.
_BACKEND_DEVICE_TYPES = {
    'gloo': 'cpu',
}

_BROADCAST_DTYPES = (torch.float32, torch.bfloat16, torch.int64, torch.uint8)


def _device_for_rank(device_type: str, rank: int) -> torch.device:
    return torch.device('cpu') if device_type == 'cpu' else torch.device(device_type, rank)


def _foreign_device(device: torch.device) -> torch.device:
    return torch.device('meta') if device.type == 'cpu' else torch.device('cpu')


def _payload(seed: int, dtype: torch.dtype, device: torch.device, numel: int = 24) -> torch.Tensor:
    return ((torch.arange(numel) % 7) + seed).to(dtype=dtype, device=device)


def _assert_payload(t: torch.Tensor, seed: int, what) -> None:
    expected = _payload(seed, t.dtype, torch.device('cpu'), t.numel())
    assert torch.equal(t.cpu().reshape(-1), expected), what


# ---- scenarios (run inside every rank) -------------------------------------------


def _check_broadcast(comm, device):
    for src in range(comm.world_size):
        for dtype in _BROADCAST_DTYPES:
            t = _payload(comm.rank, dtype, device).reshape(2, 3, 4)
            comm.broadcast(t, src=src)
            comm.synchronize()
            _assert_payload(t, src, ('broadcast', src, dtype))
    comm.broadcast(torch.empty(0, dtype=torch.uint8, device=device), src=0)
    large = torch.full((1 << 20, ), comm.rank, dtype=torch.uint8, device=device)
    comm.broadcast(large, src=comm.world_size - 1)
    comm.synchronize()
    assert bool((large == comm.world_size - 1).all())


def _check_async_overlap(comm, device):
    src = comm.world_size - 1
    t = _payload(comm.rank, torch.float32, device, numel=1 << 16)
    work = comm.broadcast(t, src=src, async_op=True)
    assert work is not None
    sum(range(200_000))  # host work while the broadcast is in flight
    work.wait()
    comm.synchronize()
    _assert_payload(t, src, 'async broadcast')


def _check_all_reduce(comm, device):
    from twinkle.comm import ReduceOp

    if not comm.supports('all_reduce'):
        return
    n = comm.world_size
    expected = {
        ReduceOp.SUM: n * (n + 1) // 2,
        ReduceOp.PRODUCT: math.factorial(n),
        ReduceOp.MIN: 1,
        ReduceOp.MAX: n,
        'sum': n * (n + 1) // 2,
    }
    for op, value in expected.items():
        for dtype in (torch.float32, torch.int64):
            t = torch.full((3, ), comm.rank + 1, dtype=dtype, device=device)
            comm.all_reduce(t, op=op)
            comm.synchronize()
            assert t.cpu().tolist() == [value] * 3, (op, dtype, t)
    t = torch.full((3, ), float(comm.rank + 1), device=device)
    comm.all_reduce(t, op='avg')
    comm.synchronize()
    assert torch.allclose(t.cpu(), torch.full((3, ), (n + 1) / 2)), t


def _check_threaded_double_buffer(comm, device):
    """Production shape of the NCCL engine: broadcast on a worker thread while the next bucket fills."""
    rounds = 6
    buffers = [torch.empty(4096, dtype=torch.int64, device=device) for _ in range(2)]
    pending = None
    with ThreadPoolExecutor(max_workers=1) as pool:
        for r in range(rounds):
            buf = buffers[r % 2]
            buf.copy_(_payload(r if comm.rank == 0 else -1, torch.int64, device, buf.numel()))
            if pending is not None:
                future, prev_buf, prev_round = pending
                future.result()
                comm.synchronize()
                _assert_payload(prev_buf, prev_round, ('threaded round', prev_round))
            pending = (pool.submit(comm.broadcast, buf, 0), buf, r)
        future, prev_buf, prev_round = pending
        future.result()
        comm.synchronize()
        _assert_payload(prev_buf, prev_round, ('threaded round', prev_round))


def _check_barrier(comm, device, record):
    if comm.rank == 0:
        time.sleep(_BARRIER_DELAY_S)
        record['barrier_enter'] = time.time()
    comm.barrier()
    record['barrier_exit'] = time.time()


def _check_rejections(comm, device):
    for bad in (torch.zeros(4, 4, device=device).t(), torch.zeros(2, device=_foreign_device(device))):
        try:
            comm.broadcast(bad, src=0)
        except ValueError:
            continue
        raise AssertionError(f'broadcast accepted an invalid tensor: {bad.device}, contiguous={bad.is_contiguous()}')


def _check_destroyed(comm, device):
    comm.destroy()
    comm.destroy()
    for call in (lambda: comm.broadcast(torch.zeros(1, device=device)),
                 lambda: comm.all_reduce(torch.zeros(1, device=device)), comm.barrier):
        try:
            call()
        except RuntimeError:
            continue
        raise AssertionError('operation after destroy must raise RuntimeError')


def _run_worker(backend, device_type, rank, world_size, ports, out_dir):
    out = Path(out_dir)
    record = {}
    try:
        from twinkle.comm import Rendezvous, create_communicator

        device = _device_for_rank(device_type, rank)

        def build(port, name):
            return create_communicator(Rendezvous('127.0.0.1', port, group_name=name), rank, world_size, device,
                                       backend=backend, timeout=_GROUP_TIMEOUT)

        comm = build(ports[0], 'main')
        assert (comm.rank, comm.world_size, comm.backend_name) == (rank, world_size, backend)
        for check in (_check_broadcast, _check_async_overlap, _check_all_reduce, _check_threaded_double_buffer,
                      _check_rejections):
            check(comm, device)
        _check_barrier(comm, device, record)

        # A second live group (full sync next to delta sync) must not interfere with the first.
        other = build(ports[1], 'other')
        for round_ in range(3):
            a = _payload(rank, torch.float32, device)
            b = _payload(rank + 100, torch.float32, device)
            comm.broadcast(a, src=0)
            other.broadcast(b, src=world_size - 1)
            comm.synchronize()
            _assert_payload(a, 0, ('main group', round_))
            _assert_payload(b, world_size - 1 + 100, ('other group', round_))
        other.destroy()
        _check_destroyed(comm, device)

        # rebuild_group=True: a destroyed group must not block building a fresh one.
        comm = build(ports[2], 'rebuilt')
        t = _payload(rank, torch.float32, device)
        comm.broadcast(t, src=0)
        comm.synchronize()
        _assert_payload(t, 0, 'rebuilt group')
        comm.destroy()
        (out / f'rank{rank}.json').write_text(json.dumps(record))
    except BaseException:
        (out / f'rank{rank}.err').write_text(traceback.format_exc())
        raise


def _spawn(backend, device_type, world_size, out_dir):
    ctx = mp.get_context('spawn')
    ports = tuple(find_free_port() for _ in range(3))
    procs = [
        ctx.Process(target=_run_worker, args=(backend, device_type, rank, world_size, ports, str(out_dir)),
                    daemon=True) for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    deadline = time.monotonic() + 60 + 30 * world_size
    for p in procs:
        p.join(max(0.0, deadline - time.monotonic()))
    hung = [rank for rank, p in enumerate(procs) if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()
    errors = {int(f.stem[4:]): f.read_text() for f in Path(out_dir).glob('rank*.err')}
    return procs, hung, errors


def _cases():
    for backend, device_type in sorted(_BACKEND_DEVICE_TYPES.items()):
        for world_size in (2, 4):
            marks = [] if device_type == 'cpu' else [pytest.mark.accel(world_size)]
            yield pytest.param(backend, device_type, world_size, id=f'{backend}-ws{world_size}', marks=marks)


@pytest.mark.parametrize('backend, device_type, world_size', list(_cases()))
def test_conformance(backend, device_type, world_size, tmp_path):
    from twinkle.comm import available_backends

    if backend not in available_backends():
        pytest.skip(f'{backend} communicator is not implemented yet')
    for attempt in range(_PORT_RETRIES):
        out_dir = tmp_path / f'attempt{attempt}'
        out_dir.mkdir()
        procs, hung, errors = _spawn(backend, device_type, world_size, out_dir)
        port_clash = any('Address already in use' in tb or 'EADDRINUSE' in tb for tb in errors.values())
        if not (port_clash and attempt + 1 < _PORT_RETRIES):
            break

    report = '\n'.join(f'--- rank {rank} ---\n{tb}' for rank, tb in sorted(errors.items()))
    assert not hung, f'ranks {hung} hung\n{report}'
    assert not errors, report
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    records = [json.loads((out_dir / f'rank{rank}.json').read_text()) for rank in range(world_size)]
    entered = records[0]['barrier_enter']
    late = {rank: entered - r['barrier_exit'] for rank, r in enumerate(records) if r['barrier_exit'] < entered}
    assert not late, f'ranks left the barrier before rank 0 entered it (seconds early): {late}'
