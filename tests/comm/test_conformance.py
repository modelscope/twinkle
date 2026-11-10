# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend conformance: the same multi-process scenarios must pass on every communicator backend.

Each case spawns ``world_size`` processes that build one group through ``create_communicator``
and run every scenario against it. A new backend joins by adding a ``_BACKENDS`` entry that
names the device each rank binds to (and an ``accel`` mark when it needs hardware).
"""
import math
import multiprocessing as mp
import queue
import traceback
from datetime import timedelta

import pytest
import torch

from twinkle.utils.network import find_free_port

_JOIN_TIMEOUT_S = 180
_GROUP_TIMEOUT = timedelta(seconds=60)


def _cpu_device(rank: int) -> torch.device:
    return torch.device('cpu')


_BACKENDS = {
    'gloo': _cpu_device,
}

_BROADCAST_DTYPES = (torch.float32, torch.bfloat16, torch.int64, torch.uint8)


def _payload(seed: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return ((torch.arange(24) % 7) + seed).reshape(2, 3, 4).to(dtype=dtype, device=device)


def _check_broadcast(comm, device):
    for src in range(comm.world_size):
        for dtype in _BROADCAST_DTYPES:
            t = _payload(comm.rank, dtype, device)
            comm.broadcast(t, src=src)
            comm.synchronize()
            assert torch.equal(t.cpu(), _payload(src, dtype, device).cpu()), (src, dtype)

    empty = torch.empty(0, dtype=torch.uint8, device=device)
    comm.broadcast(empty, src=0)

    large = torch.full((1 << 20, ), comm.rank, dtype=torch.uint8, device=device)
    comm.broadcast(large, src=comm.world_size - 1)
    comm.synchronize()
    assert bool((large == comm.world_size - 1).all())


def _check_async(comm, device):
    src = comm.world_size - 1
    t = _payload(comm.rank, torch.float32, device)
    work = comm.broadcast(t, src=src, async_op=True)
    assert work is not None
    work.wait()
    comm.synchronize()
    assert torch.equal(t.cpu(), _payload(src, torch.float32, device).cpu())


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


def _check_barrier(comm, device):
    for _ in range(3):
        comm.barrier()


def _run_worker(backend, rank, world_size, ports, errors):
    try:
        from twinkle.comm import Rendezvous, create_communicator

        device = _BACKENDS[backend](rank)
        comm = create_communicator(Rendezvous('127.0.0.1', ports[0]), rank, world_size, device, backend=backend,
                                   timeout=_GROUP_TIMEOUT)
        assert (comm.rank, comm.world_size, comm.backend_name) == (rank, world_size, backend)
        for check in (_check_broadcast, _check_async, _check_all_reduce, _check_barrier):
            check(comm, device)

        comm.destroy()
        comm.destroy()
        try:
            comm.broadcast(torch.zeros(1, device=device))
        except RuntimeError:
            pass
        else:
            raise AssertionError('broadcast after destroy must raise')

        # A destroyed group must not block building a fresh one (rebuild_group=True path).
        comm = create_communicator(Rendezvous('127.0.0.1', ports[1]), rank, world_size, device, backend=backend,
                                   timeout=_GROUP_TIMEOUT)
        t = _payload(rank, torch.float32, device)
        comm.broadcast(t, src=0)
        comm.synchronize()
        assert torch.equal(t.cpu(), _payload(0, torch.float32, device).cpu())
        comm.destroy()
    except BaseException:
        errors.put((rank, traceback.format_exc()))
        raise


@pytest.mark.parametrize('world_size', [2, 4])
@pytest.mark.parametrize('backend', sorted(_BACKENDS))
def test_conformance(backend, world_size):
    ctx = mp.get_context('spawn')
    errors = ctx.Queue()
    ports = (find_free_port(), find_free_port())
    procs = [
        ctx.Process(target=_run_worker, args=(backend, rank, world_size, ports, errors), daemon=True)
        for rank in range(world_size)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(_JOIN_TIMEOUT_S)
    hung = [rank for rank, p in enumerate(procs) if p.is_alive()]
    for p in procs:
        if p.is_alive():
            p.kill()
            p.join()

    failures = []
    while True:
        try:
            failures.append(errors.get(timeout=1))
        except queue.Empty:
            break
    report = '\n'.join(f'--- rank {rank} ---\n{tb}' for rank, tb in sorted(failures))
    assert not hung, f'ranks {hung} hung after {_JOIN_TIMEOUT_S}s\n{report}'
    assert not failures, report
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]
