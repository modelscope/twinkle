# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-process harness for the HCCL checkpoint engine, on CPU (gloo) or on real NPUs (HCCL).

The engine under test is the real :class:`HCCLCheckpointEngine`: its bucketing and chunking, the
authenticated ZMQ REQ/REP bucket handshake, the communicator lifecycle (barrier, reuse, destroy).
With ``cfg['device'] == 'npu'`` each child pins NPU ``rank`` and runs the engine unmodified.
On CPU only the two NPU-specific pieces are substituted, through the seams the engine already
has: the bucket buffers live on CPU (``_buffer_device``) and ``create_communicator`` is asked
for the gloo backend, which honours the same :class:`~twinkle.comm.Communicator` contract.

Everything the children need lives here (spawned processes must import it).
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue as queue_mod
import time
import traceback

import torch

BARRIER_TIMEOUT_S = 60


def make_state(seed: int) -> dict[str, torch.Tensor]:
    """Mixed dtypes; with a small bucket the 4000-byte ``w2`` spans several buckets and chunks."""
    g = torch.Generator().manual_seed(seed)
    return {
        'layer.w1': torch.randn(16, 32, generator=g).to(torch.bfloat16),
        'layer.w2': torch.randn(1000, generator=g),
        'layer.idx': torch.randint(-1000, 1000, (64, ), generator=g),
        'layer.empty': torch.zeros(0),
        'layer.w3': torch.randn(8, 8, generator=g),
    }


def _engine_cls(cfg, rank):
    """The engine class for this child: unmodified on NPU, CPU buffers + gloo otherwise."""
    from twinkle.checkpoint_engine import hccl_checkpoint_engine as mod

    import twinkle.infra
    # Production engines run inside Ray actors; the node IP is the only Ray dependency.
    twinkle.infra.node_ip = lambda: '127.0.0.1'

    if cfg.get('device', 'cpu') == 'npu':
        import torch_npu  # noqa: F401
        torch.npu.set_device(rank)  # the engine binds to the current device at construction
        return mod.HCCLCheckpointEngine

    real_create = mod.create_communicator

    def create_cpu_communicator(rendezvous, rank, world_size, device, backend=None, **options):
        assert backend == 'hccl' and str(device).startswith('cpu'), (backend, device)
        options.pop('socket_env', None)  # HCCL-only; gloo has no port layout to agree on
        return real_create(rendezvous, rank, world_size, torch.device('cpu'), backend='gloo', **options)

    mod.create_communicator = create_cpu_communicator

    class CpuHCCLCheckpointEngine(mod.HCCLCheckpointEngine):
        _buffer_device = 'cpu'

    return CpuHCCLCheckpointEngine


def _engine_state(engine) -> dict:
    comm = engine._comm
    return dict(group=None if comm is None else id(comm), comm_destroyed=None if comm is None else comm.destroyed,
                socket=engine.socket is not None, prepared=engine._prepared)


def trainer_main(cfg, md_qs, res_q, barrier):
    try:
        engine_cls = _engine_cls(cfg, 0)
        world_size = 1 + cfg['n_recv']
        eng = engine_cls(bucket_size=cfg['bucket_size'], rebuild_group=cfg['rebuild_group'])
        eng.is_master = True
        for r, state in enumerate(cfg['rounds']):
            md = eng.prepare()
            if r == 0 or cfg['rebuild_group']:
                for q in md_qs:
                    q.put(md)
            eng.init_process_group(0, world_size, md)
            res_q.put(('trainer_group', r, dict(_engine_state(eng), hccl_env=md.hccl_env, auth_key=md.auth_key)))
            asyncio.run(eng.send_weights(iter(state.items())))
            res_q.put(('trainer_round', r, _engine_state(eng)))
            barrier.wait(BARRIER_TIMEOUT_S)
            comm = eng._comm
            eng.finalize()
            res_q.put(('trainer_finalized', r, dict(_engine_state(eng), held_destroyed=comm.destroyed)))
        comm = eng._comm
        eng.close()
        res_q.put(('trainer_closed', dict(_engine_state(eng),
                                          held_destroyed=None if comm is None else comm.destroyed)))
    except BaseException:
        res_q.put(('error', 'trainer', traceback.format_exc()))


def receiver_main(cfg, rank, md_q, res_q, barrier):
    try:
        engine_cls = _engine_cls(cfg, rank)
        world_size = 1 + cfg['n_recv']
        eng = engine_cls(bucket_size=cfg['bucket_size'], rebuild_group=cfg['rebuild_group'])
        md = None
        for r in range(len(cfg['rounds'])):
            if r == 0 or cfg['rebuild_group']:
                md = md_q.get(timeout=60)
            eng.prepare()
            eng.init_process_group(rank, world_size, md)

            async def consume():
                got = {}
                async for name, tensor in eng.receive_weights():
                    got[name] = tensor.cpu().clone()
                return got

            got = asyncio.run(consume())
            res_q.put(('recv_round', rank, r, got, _engine_state(eng)))
            barrier.wait(BARRIER_TIMEOUT_S)
            eng.finalize()
        eng.close()
        res_q.put(('recv_closed', rank, _engine_state(eng)))
    except BaseException:
        res_q.put(('error', f'receiver {rank}', traceback.format_exc()))


def run(cfg: dict, timeout: float = 180.0) -> dict:
    """Run trainer + receivers; returns every message keyed by kind. Children are always killed."""
    ctx = mp.get_context('spawn')
    n_recv = cfg['n_recv']
    res_q = ctx.Queue()
    md_qs = [ctx.Queue() for _ in range(n_recv)]
    barrier = ctx.Barrier(1 + n_recv)
    procs = [ctx.Process(target=trainer_main, args=(cfg, md_qs, res_q, barrier), daemon=True)]
    procs += [
        ctx.Process(target=receiver_main, args=(cfg, rank, md_qs[rank - 1], res_q, barrier), daemon=True)
        for rank in range(1, n_recv + 1)
    ]
    out: dict = {'recv': {}, 'trainer': {}, 'closed': {}}
    deadline = time.time() + timeout
    try:
        for p in procs:
            p.start()
        while len(out['closed']) < 1 + n_recv:
            try:
                msg = res_q.get(timeout=1.0)
            except queue_mod.Empty:
                if time.time() > deadline:
                    raise AssertionError(f'hccl cpu loopback timed out after {timeout}s; got {sorted(out["recv"])}')
                dead = [i for i, p in enumerate(procs) if p.exitcode not in (None, 0)]
                if dead:
                    raise AssertionError(f'processes {dead} died: {[procs[i].exitcode for i in dead]}')
                continue
            kind = msg[0]
            if kind == 'error':
                raise AssertionError(f'child failure in {msg[1]}:\n{msg[2]}')
            if kind == 'recv_round':
                out['recv'][(msg[1], msg[2])] = msg[3:]
            elif kind in ('trainer_round', 'trainer_finalized', 'trainer_group'):
                out['trainer'][(kind, msg[1])] = msg[2]
            elif kind == 'trainer_closed':
                out['closed'][0] = msg[1]
            elif kind == 'recv_closed':
                out['closed'][msg[1]] = msg[2]
    finally:
        for p in procs:
            p.join(timeout=10)
        for p in procs:
            if p.is_alive():
                p.kill()
                p.join(timeout=5)
        for q in [res_q, *md_qs]:
            q.close()
            q.cancel_join_thread()
    return out
