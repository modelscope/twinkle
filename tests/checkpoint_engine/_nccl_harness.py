# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-GPU harness for the NCCL checkpoint engines: one trainer (cuda:0) and ``n_recv`` receivers.

Everything the children need lives here (spawned processes must import it). Nothing is stubbed:
real ``NCCLCheckpointEngine`` / ``DeltaShardedCheckpointEngine``, real NCCL, real zmq. Expected
states are built by the parent, independently of the engine code.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue as queue_mod
import time
import traceback

import torch

from ._hccl_loopback import FakeTrainingEngine, apply_flush

BARRIER_TIMEOUT_S = 60


def make_full_state(seed: int) -> dict[str, torch.Tensor]:
    """Mixed dtypes; with a 4 KiB bucket the 4000-byte ``w2`` forces several buckets."""
    g = torch.Generator().manual_seed(seed)
    return {
        'layer.w1': torch.randn(16, 32, generator=g).to(torch.bfloat16),
        'layer.w2': torch.randn(1000, generator=g),
        'layer.idx': torch.randint(-1000, 1000, (64, ), generator=g),
        'layer.w3': torch.randn(8, 8, generator=g),
    }


def _child_setup(device_index: int) -> None:
    import twinkle.infra

    # Production engines run inside Ray actors; the node IP is the only Ray dependency.
    twinkle.infra.node_ip = lambda: '127.0.0.1'
    torch.cuda.set_device(device_index)


def _group_id(engine) -> int | None:
    group = getattr(engine, '_comm', None) or getattr(engine, '_pg', None)
    return None if group is None else id(group)


def _engine_state(engine) -> dict:
    comm = getattr(engine, '_comm', None)
    return dict(group=_group_id(engine), comm_destroyed=None if comm is None else comm.destroyed)


class CudaTrainingEngine(FakeTrainingEngine):
    """The real backends hand the delta engine device tensors."""

    def get_per_tensor_param(self):
        gen, pg = super().get_per_tensor_param()
        return ((n, t.cuda()) for n, t in gen), pg

    def get_per_tensor_param_delta_shard(self):
        gen, pg = super().get_per_tensor_param_delta_shard()
        return ((slots, dtype, counts, idx.cuda(), val.cuda(), g) for slots, dtype, counts, idx, val, g in gen), pg


# --------------------------------------------------------------------------- full-sync engine
def full_trainer_main(cfg, md_qs, res_q, barrier):
    try:
        from twinkle.checkpoint_engine import NCCLCheckpointEngine
        _child_setup(0)
        world_size = 1 + cfg['n_recv']
        eng = NCCLCheckpointEngine(bucket_size=cfg['bucket_size'], rebuild_group=cfg['rebuild_group'])
        eng.is_master = True
        for r, state in enumerate(cfg['rounds']):
            md = eng.prepare()
            if r == 0 or cfg['rebuild_group']:
                for q in md_qs:
                    q.put(md)
            eng.init_process_group(0, world_size, md)
            asyncio.run(eng.send_weights((n, t.cuda()) for n, t in state.items()))
            res_q.put(('trainer_round', r, _engine_state(eng)))
            barrier.wait(BARRIER_TIMEOUT_S)
            comm = eng._comm
            eng.finalize()
            res_q.put(('trainer_finalized', r, dict(_engine_state(eng), held_destroyed=comm.destroyed)))
        comm = eng._comm
        eng.close()
        res_q.put(('trainer_closed', dict(_engine_state(eng), held_destroyed=None if comm is None else comm.destroyed)))
    except BaseException:
        res_q.put(('error', 'trainer', traceback.format_exc()))


def full_receiver_main(cfg, rank, md_q, res_q, barrier):
    try:
        from twinkle.checkpoint_engine import NCCLCheckpointEngine
        _child_setup(rank)
        world_size = 1 + cfg['n_recv']
        eng = NCCLCheckpointEngine(bucket_size=cfg['bucket_size'], rebuild_group=cfg['rebuild_group'])
        md = None
        for r in range(len(cfg['rounds'])):
            if r == 0 or cfg['rebuild_group']:
                md = md_q.get(timeout=60)
            eng.prepare()
            eng.init_process_group(rank, world_size, md)

            async def consume():
                got = {}
                async for name, tensor in eng.receive_weights():
                    got[name] = tensor.clone()
                return got

            got = asyncio.run(consume())
            torch.cuda.synchronize()
            res_q.put(('recv_round', rank, r, {n: t.cpu() for n, t in got.items()}, _engine_state(eng)))
            barrier.wait(BARRIER_TIMEOUT_S)
            eng.finalize()
        eng.close()
        res_q.put(('recv_closed', rank, _engine_state(eng)))
    except BaseException:
        res_q.put(('error', f'receiver {rank}', traceback.format_exc()))


# --------------------------------------------------------------------------- delta engine
def delta_trainer_main(cfg, md_qs, res_q, barrier):
    try:
        from twinkle.checkpoint_engine.delta.transport.nccl_engine import DeltaShardedCheckpointEngine
        _child_setup(0)
        world_size = 1 + cfg['n_recv']
        eng = DeltaShardedCheckpointEngine(bucket_size=cfg['bucket_size'], batch_gather=4)
        eng.is_master = True
        md = eng.prepare()
        for q in md_qs:
            q.put(md)
        eng.init_process_group(0, world_size, md)
        train = CudaTrainingEngine(cfg['init_state'], cfg['rounds'])
        for r in range(1 + len(cfg['rounds'])):
            metrics = asyncio.run(eng.send_weights(train, global_steps=r))
            res_q.put(('trainer_round', r, metrics))
            barrier.wait(BARRIER_TIMEOUT_S)
        eng.close()
        res_q.put(('trainer_closed', _engine_state(eng)))
    except BaseException:
        res_q.put(('error', 'trainer', traceback.format_exc()))


def delta_receiver_main(cfg, rank, md_q, res_q, barrier):
    try:
        from twinkle.checkpoint_engine.delta.transport.nccl_engine import DeltaShardedCheckpointEngine
        _child_setup(rank)
        world_size = 1 + cfg['n_recv']
        eng = DeltaShardedCheckpointEngine(bucket_size=cfg['bucket_size'])
        eng.prepare()
        eng.init_process_group(rank, world_size, md_q.get(timeout=60))
        state: dict = {}
        for r in range(1 + len(cfg['rounds'])):
            flushes = 0
            for named, _is_last in eng.receive_weights(global_steps=r):
                apply_flush(state, named)
                flushes += 1
            torch.cuda.synchronize()
            res_q.put(('recv_round', rank, r, {n: t.cpu().clone() for n, t in state.items()}, flushes))
            barrier.wait(BARRIER_TIMEOUT_S)
        eng.close()
        res_q.put(('recv_closed', rank, _engine_state(eng)))
    except BaseException:
        res_q.put(('error', f'receiver {rank}', traceback.format_exc()))


# --------------------------------------------------------------------------- driver (parent)
def run(mode: str, cfg: dict, timeout: float = 120.0) -> dict:
    """Run trainer + receivers; returns every message keyed by kind. Children are always killed."""
    trainer, receiver = {'full': (full_trainer_main, full_receiver_main),
                         'delta': (delta_trainer_main, delta_receiver_main)}[mode]
    ctx = mp.get_context('spawn')
    n_recv = cfg['n_recv']
    res_q = ctx.Queue()
    md_qs = [ctx.Queue() for _ in range(n_recv)]
    barrier = ctx.Barrier(1 + n_recv)
    procs = [ctx.Process(target=trainer, args=(cfg, md_qs, res_q, barrier), daemon=True)]
    procs += [
        ctx.Process(target=receiver, args=(cfg, rank, md_qs[rank - 1], res_q, barrier), daemon=True)
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
                    raise AssertionError(f'{mode} sync timed out after {timeout}s; got {sorted(out["recv"])}')
                dead = [i for i, p in enumerate(procs) if p.exitcode not in (None, 0)]
                if dead:
                    raise AssertionError(f'processes {dead} died: {[procs[i].exitcode for i in dead]}')
                continue
            kind = msg[0]
            if kind == 'error':
                raise AssertionError(f'child failure in {msg[1]}:\n{msg[2]}')
            if kind == 'recv_round':
                out['recv'][(msg[1], msg[2])] = msg[3:]
            elif kind in ('trainer_round', 'trainer_finalized'):
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
