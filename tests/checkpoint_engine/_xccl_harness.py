# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-card harness for :class:`XCCLCheckpointEngine`: one trainer (cuda:0) + ``n_recv`` receivers.

Everything the children need lives here, because spawned processes have to import it. Nothing
is stubbed: the real engine, a real XCCL group, real zmq. The expected weights are built by the
parent, independently of the engine code.
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


def _child_setup(device_index: int) -> None:
    import twinkle.infra

    # Production engines run inside Ray actors; the node IP is the only Ray dependency.
    twinkle.infra.node_ip = lambda: '127.0.0.1'
    # Result-queue tensors must outlive the sender: torch's default CPU sharing hands over a
    # file descriptor served by the sending process's resource_sharer socket, and a child that
    # has put its last message and exited takes that socket with it.
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.cuda.set_device(device_index)


def _group_id(engine) -> int | None:
    return None if engine._comm is None else id(engine._comm)


def trainer_main(cfg, md_qs, res_q, barrier):
    try:
        from twinkle.checkpoint_engine import XCCLCheckpointEngine
        _child_setup(0)
        world_size = 1 + cfg['n_recv']
        eng = XCCLCheckpointEngine(bucket_size=cfg['bucket_size'], rebuild_group=cfg['rebuild_group'])
        eng.is_master = True
        for r, state in enumerate(cfg['rounds']):
            md = eng.prepare()
            if r == 0 or cfg['rebuild_group']:
                for q in md_qs:
                    q.put(md)
            eng.init_process_group(0, world_size, md)
            asyncio.run(eng.send_weights((n, t.cuda()) for n, t in state.items()))
            res_q.put(('trainer_round', r, _group_id(eng)))
            barrier.wait(BARRIER_TIMEOUT_S)
            comm = eng._comm
            eng.finalize()
            res_q.put(('trainer_finalized', r, dict(group=_group_id(eng), held_destroyed=comm.destroyed)))
        eng.close()
        res_q.put(('trainer_closed', dict(group=_group_id(eng))))
    except BaseException:
        res_q.put(('error', 'trainer', traceback.format_exc()))


def receiver_main(cfg, rank, md_q, res_q, barrier):
    try:
        from twinkle.checkpoint_engine import XCCLCheckpointEngine
        _child_setup(rank)
        world_size = 1 + cfg['n_recv']
        eng = XCCLCheckpointEngine(bucket_size=cfg['bucket_size'], rebuild_group=cfg['rebuild_group'])
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
            res_q.put(('recv_round', rank, r, {n: t.cpu() for n, t in got.items()}, _group_id(eng)))
            barrier.wait(BARRIER_TIMEOUT_S)
            eng.finalize()
        eng.close()
        res_q.put(('recv_closed', rank, dict(group=_group_id(eng))))
    except BaseException:
        res_q.put(('error', f'receiver {rank}', traceback.format_exc()))


def run(cfg: dict, timeout: float = 180.0) -> dict:
    """Run trainer + receivers; returns every message keyed by kind. Children are always killed."""
    torch.multiprocessing.set_sharing_strategy('file_system')
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
                    raise AssertionError(f'xccl sync timed out after {timeout}s; got {sorted(out["recv"])}')
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
            if p.is_alive():
                p.kill()
            p.join()
    return out
