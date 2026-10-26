# Copyright (c) ModelScope Contributors. All rights reserved.
"""Multi-process loopback harness for the HCCL delta-sync protocol (CPU: gloo + zmq, no NPU).

Everything the children need lives here (spawned processes must import it). Real code under test:
trainer ``DeltaHCCLCheckpointEngine`` (lazy communicator, zmq PUB, ``DeltaSyncMixin``) and receiver
``connect_subscriber`` + ``build_delta_hccl_communicator`` + ``pump_hccl_delta_session`` +
``decode_delta_wire``. Only ``PyHccl`` (-> gloo broadcast), ``torch.npu`` and the training engine
are stand-ins.
"""
from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue as queue_mod
import random
import time
import traceback
from datetime import timedelta
from types import SimpleNamespace

import torch

ENGINE_KW = dict(batch_gather=4)
GROUP_TIMEOUT_S = 40
BF16 = torch.bfloat16


# --------------------------------------------------------------------------- model / rounds
def make_state(seed: int = 0, shapes=None) -> dict[str, torch.Tensor]:
    shapes = shapes or {'layer.w1': (16, 32), 'layer.w2': (4000, ), 'layer.bias': (64, )}
    g = torch.Generator().manual_seed(seed)
    return {n: torch.randn(*s, generator=g).to(BF16) for n, s in shapes.items()}


def build_rounds(init_state, change_counts: list[dict[str, int]], seed: int = 1):
    """Steady rounds from ``{name: n_changed}`` dicts.

    Returns ``(rounds, expected)``: ``rounds[i] = {name: (idx int64 sorted, val bf16)}`` and
    ``expected[0]`` = seed state, ``expected[i+1]`` = state after steady round i (independently
    computed here, so it does not depend on trainer/receiver code).
    """
    g = torch.Generator().manual_seed(seed)
    cur = {n: t.clone() for n, t in init_state.items()}
    expected = [{n: t.clone() for n, t in cur.items()}]
    rounds = []
    for counts in change_counts:
        rnd = {}
        for name, n in counts.items():
            numel = cur[name].numel()
            idx = torch.randperm(numel, generator=g)[:n].sort().values
            val = (cur[name].view(-1)[idx].float() + 1.0 + torch.rand(n, generator=g)).to(BF16)
            cur[name].view(-1).index_copy_(0, idx, val)
            rnd[name] = (idx, val)
        rounds.append(rnd)
        expected.append({n: t.clone() for n, t in cur.items()})
    return rounds, expected


def state_to_bytes(state) -> dict:
    return {n: (tuple(t.shape), str(t.dtype), t.contiguous().view(torch.uint8).numpy().tobytes())
            for n, t in state.items()}


def wire_bytes(metas) -> int:
    """Payload bytes on the broadcast wire for a list of (non-terminal) manifests."""
    return sum(m.get('pos_numel', 0) + m['val_numel'] * 2 for m in metas if not m.get('terminal_empty'))


# --------------------------------------------------------------------------- stand-ins
class LoopComm:
    """PyHccl stand-in: broadcast over the gloo default group of this process."""

    def __init__(self):
        self.comm = 'handle'

    def broadcast(self, tensor, src):
        import torch.distributed as dist
        assert tensor.dtype == torch.uint8
        dist.broadcast(tensor, src=src)

    def destroyComm(self, handle):
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()


class CorruptingComm(LoopComm):
    """Rank 0 only: flips one bit of the ``corrupt_call``-th broadcast buffer (after the checksum)."""

    def __init__(self, corrupt_call):
        super().__init__()
        self.corrupt_call = corrupt_call
        self.calls = 0

    def broadcast(self, tensor, src):
        if self.calls == self.corrupt_call:
            tensor[0] ^= 1
        self.calls += 1
        super().broadcast(tensor, src)


def _install_stand_ins(comm_factory, build_log):
    """Patch torch.npu and stateless_init_process_group in BOTH modules (child process only)."""
    import torch.distributed as dist

    from twinkle.checkpoint_engine.delta.rollout import hccl_receiver
    from twinkle.checkpoint_engine.delta.transport import hccl_engine

    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    torch.set_num_threads(1)
    torch.npu = SimpleNamespace(current_device=lambda: 0, synchronize=lambda: None, empty_cache=lambda: None)

    def fake_init(master_address, master_port, rank, world_size, device=None, backend='nccl', **kw):
        assert backend == 'hccl'
        # blocks until every rank joined: the real HCCL rendezvous semantics
        dist.init_process_group('gloo', init_method=f'tcp://{master_address}:{master_port}', rank=rank,
                                world_size=world_size, timeout=timedelta(seconds=GROUP_TIMEOUT_S))
        build_log.append(rank)
        return comm_factory(rank)

    hccl_engine.stateless_init_process_group = fake_init
    hccl_receiver.stateless_init_process_group = fake_init
    hccl_engine.find_node_ip = lambda: '127.0.0.1'


class FakeTrainingEngine:
    """get_per_tensor_param / prime_delta_snapshots / get_per_tensor_param_delta_shard stand-in."""

    def __init__(self, init_state, rounds):
        self.state = {n: t.clone() for n, t in init_state.items()}
        self.rounds = rounds
        self.next_round = 0
        self.primed = 0

    def get_per_tensor_param(self):
        return ((n, t.clone()) for n, t in self.state.items()), None

    def prime_delta_snapshots(self):
        self.primed += 1

    def get_per_tensor_param_delta_shard(self):
        changes = self.rounds[self.next_round]
        self.next_round += 1

        def gen():
            for name, (idx, val) in changes.items():
                shape = tuple(self.state[name].shape)
                yield [(name, shape)], 'bfloat16', torch.tensor([idx.numel()]), idx.clone(), val.clone(), None

        return gen(), None


# --------------------------------------------------------------------------- trainer child
def trainer_main(cfg, ep_qs, res_q, done_evs):
    try:
        build_log: list = []

        def factory(rank):
            if cfg.get('corrupt_call') is not None:
                return CorruptingComm(cfg['corrupt_call'])
            return LoopComm()

        _install_stand_ins(factory, build_log)
        from twinkle.checkpoint_engine.delta.transport.hccl_engine import DeltaHCCLCheckpointEngine
        from twinkle.checkpoint_engine.delta.transport.pipeline import DeltaSyncMixin

        class LoopbackEngine(DeltaHCCLCheckpointEngine):

            @property
            def _wire_device(self):
                return torch.device('cpu')

        if cfg.get('max_entry_elems'):
            DeltaSyncMixin.MAX_ENTRY_ELEMS = cfg['max_entry_elems']
        world_size = 1 + cfg['n_recv']
        for s, (init_state, rounds) in enumerate(cfg['sessions']):
            eng = LoopbackEngine(bucket_size=cfg['bucket_size'], **ENGINE_KW)
            eng.is_master = True
            md = eng.prepare()
            assert eng.prepare() == md  # idempotent
            builds_before = len(build_log)
            for _ in range(cfg['n_recv']):
                ep_qs[s].put(dict(zmq_ip=md.zmq_ip, zmq_port=md.zmq_port, dist_ip=md.dist_ip,
                                  dist_port=md.dist_port, world_size=world_size))
            eng.init_process_group(0, world_size, md)
            assert eng._comm is None  # lazy: nothing built before the first publish
            train = FakeTrainingEngine(init_state, rounds)
            for r in range(1 + len(rounds)):
                metrics = asyncio.run(eng.send_weights(train, global_steps=r))
                res_q.put(('trainer_round', s, r, metrics, len(build_log) - builds_before))
            res_q.put(('trainer_sent', s, len(build_log) - builds_before, train.primed))
            if not done_evs[s].wait(60):
                raise RuntimeError('receivers never finished')
            eng.close()
            closed = dict(comm=eng._comm, socket=eng.socket, seeded=eng._shard_seeded, rank=eng.rank)
            res_q.put(('trainer_closed', s, closed))
        res_q.put(('trainer_exit', ))
    except BaseException:
        res_q.put(('error', 'trainer', traceback.format_exc()))


# --------------------------------------------------------------------------- receiver child
class RecordingSub:

    def __init__(self, sub):
        self.sub = sub
        self.metas: list = []

    def recv_string(self):
        return self.sub.recv_string()

    def recv_pyobj(self):
        meta = self.sub.recv_pyobj()
        self.metas.append(meta)
        return meta


def apply_flush(state, named):
    from twinkle.checkpoint_engine.delta.protocol.decode import decode_delta_wire
    _enc, patches = decode_delta_wire(named)  # includes the checksum check
    for p in patches:
        if p.positions is None:
            n = 1
            for d in p.shape:
                n *= d
            assert p.values.numel() == n, (p.name, p.values.numel(), n)
            state[p.name] = p.values.clone().reshape(p.shape)  # dense: replace the whole param
        else:
            flat = state[p.name].view(-1)
            flat.index_copy_(0, p.positions.long(), p.values)


def receiver_main(cfg, ep_q, res_q):
    s, rank = cfg['session'], cfg['group_rank']
    try:
        build_log: list = []
        _install_stand_ins(lambda r: LoopComm(), build_log)
        import zmq

        from twinkle.checkpoint_engine.delta.rollout.hccl_receiver import (build_delta_hccl_communicator,
                                                                   pump_hccl_delta_session)
        from twinkle.checkpoint_engine.zmq_pubsub import connect_subscriber
        time.sleep(cfg.get('start_delay', 0))
        ep = ep_q.get(timeout=60)
        device = torch.device('cpu')
        sub = None
        if not cfg.get('sub_after_join'):
            sub = connect_subscriber(ep['zmq_ip'], ep['zmq_port'], 'bucket_metadata')  # SUB first ...
        comm = build_delta_hccl_communicator(ep['dist_ip'], ep['dist_port'], rank, ep['world_size'], device)
        if cfg.get('sub_after_join'):
            time.sleep(cfg['sub_after_join'])  # ... the wrong order, made deterministic
            sub = connect_subscriber(ep['zmq_ip'], ep['zmq_port'], 'bucket_metadata')
        sub.setsockopt(zmq.RCVTIMEO, cfg.get('pump_timeout_ms', 20000))
        rsub = RecordingSub(sub)
        state: dict = {}
        for r in range(cfg['n_rounds']):
            del rsub.metas[:]
            err = None
            try:
                enc = pump_hccl_delta_session(rsub, comm, device, lambda named: apply_flush(state, named))
            except Exception as e:  # reported, then this receiver stops
                enc, err = None, (type(e).__name__, str(e))
            res_q.put(('recv_round', s, rank, r, dict(state=state_to_bytes(state), metas=list(rsub.metas),
                                                      encoding=enc, error=err)))
            if err:
                break
        res_q.put(('recv_finished', s, rank))
        try:
            comm.destroyComm(comm.comm)
        except Exception:
            pass
    except BaseException:
        res_q.put(('error', f'receiver s{s} r{rank}', traceback.format_exc()))


# --------------------------------------------------------------------------- driver (parent)
class Result:

    def __init__(self):
        self.recv: dict = {}  # (session, rank) -> {round: payload}
        self.trainer_rounds: dict = {}  # (session, round) -> (metrics, comm builds so far)
        self.trainer_sent: dict = {}  # session -> (builds, primed)
        self.trainer_closed: dict = {}
        self.finished: set = set()
        self.errors: list = []
        self.trainer_exited = False
        self.elapsed = 0.0


def run_loopback(sessions, n_recv=2, bucket_size=1 << 20, max_entry_elems=None, corrupt_call=None,
                 sub_after_join=0.0, pump_timeout_ms=20000, shuffle=False, timeout=60.0, allow_errors=False,
                 require_trainer_exit=True, seed=0) -> Result:
    """Run ``sessions`` = [(init_state, rounds), ...] across 1 trainer + n_recv receiver processes.

    Session i>0 reuses the trainer process with a NEW engine (fresh ports/group) after ``close()``,
    and gets NEW receiver processes. Always kills leftover children.
    """
    ctx = mp.get_context('spawn')
    res_q = ctx.Queue()
    ep_qs = [ctx.Queue() for _ in sessions]
    done_evs = [ctx.Event() for _ in sessions]
    tcfg = dict(sessions=sessions, n_recv=n_recv, bucket_size=bucket_size, max_entry_elems=max_entry_elems,
                corrupt_call=corrupt_call)
    procs = [ctx.Process(target=trainer_main, args=(tcfg, ep_qs, res_q, done_evs), daemon=True)]
    rng = random.Random(seed)
    for s, (_st, rounds) in enumerate(sessions):
        ranks = list(range(1, n_recv + 1))
        if shuffle:
            rng.shuffle(ranks)
        for rk in ranks:
            rcfg = dict(session=s, group_rank=rk, n_rounds=1 + len(rounds), pump_timeout_ms=pump_timeout_ms,
                        sub_after_join=sub_after_join, start_delay=rng.random() * 0.8 if shuffle else 0.0)
            procs.append(ctx.Process(target=receiver_main, args=(rcfg, ep_qs[s], res_q), daemon=True))
    res = Result()
    t0 = time.time()
    deadline = t0 + timeout
    try:
        for p in procs:
            p.start()
        while True:
            all_recv_done = all((s, rk) in res.finished for s in range(len(sessions)) for rk in range(1, n_recv + 1))
            if all_recv_done and (res.trainer_exited or not require_trainer_exit):
                break
            try:
                msg = res_q.get(timeout=max(0.05, min(1.0, deadline - time.time())))
            except queue_mod.Empty:
                if time.time() >= deadline:
                    if all_recv_done and not require_trainer_exit:
                        break
                    raise AssertionError(f'loopback timed out after {timeout}s; got so far: finished={res.finished} '
                                         f'trainer_rounds={sorted(res.trainer_rounds)} errors={res.errors}')
                continue
            kind = msg[0]
            if kind == 'error':
                res.errors.append(msg[1:])
                if not allow_errors:
                    raise AssertionError(f'child failure in {msg[1]}:\n{msg[2]}')
            elif kind == 'recv_round':
                _, s, rk, r, payload = msg
                res.recv.setdefault((s, rk), {})[r] = payload
            elif kind == 'recv_finished':
                res.finished.add((msg[1], msg[2]))
                if all((msg[1], k) in res.finished for k in range(1, n_recv + 1)):
                    done_evs[msg[1]].set()
            elif kind == 'trainer_round':
                res.trainer_rounds[(msg[1], msg[2])] = (msg[3], msg[4])
            elif kind == 'trainer_sent':
                res.trainer_sent[msg[1]] = (msg[2], msg[3])
            elif kind == 'trainer_closed':
                res.trainer_closed[msg[1]] = msg[2]
            elif kind == 'trainer_exit':
                res.trainer_exited = True
    finally:
        res.elapsed = time.time() - t0
        for ev in done_evs:
            ev.set()
        for p in procs:
            p.join(timeout=3 if res.trainer_exited or p is not procs[0] else 0.1)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=3)
            if p.is_alive():
                p.kill()
                p.join(timeout=3)
        for q in [res_q, *ep_qs]:
            q.close()
            q.cancel_join_thread()
    return res
