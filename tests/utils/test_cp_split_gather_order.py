# Copyright (c) ModelScope Contributors. All rights reserved.
"""#10071: the CP split and its inverse must restore the original token order.

Context parallelism slices each packed segment into ``2*cp`` chunks and gives rank ``r`` chunks
``[r, 2*cp-1-r]`` (load-balanced). If the inverse gather put those chunks back in a different order,
every CP rank would score a permuted sequence: token ``i``'s prediction would be matched against a
neighbour's label while every shape still checks out. The two functions are exclusive inverses, so a
round trip over distinguishing values proves the order survives.
"""
import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import pytest

pytest.importorskip('megatron.core')


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _worker(rank: int, world: int, port: int, out) -> None:
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group('gloo', rank=rank, world_size=world)
    try:
        from megatron.core import mpu
        from twinkle.utils.torch_utils import gather_cp_load_balanced, split_cp_inputs

        mpu.get_context_parallel_world_size = lambda: world
        mpu.get_context_parallel_rank = lambda: rank

        # Two packed segments (40 + 24 tokens); every value is distinct so any reordering shows up.
        cu_seqlens = torch.tensor([0, 40, 64])
        original = torch.arange(64, dtype=torch.float32).view(1, 64)

        local = split_cp_inputs(original, cu_seqlens, dim=1)
        gathered = gather_cp_load_balanced(local, dist.group.WORLD, seq_dim=1, cu_seqlens=cu_seqlens)
        out[rank] = (torch.equal(gathered, original), gathered.tolist())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('world', [2, 4])
def test_cp_split_gather_restores_order(world):
    manager = mp.Manager()
    out = manager.dict()
    mp.spawn(_worker, args=(world, _free_port(), out), nprocs=world, join=True)
    for rank in range(world):
        equal, values = out[rank]
        assert equal, f'rank {rank} gathered a reordered sequence: {values}'
