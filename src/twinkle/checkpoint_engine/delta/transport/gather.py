# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sparse gather of per-rank deltas to rank 0, based on verl."""

from __future__ import annotations

import torch
import torch.distributed as dist


def gather_slot_entries_to_rank0(
    idx_concat: torch.Tensor,
    val_concat: torch.Tensor,
    counts: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    max_round_bytes: int | None = None,
) -> list | None:
    """Variable-length sparse gather, batched: one collective round for K parameters.

    Each rank passes its K per-parameter deltas concatenated (``idx_concat``,
    ``val_concat``) plus the per-parameter length vector ``counts`` ([K] int64).
    One all_gather exchanges the K x world count matrix; two padded gathers move
    the blobs. Rank 0 slices per (rank, param) and returns K ``(idx, val)`` pairs
    (None elsewhere) -- bit-identical to K individual gathers, ~K x fewer
    collectives and host syncs.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    dst = dist.get_global_rank(group, 0) if group is not None else 0
    dev = idx_concat.device
    k = int(counts.numel())

    counts_all = [torch.zeros_like(counts) for _ in range(world)]
    dist.all_gather(counts_all, counts.to(dev), group=group)
    counts_cpu = torch.stack(counts_all).cpu().tolist()  # one D2H sync instead of `world`

    if max_round_bytes is not None and k > 1:
        # Deterministic sub-rounds: every rank sees the same counts matrix, so all
        # ranks derive the SAME slot partition (no per-rank trigger asymmetry) and
        # each padded round's largest blob stays within the byte budget.
        per_elem = idx_concat.element_size() + val_concat.element_size()
        budget = max(int(max_round_bytes) // per_elem, 1)  # elements per rank per round
        cuts = [0]
        run = [0] * world
        for i in range(k):
            run = [run[r] + counts_cpu[r][i] for r in range(world)]
            if max(run) > budget and cuts[-1] != i:
                cuts.append(i)
                run = [counts_cpu[r][i] for r in range(world)]
        cuts.append(k)
        if len(cuts) > 2:
            out_all: list = []
            my_off = [0]
            for i in range(k):
                my_off.append(my_off[-1] + counts_cpu[rank][i])
            for lo, hi in zip(cuts[:-1], cuts[1:], strict=False):
                sub_counts = torch.tensor(counts_cpu[rank][lo:hi], dtype=torch.int64, device=dev)
                sub = gather_slot_entries_to_rank0(
                    idx_concat[my_off[lo] : my_off[hi]],
                    val_concat[my_off[lo] : my_off[hi]],
                    sub_counts,
                    group=group,
                )
                if rank == 0:
                    out_all.extend(sub)
            return out_all if rank == 0 else None

    totals = [sum(c) for c in counts_cpu]
    max_n = max(totals) if totals else 0
    if max_n == 0:
        if rank != 0:
            return None
        empty_i = torch.empty(0, dtype=idx_concat.dtype, device=dev)
        empty_v = torch.empty(0, dtype=val_concat.dtype, device=dev)
        return [(empty_i, empty_v) for _ in range(k)]

    idx_pad = torch.zeros(max_n, dtype=idx_concat.dtype, device=dev)
    val_pad = torch.zeros(max_n, dtype=val_concat.dtype, device=dev)
    n = int(idx_concat.numel())
    idx_pad[:n] = idx_concat
    val_pad[:n] = val_concat

    idx_list = [torch.zeros(max_n, dtype=idx_pad.dtype, device=dev) for _ in range(world)] if rank == 0 else None
    val_list = [torch.zeros(max_n, dtype=val_concat.dtype, device=dev) for _ in range(world)] if rank == 0 else None
    dist.gather(idx_pad, idx_list, dst=dst, group=group)
    dist.gather(val_pad, val_list, dst=dst, group=group)
    if rank != 0:
        return None

    # per-rank cumulative offsets into each blob, sliced per param then stitched across ranks
    offs = [[0] * (k + 1) for _ in range(world)]
    for r in range(world):
        for i in range(k):
            offs[r][i + 1] = offs[r][i] + counts_cpu[r][i]
    out = []
    for i in range(k):
        idx_pieces = [idx_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
        val_pieces = [val_list[r][offs[r][i] : offs[r][i + 1]] for r in range(world) if counts_cpu[r][i]]
        if idx_pieces:
            out.append((torch.cat(idx_pieces), torch.cat(val_pieces)))
        else:
            out.append(
                (torch.empty(0, dtype=idx_concat.dtype, device=dev), torch.empty(0, dtype=val_concat.dtype, device=dev))
            )
    return out
