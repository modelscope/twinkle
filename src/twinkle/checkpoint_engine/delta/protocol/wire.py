# Copyright (c) ModelScope Contributors. All rights reserved.
"""Delta payload schema and checksum based on verl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeltaEncodingName = Literal["indices"]


# ---------- diff ----------------------------------------------------------


@dataclass
class DeltaParam:
    """Per-parameter manifest entry for a single chunk / bucket.

    Offsets are byte offsets into the surrounding ``__positions__`` blob and
    element offsets into the surrounding ``__values__`` tensor.
    """

    name: str
    dtype: str
    shape: list[int]
    pos_start: int
    pos_end: int
    pos_width: int  # 2 or 4
    val_start: int
    val_end: int


def _xor_tree(t: torch.Tensor) -> int:
    """Device-side XOR fold of a tensor's raw bytes (log2(n) rounds, one sync).

    Works on any device (including NPU), unlike ``torch.hash_tensor`` which only
    has CPU/CUDA kernels. The fold is out-of-place, so the argument's storage is
    never mutated, and the result is a pure function of the bytes, independent
    of dtype, so sender and receiver agree anywhere.
    """
    if t.numel() == 0:
        return 0
    flat = t.contiguous().reshape(-1)
    v = flat.view(torch.uint8) if flat.dtype != torch.uint8 else flat
    # Halving XOR fold; an odd middle element carries to the next round.
    while v.numel() > 1:
        n = v.numel()
        half = n // 2
        if n % 2 == 0:
            v = v[:half] ^ v[half:]
        else:
            v = torch.cat((v[:half] ^ v[half + 1 :], v[half : half + 1]))
    return int(v.item())


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Wire-corruption check; sender computes pre-flush, receiver post-recv.

    XOR tree over the bytes of each argument (device-side, one ``.item()`` sync
    per argument). The same function runs on both ends, so the value is stable
    across dtypes and devices.
    """
    p = _xor_tree(positions)
    v = _xor_tree(values)
    return (p ^ (v << 1)) & ((1 << 64) - 1)


# ---------- encode --------------------------------------------------------


@dataclass
class DeltaFlush:
    """One ready-to-dispatch flush.

    * ``positions_cpu`` is a uint8 positions blob. Despite the name it lives on
      the GPU in the sharded engine (the wire broadcasts from the GPU, so a
      host round-trip would be pure overhead).
    * ``values_gpu`` stays on the GPU until the checkpoint engine broadcasts it
      over NCCL.
    * ``params`` carries the per-parameter manifest the receiver needs to
      decode the blob (sent alongside the data over the zmq side-channel).
    """

    encoding: DeltaEncodingName
    params: list[DeltaParam]
    positions_cpu: torch.Tensor
    values_gpu: torch.Tensor
    checksum: int
