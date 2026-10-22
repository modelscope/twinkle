# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations
import random
from contextlib import contextmanager
from typing import Any, Iterator, NamedTuple, Tuple

import numpy as np
import torch

_MASK_64 = (1 << 64) - 1
_MASK_63 = (1 << 63) - 1


class SampleIndex(NamedTuple):
    """A map-style source index together with the epoch that selected it."""

    index: Any
    epoch: int


def split_sample_index(index: Any) -> Tuple[Any, int]:
    if isinstance(index, SampleIndex):
        return index.index, index.epoch
    return index, 0


def derive_seed(seed: int, *parts: int) -> int:
    """Combine integer coordinates into one stable seed without process-local hashing."""
    value = int(seed or 0) & _MASK_64
    for part in parts:
        value = (value + 0x9E3779B97F4A7C15 + (int(part) & _MASK_64)) & _MASK_64
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
        value ^= value >> 31
    return value & _MASK_63


def seed_process_rng(seed: int) -> None:
    """Seed the process-local Python, NumPy and Torch CPU generators."""
    seed = int(seed) & _MASK_63
    random.seed(seed)
    np.random.seed(seed % 2**32)
    generator = torch.Generator()
    generator.manual_seed(seed)
    torch.set_rng_state(generator.get_state())


@contextmanager
def isolated_random_state(seed: int, *parts: int) -> Iterator[int]:
    """Run one logical data operation with deterministic RNGs, then restore worker state."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    local_seed = derive_seed(seed, *parts)
    seed_process_rng(local_seed)
    try:
        yield local_seed
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
