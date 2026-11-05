# Copyright (c) ModelScope Contributors. All rights reserved.
"""Metax (MACA) GPU platform.

The MACA stack is CUDA-compatible at the torch level (``torch.cuda.*`` works), so the platform
class inherits GPU behavior wholesale: ``CUDA_VISIBLE_DEVICES``, ``cuda:N`` devices and the nccl
communicator path. The only Metax-specific work is applying the fixes shipped inside the
``vllm_metax`` wheel before vLLM engine use -- the same arrangement ms-swift relied on
(``import vllm_metax.patch`` alongside its vllm imports).
"""
import shutil
import sys

from .gpu import GPU


def is_metax_available() -> bool:
    """Detect Metax hardware by its management CLI (``mx-smi`` is the canonical check)."""
    return shutil.which('mx-smi') is not None


def ensure_vllm_metax_patch() -> None:
    """Apply the ``vllm_metax`` fixes once per process, before vLLM engine use.

    No-op when the wheel is absent (generic CUDA hosts never ship it). Importing the patch
    after ``vllm`` itself is fine: ms-swift imported it after its ``from vllm...`` imports too,
    and the patch monkeypatches in place.
    """
    import importlib
    import importlib.util
    if importlib.util.find_spec('vllm_metax') is None:
        return
    if 'vllm_metax.patch' not in sys.modules:
        importlib.import_module('vllm_metax.patch')


class Metax(GPU):
    """Metax MACA GPU: inherits every CUDA-compat behavior from GPU."""

    @staticmethod
    def ensure_vllm_patch() -> None:
        ensure_vllm_metax_patch()
