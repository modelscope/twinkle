# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import os
import re
import socket
import subprocess
from typing import Optional

from .gpu import GPU


def _get_xpu_bus_id_from_xpu_smi(device_id: int) -> Optional[str]:
    """Get XPU Bus-Id from `xpu-smi -q` output.
    """
    try:
        output = subprocess.check_output(
            ['xpu-smi', '-q'],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except Exception:
        return None

    pattern = re.compile(
        r'^XPU\s+(\d+).*?Bus Id\s*:\s*([0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])',
        re.MULTILINE | re.DOTALL,
    )
    for match in pattern.finditer(output):
        if int(match.group(1)) == device_id:
            return match.group(2).lower()
    return None


def _get_xpu_uuid_from_xpu_smi(device_id: int) -> Optional[str]:
    """Get XPU UUID from `xpu-smi -L` output.
    """
    try:
        output = subprocess.check_output(
            ['xpu-smi', '-L'],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except Exception:
        return None

    pattern = re.compile(r'^XPU\s+(\d+):\s+\S.*\(UUID:\s*(\S+)\)', re.MULTILINE)
    for match in pattern.finditer(output):
        if int(match.group(1)) == device_id:
            return match.group(2)
    return None


def ensure_xpu_compat() -> None:
    """Neutralize the native Intel-XPU stub shipped inside torch.
    """
    import torch

    xp = getattr(torch, 'xpu', None)
    if xp is None:
        return
    try:
        if xp.is_available():
            return  # a real (Intel) XPU runtime is present; nothing to stub
    except Exception:
        return
    if not getattr(xp.get_device_name, '_twinkle_xpu_stub', False):
        def _get_device_name(*args, **kwargs):  # noqa: E306
            return 'Kunlunxin XPU'
        _get_device_name._twinkle_xpu_stub = True
        xp.get_device_name = _get_device_name


def ensure_bkcl_nonuniform_env() -> None:
    """Enable BKCL's non-uniform-visible-devices mode for Kunlunxin XPU.

    BKCL identifies a rank's device by the *local* device index returned by
    ``xpu_current_device()`` (see bkcl/src/init.cpp: ``msg.dev_id =
    device_id``). twinkle's Ray layout gives each single-card worker its own
    ``CUDA_VISIBLE_DEVICES`` and ``LOCAL_RANK=0``, so every rank in a group
    reports local index 0 -> ``rank2dev`` collision -> ring build fails ->
    collective deadlock.

    Newer BKCL builds add ``BKCL_NONUNIFORM_VISIBLE_DEVICES``: when set, BKCL
    reads each rank's PCI address (``XPUATTR_PCI_ADDRESS``) and translates it
    to a physical device id (``pci_addr_to_default_dev_id_and_phy_id``),
    correctly distinguishing ranks that each expose a single distinct card as
    local index 0. This is the vendor-intended way to support the
    one-visible-card-per-process deployment, so we opt in by default (unless
    the user has explicitly set the variable).
    """
    os.environ.setdefault('BKCL_NONUNIFORM_VISIBLE_DEVICES', '1')


class XPU(GPU):

    @staticmethod
    def visible_device_env():
        # Kunlunxin XPU runtime takes over CUDA_VISIBLE_DEVICES semantics.
        return 'CUDA_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        # torch_xmlir symbol-rewrites torch.cuda to XPU, so tensors live on
        # 'cuda:N' devices from PyTorch's point of view.
        return 'cuda'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return f'cuda:{idx}'

    @staticmethod
    def device_backend(platform: str = None):
        # BKCL is mounted into torch as the 'nccl' backend (XCCL).
        return 'nccl'

    @staticmethod
    def get_vllm_device_uuid(device_id: int = 0) -> str:
        from vllm.platforms import current_platform
        try:
            return current_platform.get_device_uuid(device_id)
        except NotImplementedError:
            bus_id = _get_xpu_bus_id_from_xpu_smi(device_id)
            if bus_id:
                return bus_id
            xpu_uuid = _get_xpu_uuid_from_xpu_smi(device_id)
            if xpu_uuid:
                return xpu_uuid
            # Deterministic fallback so both sides compute the same socket name.
            visible = os.environ.get(XPU.visible_device_env())
            raw = f'{socket.gethostname()}:{visible}:{device_id}'
            return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    @staticmethod
    def bootstrap() -> None:
        ensure_xpu_compat()
        # Opt into BKCL's non-uniform-visible-devices mode so that
        # single-card-per-worker Ray layouts (LOCAL_RANK=0 on every rank) are
        # mapped by physical PCI id instead of the colliding local index.
        ensure_bkcl_nonuniform_env()

    @staticmethod
    def requires_unbound_default_pg() -> bool:
        # BKCL is mounted as the 'nccl' backend, but its ``ProcessGroupXCCL``
        # has no ``perform_nocolor_split``. A leftover ``bound_device_id``
        # makes Megatron's ``new_group`` take the comm-split path and crash.
        return True

    @staticmethod
    def requires_unbound_pg_for_subgroups() -> bool:
        # Same root cause as above, but Megatron may run on a default PG that
        # twinkle did not initialize, so the binding has to be cleared again
        # immediately before ``initialize_model_parallel``.
        return True

    @staticmethod
    def ray_resources():
        # The XPU runtime ships no nvidia-smi, so Ray autodetects zero GPUs.
        # Register the cuda-alike count explicitly; ``RayHelper`` routes the
        # 'GPU' key through ``num_gpus`` because Ray rejects its own default
        # resource names in ``resources=``.
        try:
            import torch

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except (ImportError, AttributeError, RuntimeError):
            return {}
        return {'GPU': float(gpu_count)} if gpu_count > 0 else {}

    @staticmethod
    def checkpoint_engine():
        # Must be overridden explicitly: inheriting GPU's NCCL engine would
        # deadlock, since directly constructing ``ProcessGroupNCCL`` hangs
        # when two ranks share a local device index.
        from twinkle.checkpoint_engine import XCCLCheckpointEngine
        return XCCLCheckpointEngine, {}
