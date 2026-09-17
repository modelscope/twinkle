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
