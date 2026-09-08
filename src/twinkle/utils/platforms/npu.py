# Copyright (c) ModelScope Contributors. All rights reserved.
import hashlib
import os
import platform
import re
import socket
import subprocess
from typing import Optional

from packaging import version

from .base import Platform

# ref: https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0144.html
# HCCL base port anchor. HCCL derives internal listen/connect ports from this base.
_HCCL_IF_BASE_PORT_ENV = 'HCCL_IF_BASE_PORT'
# Host-side socket port pool used by HCCL in multi-process communication.
_HCCL_HOST_SOCKET_PORT_RANGE_ENV = 'HCCL_HOST_SOCKET_PORT_RANGE'
# NPU-side socket port pool used by HCCL for device communication channels.
_HCCL_NPU_SOCKET_PORT_RANGE_ENV = 'HCCL_NPU_SOCKET_PORT_RANGE'

def _derive_hccl_socket_env_defaults(master_port: int) -> dict:
    """Derive deterministic default HCCL socket env values from master_port."""
    # Keep values stable per job and spread jobs across non-overlapping ranges.
    host_offset = master_port % 8000
    return {
        _HCCL_IF_BASE_PORT_ENV: str(20000 + ((master_port + 997) % 20000)),
        _HCCL_HOST_SOCKET_PORT_RANGE_ENV: f'{40000 + host_offset}-{40000 + host_offset + 511}',
        _HCCL_NPU_SOCKET_PORT_RANGE_ENV: f'{50000 + host_offset}-{50000 + host_offset + 511}',
    }


def ensure_hccl_socket_env(master_port: int, environ: Optional[dict] = None) -> None:
    """Set deterministic HCCL socket env defaults to avoid port collisions.

    In multi-job environments, HCCL's default base port (60000) can collide
    across concurrent jobs and lead to:
    `ra_hdc_socket_listen_start ... ret(-98)`.

    We derive a per-job port layout from `master_port` so all ranks use the
    same values while reducing cross-job conflicts. Explicit user settings are
    preserved and never overwritten.
    """
    # fix: We hit `ra_hdc_socket_listen_start ... ret(-98)` due to HCCL port collisions.
    # fix: Derive stable ranges from master_port and preserve explicit user overrides.
    env = os.environ if environ is None else environ
    for key, value in _derive_hccl_socket_env_defaults(master_port).items():
        env.setdefault(key, value)


def _resolve_ascend_physical_device_id(device_id: int) -> int:
    """Map local NPU device index to physical device id via visible devices."""
    visible = os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '').strip()
    if not visible:
        return device_id
    parts = [p.strip() for p in visible.split(',') if p.strip()]
    if device_id < 0 or device_id >= len(parts):
        return device_id
    return int(parts[device_id])


def _get_npu_bus_id_from_npu_smi(device_id: int) -> Optional[str]:
    """Get NPU Bus-Id from `npu-smi info` output."""
    try:
        physical_id = _resolve_ascend_physical_device_id(device_id)
    except Exception:
        physical_id = device_id

    try:
        output = subprocess.check_output(
            ['npu-smi', 'info'],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=5,
        )
    except Exception:
        return None

    # fix: vllm-ascend may not implement get_device_uuid, but we still need a reproducible cross-process device id.
    # fix: Prefer physical Bus-Id parsed from npu-smi instead of unstable/random identifiers.
    # Typical line:
    # | 0     0                   | 0000:9D:00.0  | ...
    pattern = re.compile(
        r'^\|\s*\d+\s+(\d+)\s*\|\s*'
        r'([0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])\s*\|',
        re.MULTILINE,
    )
    for match in pattern.finditer(output):
        phy_id = int(match.group(1))
        if phy_id == physical_id:
            return match.group(2).lower()
    return None


def ensure_npu_backend() -> None:
    try:
        import torch_npu  # noqa: F401
    except Exception as exc:
        raise RuntimeError('NPU backend is not available. Please install torch_npu/Ascend PyTorch.') from exc


class NPU(Platform):

    @staticmethod
    def is_ipc_supported() -> bool:
        """Return whether HDK and CANN meet the NPU device-IPC requirement."""
        try:
            result = subprocess.run(
                ['npu-smi', 'info', '-t', 'board', '-i', '1'], capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError:
            visible_devices = (os.environ.get('ASCEND_VISIBLE_DEVICES')
                               or os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
            if not visible_devices:
                raise
            device_id = int(visible_devices.split(',')[0])
            try:
                result = subprocess.run(
                    ['npu-smi', 'info', '-t', 'board', '-i', str(device_id)],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except subprocess.CalledProcessError:
                result = subprocess.run(
                    ['npu-smi', 'info', '-t', 'board', '-i', str(device_id // 2)],
                    capture_output=True,
                    text=True,
                    check=True,
                )

        software_version = next(
            (line.split(':', 1)[1].strip().lower()
             for line in result.stdout.splitlines() if 'Software Version' in line),
            None,
        )
        if software_version is None:
            raise RuntimeError('Could not find Software Version in npu-smi output')

        ascend_home = os.environ.get('ASCEND_HOME_PATH', '/usr/local/Ascend/ascend-toolkit/latest')
        info_file = os.path.join(ascend_home, f'{platform.machine()}-linux', 'ascend_toolkit_install.info')
        with open(info_file) as info:
            cann_version = next(
                (line.split('=', 1)[1].strip().lower() for line in info if line.startswith('version=')),
                None,
            )
        if cann_version is None:
            raise RuntimeError('Could not find version in CANN toolkit info file')

        pattern = r'(\d+\.\d+(?=\.t))|(\d+\.\d+(?:\.(?:rc\d+|\d+))?)'
        software_match = re.match(pattern, software_version)
        cann_match = re.match(pattern, cann_version)
        if software_match is None or cann_match is None:
            raise RuntimeError(f'Invalid NPU versions: HDK={software_version}, CANN={cann_version}')
        software_base = software_match.group(1) or software_match.group(2)
        cann_base = cann_match.group(1) or cann_match.group(2)
        return (version.parse(software_base) >= version.parse('25.3.rc1')
                and version.parse(cann_base) >= version.parse('8.3.rc1'))

    @staticmethod
    def visible_device_env():
        # Ascend runtime uses ASCEND_RT_VISIBLE_DEVICES.
        return 'ASCEND_RT_VISIBLE_DEVICES'

    @staticmethod
    def device_prefix():
        return 'npu'

    @staticmethod
    def get_local_device(idx, **kwargs) -> str:
        return f'npu:{idx}'

    @staticmethod
    def device_backend(platform: str = None):
        return 'hccl'

    @staticmethod
    def get_vllm_device_uuid(device_id: int = 0) -> str:
        from vllm.platforms import current_platform
        try:
            return current_platform.get_device_uuid(device_id)
        except NotImplementedError:
            # fix: Root cause was NPU platform calling vLLM base placeholder and raising NotImplementedError.
            # fix: Use Bus-Id fallback first so sender/receiver compute the same IPC endpoint.
            # NPU special case: prefer stable PCIe Bus-Id from npu-smi.
            bus_id = _get_npu_bus_id_from_npu_smi(device_id)
            if bus_id:
                return bus_id
            # fix: If npu-smi is unavailable, fall back to deterministic hash instead of failing hard.
            # Generic deterministic fallback to keep sender/receiver socket names aligned.
            visible = os.environ.get(Platform.visible_device_env())
            raw = f'{socket.gethostname()}:{visible}:{device_id}'
            return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    @staticmethod
    def get_device_rng_state():
        import torch
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return None
        if hasattr(torch, 'npu') and torch.npu.is_available():
            return torch.npu.get_rng_state()
        return None

    @staticmethod
    def set_device_rng_state(state) -> None:
        import torch
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return
        if hasattr(torch, 'npu') and torch.npu.is_available():
            torch.npu.set_rng_state(state)
