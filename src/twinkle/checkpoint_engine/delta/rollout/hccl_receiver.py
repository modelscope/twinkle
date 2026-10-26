# Copyright (c) ModelScope Contributors. All rights reserved.
"""Receiver-side HCCL delta transport (runs inside each vLLM worker process).

Import-safe without ``torch_npu`` / ``vllm_ascend``: the communicator class is imported lazily by
``stateless_init_process_group`` and device modules are looked up by name at call time.

The SUB socket MUST be connected before the communicator is built: the trainer builds its
communicator lazily at its first publish, which only completes once every worker joined, and a
zmq PUB drops messages sent before a subscriber connected.
"""
from __future__ import annotations

import json
import os
import torch
from collections.abc import Callable

from twinkle.utils import stateless_init_process_group

_REQUIRED_KEYS = ('is_last', )
_REQUIRED_FLUSH_KEYS = ('encoding', 'val_numel', 'val_dtype', 'spec')


def build_delta_hccl_communicator(dist_ip: str, dist_port: int, group_rank: int, world_size: int, device,
                                  hccl_env: dict[str, str] | None = None):
    """Join the trainer's HCCL broadcast group at ``group_rank``.

    ``hccl_env`` is trainer rank 0's HCCL socket env. It overrides this process's values: the trainer may
    have derived them from its FSDP ``MASTER_PORT``, and ranks with different port layouts time out in
    the communicator's first collective.
    """
    if hccl_env:
        os.environ.update(hccl_env)
    if isinstance(device, str):
        device = torch.device(device)
    index = device.index
    if index is None:
        index = getattr(torch, device.type).current_device()
    return stateless_init_process_group(
        master_address=dist_ip,
        master_port=dist_port,
        rank=group_rank,
        world_size=world_size,
        device=index,
        backend='hccl',
    )


def _validate_manifest(meta) -> None:
    if not isinstance(meta, dict):
        raise ValueError(f'Malformed delta manifest: expected dict, got {type(meta).__name__}')
    if meta.get('terminal_empty'):
        return
    missing = [k for k in _REQUIRED_KEYS + _REQUIRED_FLUSH_KEYS if k not in meta]
    if meta.get('encoding') != 'dense' and 'pos_numel' not in meta:
        missing.append('pos_numel')
    if missing:
        raise ValueError(f'Malformed delta manifest: missing keys {sorted(missing)}')
    if not isinstance(getattr(torch, str(meta['val_dtype']), None), torch.dtype):
        raise ValueError(f"Malformed delta manifest: unknown val_dtype {meta['val_dtype']!r}")


def pump_hccl_delta_session(
    sub,
    comm,
    device,
    on_flush: Callable[[list[tuple[str, torch.Tensor]]], None],
) -> str:
    """Pull one delta session (manifests + broadcasts) and hand each flush to ``on_flush``.

    Returns the session encoding (``'dense'`` | ``'indices'``); a terminal-only session returns
    ``'indices'`` with zero ``on_flush`` calls. Broadcast order mirrors the trainer: positions
    (non-dense only) then values.
    """
    synchronize = getattr(torch, device.type).synchronize
    encoding = 'indices'
    while True:
        sub.recv_string()
        meta = sub.recv_pyobj()
        _validate_manifest(meta)
        if meta.get('terminal_empty'):
            break
        encoding = meta['encoding']
        dense = encoding == 'dense'
        val_dtype = getattr(torch, meta['val_dtype'])
        elem = torch.empty(0, dtype=val_dtype).element_size()
        val_u8 = torch.empty(meta['val_numel'] * elem, dtype=torch.uint8, device=device)
        pos = None
        if not dense:
            pos = torch.empty(meta['pos_numel'], dtype=torch.uint8, device=device)
            comm.broadcast(pos, src=0)
        comm.broadcast(val_u8, src=0)
        synchronize()
        val = val_u8.view(val_dtype)
        spec_t = torch.frombuffer(bytearray(json.dumps(meta['spec']).encode()), dtype=torch.uint8).to(device)
        named = [('__delta_spec__', spec_t), ('__values__', val)]
        if pos is not None:
            named.insert(1, ('__positions__', pos))
        on_flush(named)
        del pos, val_u8, val, spec_t, named
        if meta['is_last']:
            break
    return encoding
