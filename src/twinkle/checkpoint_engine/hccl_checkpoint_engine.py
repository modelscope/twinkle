# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/hccl_checkpoint_engine.py
"""HCCL-based checkpoint engine for Ascend NPU.

This implementation keeps HCCL for weight payload transfer and uses a
reliable ZMQ REQ/REP control channel for bucket metadata handshakes.
"""

from __future__ import annotations

import os
import time
import torch
import zmq
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Generator

from twinkle import get_logger
from twinkle.utils import find_free_port, find_node_ip, is_valid_ipv6_address, stateless_init_process_group
from twinkle.utils.zmq_utils import configure_zmq_socket
from .base import CheckpointEngine

logger = get_logger()


@dataclass
class MasterMetadata:
    """Metadata from the master for process group initialization."""

    zmq_ip: str
    zmq_port: int
    dist_ip: str
    dist_port: int


class HCCLCheckpointEngine(CheckpointEngine):
    """HCCL checkpoint engine for Ascend NPU."""

    def __init__(
        self,
        bucket_size: int = 3072 << 20,
        group_name: str = 'twinkle_ckpt',
        rebuild_group: bool = True,
        rollout_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ) -> None:
        self.bucket_size = bucket_size
        self.group_name = group_name
        self.rebuild_group = rebuild_group
        self.rollout_dtype = rollout_dtype
        self.pyhccl = None

        self.meta_timeout_s = int(os.environ.get('TWINKLE_CKPT_HCCL_META_TIMEOUT_S', '300'))
        self.meta_timeout_ms = self.meta_timeout_s * 1000

        try:
            self.device = torch.npu.current_device()
        except Exception:
            self.device = 0

        self.is_master = False

        self.rank: int | None = None
        self.world_size: int | None = None
        self.send_buf: torch.Tensor | None = None
        self.recv_buf: torch.Tensor | None = None
        self.socket: zmq.Socket | None = None
        self._zmq_ctx: zmq.Context | None = None

        self._prepared = False
        self._group_initialized = False
        self.ip: str | None = None
        self.zmq_port: int | None = None
        self.dist_port: int | None = None

    def _new_socket(self, socket_type: int) -> zmq.Socket:
        assert self._zmq_ctx is not None
        socket = self._zmq_ctx.socket(socket_type)
        configure_zmq_socket(socket, timeout_ms=self.meta_timeout_ms, linger=0)
        return socket

    def _recv_pyobj(self, where: str) -> Any:
        assert self.socket is not None
        try:
            return self.socket.recv_pyobj()
        except zmq.error.Again as e:
            raise RuntimeError(f'HCCL metadata timeout ({self.meta_timeout_s}s) waiting at {where}.') from e

    def _send_pyobj(self, payload: Any, where: str) -> None:
        assert self.socket is not None
        try:
            self.socket.send_pyobj(payload)
        except zmq.error.Again as e:
            raise RuntimeError(f'HCCL metadata timeout ({self.meta_timeout_s}s) sending at {where}.') from e

    def _start_zmq_server(self):
        self.ip = find_node_ip()
        self.zmq_port = find_free_port()
        self.dist_port = find_free_port()

        self._zmq_ctx = zmq.Context()
        self.socket = self._new_socket(zmq.REP)
        if is_valid_ipv6_address(self.ip):
            address = f'tcp://[{self.ip}]:{self.zmq_port}'
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f'tcp://{self.ip}:{self.zmq_port}'
        self.socket.bind(address)
        logger.debug(f'ZMQ REP server started at {address}')

    def _connect_zmq_client(self, metadata: MasterMetadata):
        self._zmq_ctx = zmq.Context()
        self.socket = self._new_socket(zmq.REQ)
        if is_valid_ipv6_address(metadata.zmq_ip):
            address = f'tcp://[{metadata.zmq_ip}]:{metadata.zmq_port}'
            self.socket.setsockopt(zmq.IPV6, 1)
        else:
            address = f'tcp://{metadata.zmq_ip}:{metadata.zmq_port}'
        self.socket.connect(address)
        logger.debug(f'ZMQ REQ client connected to {address}')

    def prepare(self) -> MasterMetadata | None:
        if self._prepared:
            if self.is_master:
                return MasterMetadata(
                    zmq_ip=self.ip,
                    zmq_port=self.zmq_port,
                    dist_ip=self.ip,
                    dist_port=self.dist_port,
                )
            return None

        self.send_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device='npu')
        self.recv_buf = torch.zeros(self.bucket_size, dtype=torch.uint8, device='npu')

        if self.is_master:
            self._start_zmq_server()
            self._prepared = True
            return MasterMetadata(
                zmq_ip=self.ip,
                zmq_port=self.zmq_port,
                dist_ip=self.ip,
                dist_port=self.dist_port,
            )

        self._prepared = True
        return None

    def finalize(self):
        if self.rebuild_group:
            if self.socket is not None:
                try:
                    self.socket.close()
                except Exception as e:
                    logger.warning(f'Error closing ZMQ socket: {e}')
                self.socket = None

            if self._zmq_ctx is not None:
                try:
                    self._zmq_ctx.term()
                except Exception as e:
                    logger.warning(f'Error terminating ZMQ context: {e}')
                self._zmq_ctx = None

            if self.rank is not None and self.rank >= 0 and self.pyhccl is not None:
                try:
                    self.pyhccl.destroyComm(self.pyhccl.comm)
                except Exception:
                    pass
                self.pyhccl = None

            self.rank = None
            self.world_size = None
            self.send_buf = None
            self.recv_buf = None
            self._prepared = False
            self._group_initialized = False

    @classmethod
    def build_topology(
        cls,
        trainer_world_size: int,
        rollout_world_size: int,
        metadata: list[dict],
    ) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
        master_metadata = None
        for m in metadata:
            if m is not None:
                master_metadata = m
                break

        trainer_kwargs = {
            'rank': [0] + [-1] * (trainer_world_size - 1),
            'world_size': [rollout_world_size + 1] * trainer_world_size,
            'master_metadata': [master_metadata] * trainer_world_size,
        }
        rollout_kwargs = {
            'rank': list(range(1, rollout_world_size + 1)),
            'world_size': [rollout_world_size + 1] * rollout_world_size,
            'master_metadata': [master_metadata] * rollout_world_size,
        }
        return trainer_kwargs, rollout_kwargs

    def init_process_group(self, rank: int, world_size: int, master_metadata: MasterMetadata):
        if rank < 0:
            self.rank = rank
            self.world_size = world_size
            self._group_initialized = True
            return

        if self._group_initialized and not self.rebuild_group:
            return

        if self.rebuild_group or self.pyhccl is None:
            self.pyhccl = stateless_init_process_group(
                master_address=master_metadata.dist_ip,
                master_port=master_metadata.dist_port,
                rank=rank,
                world_size=world_size,
                device=self.device,
                backend='hccl',
            )
            self.rank = rank
            self.world_size = world_size
        else:
            assert self.rank == rank
            assert self.world_size == world_size

        if self.rank > 0 and self.socket is None:
            self._connect_zmq_client(master_metadata)

        signal = torch.tensor([1], dtype=torch.int8, device=torch.npu.current_device())
        self.pyhccl.all_reduce(signal)

        self._group_initialized = True
        logger.info(f'init_process_group: rank={self.rank}, world_size={self.world_size}')

    def _serve_bucket_requests(self, bucket_id: int, metadata: dict[str, Any]) -> None:
        assert self.rank == 0
        assert self.world_size is not None

        if self.world_size <= 1:
            return

        pending = set(range(1, self.world_size))
        while pending:
            req = self._recv_pyobj(f'NEXT request for bucket={bucket_id}')

            if not isinstance(req, dict) or req.get('type') != 'NEXT':
                self._send_pyobj({'ok': False, 'error': f'unexpected message: {req}'}, 'NEXT reply')
                continue

            req_rank = int(req.get('rank', -1))
            req_bucket_id = int(req.get('bucket_id', -1))

            if req_rank not in pending:
                self._send_pyobj(
                    {
                        'ok': False,
                        'error': f'unexpected/duplicate rank={req_rank}'
                    },
                    'NEXT reply',
                )
                continue
            if req_bucket_id != bucket_id:
                self._send_pyobj(
                    {
                        'ok': False,
                        'error': f'bucket mismatch rank={req_rank} got={req_bucket_id} expected={bucket_id}',
                    },
                    'NEXT reply',
                )
                continue

            self._send_pyobj({'ok': True, 'metadata': metadata}, 'NEXT reply')
            pending.remove(req_rank)

    def _request_bucket(self, bucket_id: int) -> dict[str, Any]:
        assert self.rank is not None and self.rank > 0

        self._send_pyobj(
            {
                'type': 'NEXT',
                'rank': self.rank,
                'bucket_id': bucket_id
            },
            f'NEXT send bucket={bucket_id}',
        )
        resp = self._recv_pyobj(f'NEXT recv bucket={bucket_id}')

        if not isinstance(resp, dict):
            raise RuntimeError(f'Invalid metadata response for bucket {bucket_id}: {resp}')
        if not resp.get('ok', False):
            raise RuntimeError(f'Metadata request failed for bucket {bucket_id}: {resp.get("error", "unknown")}')
        metadata = resp.get('metadata')
        if not isinstance(metadata, dict):
            raise RuntimeError(f'Invalid metadata payload for bucket {bucket_id}: {metadata}')
        got_bucket_id = int(metadata.get('bucket_id', -1))
        if got_bucket_id != bucket_id:
            raise RuntimeError(f'Metadata bucket mismatch: got {got_bucket_id}, expected {bucket_id}')
        return metadata

    @torch.no_grad()
    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor]]):
        assert self.rank is not None and self.rank <= 0
        if self.rank < 0:
            for _name, _weight in weights:
                pass
            return

        assert self.send_buf is not None

        send_buf = self.send_buf
        start_time = time.time()
        bucket_meta: list[dict[str, Any]] = []
        offset = 0
        bucket_id = 0
        total_params = 0
        total_chunks = 0
        total_bytes = 0

        def _flush(is_last: bool):
            nonlocal bucket_meta, offset, bucket_id, total_chunks, total_bytes
            if not bucket_meta and not is_last:
                return

            metadata = {
                'bucket_id': bucket_id,
                'is_last': is_last,
                'bucket_meta': bucket_meta,
                'payload_size': offset,
            }
            self._serve_bucket_requests(bucket_id, metadata)
            self.pyhccl.broadcast(send_buf, src=0)
            torch.npu.synchronize()

            total_chunks += len(bucket_meta)
            total_bytes += offset
            bucket_id += 1
            bucket_meta = []
            offset = 0

        for name, weight in weights:
            total_params += 1
            if weight.device.type != 'npu':
                weight = weight.to('npu')
            if not weight.is_contiguous():
                weight = weight.contiguous()

            weight_u8 = weight.view(-1).view(torch.uint8)
            nbytes = int(weight_u8.numel())
            if nbytes == 0:
                if offset >= self.bucket_size:
                    _flush(is_last=False)
                bucket_meta.append({
                    'name': name,
                    'shape': weight.shape,
                    'dtype': weight.dtype,
                    'offset': offset,
                    'nbytes': 0,
                    'chunk_offset': 0,
                    'total_nbytes': 0,
                })
                continue

            chunk_offset = 0
            while chunk_offset < nbytes:
                if offset >= self.bucket_size:
                    _flush(is_last=False)

                chunk_nbytes = min(self.bucket_size - offset, nbytes - chunk_offset)
                send_buf[offset:offset + chunk_nbytes].copy_(weight_u8[chunk_offset:chunk_offset + chunk_nbytes])
                bucket_meta.append({
                    'name': name,
                    'shape': weight.shape,
                    'dtype': weight.dtype,
                    'offset': offset,
                    'nbytes': chunk_nbytes,
                    'chunk_offset': chunk_offset,
                    'total_nbytes': nbytes,
                })
                offset += chunk_nbytes
                chunk_offset += chunk_nbytes

        _flush(is_last=True)

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024) if elapsed > 0 else 0.0
        logger.info(f'send_weights done: rank={self.rank}, params={total_params}, chunks={total_chunks}, '
                    f'time={elapsed:.2f}s, bandwidth={bandwidth:.2f} GB/s')

    @torch.no_grad()
    async def receive_weights(self) -> AsyncGenerator[tuple[str, torch.Tensor]]:
        assert self.rank is not None and self.rank > 0
        assert self.recv_buf is not None

        recv_buf = self.recv_buf
        bucket_id = 0
        total_params = 0
        total_chunks = 0
        total_bytes = 0
        start_time = time.time()
        partial_tensors: dict[str, dict[str, Any]] = {}

        while True:
            metadata = self._request_bucket(bucket_id)
            self.pyhccl.broadcast(recv_buf, src=0)
            torch.npu.synchronize()

            bucket_meta = metadata['bucket_meta']
            if isinstance(bucket_meta, dict):
                entries = bucket_meta.values()
            else:
                entries = bucket_meta

            payload_size = int(metadata.get('payload_size', self.bucket_size))
            total_bytes += payload_size

            for meta in entries:
                name = meta['name']
                dtype = meta['dtype']
                shape = meta['shape']
                shape = shape if isinstance(shape, torch.Size) else torch.Size(shape)
                offset = int(meta['offset'])
                nbytes = int(meta.get('nbytes', int(dtype.itemsize * shape.numel())))
                chunk_offset = int(meta.get('chunk_offset', 0))
                total_nbytes = int(meta.get('total_nbytes', int(dtype.itemsize * shape.numel())))
                total_chunks += 1

                if nbytes == total_nbytes and chunk_offset == 0:
                    tensor = recv_buf[offset:offset + nbytes].view(dtype=dtype).view(shape)
                    yield name, tensor
                    total_params += 1
                    continue

                state = partial_tensors.get(name)
                if state is None:
                    state = {
                        'buffer': torch.empty(total_nbytes, dtype=torch.uint8, device=recv_buf.device),
                        'dtype': dtype,
                        'shape': shape,
                        'total': total_nbytes,
                        'received': 0,
                    }
                    partial_tensors[name] = state
                else:
                    if state['total'] != total_nbytes or state['dtype'] != dtype or state['shape'] != shape:
                        raise RuntimeError(
                            f'Inconsistent chunk metadata for weight {name}: '
                            f'expected total={state["total"]}, dtype={state["dtype"]}, shape={state["shape"]}; '
                            f'got total={total_nbytes}, dtype={dtype}, shape={shape}.')

                if nbytes > 0:
                    state['buffer'][chunk_offset:chunk_offset + nbytes].copy_(recv_buf[offset:offset + nbytes])
                state['received'] += nbytes

                if state['received'] > state['total']:
                    raise RuntimeError(
                        f'Chunk overrun for weight {name}: received={state["received"]}, total={state["total"]}.')
                if state['received'] == state['total']:
                    full_size = int(dtype.itemsize * shape.numel())
                    tensor = state['buffer'][:full_size].view(dtype=dtype).view(shape)
                    yield name, tensor
                    total_params += 1
                    del partial_tensors[name]

            if bool(metadata['is_last']):
                if partial_tensors:
                    pending = ', '.join(sorted(partial_tensors.keys())[:8])
                    raise RuntimeError('Incomplete chunked weights at end of stream. '
                                       f'Pending {len(partial_tensors)} weight(s): {pending}')
                break
            bucket_id += 1

        elapsed = time.time() - start_time
        bandwidth = total_bytes / elapsed / (1024 * 1024 * 1024) if elapsed > 0 else 0.0
        logger.info(f'receive_weights done: rank={self.rank}, params={total_params}, chunks={total_chunks}, '
                    f'time={elapsed:.2f}s, bandwidth={bandwidth:.2f} GB/s')
