# Copyright (c) ModelScope Contributors. All rights reserved.
"""Rollout-side lifecycle for the delta checkpoint engine.

CUDA: Ray receiver actors (:mod:`.cuda_receiver`) colocated with each vLLM worker feed it over IPC. Ascend NPU:
no receiver actors; the vLLM workers join the trainer's HCCL group themselves (:mod:`.hccl_receiver`) and the
sampler only drives vLLM's lifecycle RPCs.
"""
from __future__ import annotations

import asyncio
from typing import Any

from twinkle import Platform, alive_node_ids, kill_actors, remote_function, resolve_all, spawn_sidecar
from ..config import (DeltaReceiverPlacement, DeltaSyncConfig, DeltaTransportEndpoint,
                     DeltaWorkerCapabilities, DeltaWorkerMetadata)
from .cuda_receiver import CudaDeltaReceiver


class DeltaSamplerMixin:
    """Sampler-side delta lifecycle RPCs (device-aware: CUDA uses receiver actors, NPU does not)."""
    _delta_config: DeltaSyncConfig
    _delta_job_failed: bool = False
    _delta_weight_transfer_initialized: bool = False
    _delta_receivers: list[Any]
    _delta_device: str | None = None  # 'cuda' / 'npu', remembered by validate_delta_sampler
    _delta_tp: int | None = None  # NPU: TP size of this replica
    _delta_replica_rank: int | None = None  # NPU: this actor's data-parallel replica index
    _delta_endpoint: DeltaTransportEndpoint | None = None  # NPU: HCCL rendezvous stored by init_delta_sampler_transport

    @property
    def _delta_is_npu(self) -> bool:
        return self._delta_device == 'npu'

    @staticmethod
    def _delta_actor_replica_rank() -> int:
        """Replica index of this sampler actor.

        ``RayHelper.create_workers`` starts actor ``i`` of ``sampler._actors`` with ``RANK=i`` and
        ``WORLD_SIZE=len(_actors)``, and ``dispatch='all'`` results are collected in ``_actors`` order, so
        ``Platform.get_rank()`` is the actor's position in the manager's replica list. ``DeviceMesh`` lookups
        are not used: they index by GPU id and are wrong when an actor owns several GPUs (TP).
        """
        rank = Platform.get_rank()
        world_size = Platform.get_world_size()
        if rank < 0 or rank >= world_size:
            raise RuntimeError(
                f'cannot determine the delta replica rank of this sampler actor (RANK={rank}, WORLD_SIZE={world_size}); '
                'NPU delta sync requires sampler actors created by Ray with a per-actor RANK')
        return rank

    async def _delta_rpc(self, method: str, kwargs: dict[str, Any] | None = None) -> list[Any]:
        return await self.engine.engine.collective_rpc(
            method, kwargs=kwargs or {}, timeout=self._delta_config.timeout_seconds)

    async def _reset_delta_prefix_cache(self) -> None:
        reset = await self.engine.engine.reset_prefix_cache(reset_connector=True)
        if reset is False:
            raise RuntimeError('prefix cache reset failed during delta synchronization')

    async def _clear_delta_cache(self) -> None:
        await self._reset_delta_prefix_cache()
        await self.engine.engine.reset_mm_cache()
        await self.engine.engine.reset_encoder_cache()

    @remote_function(dispatch='all', lazy_collect=False)
    def begin_delta_sync(self) -> None:
        async def begin():
            self.engine._delta_unavailable = True
            await self.engine.pause_delta_generation(self._delta_config.timeout_seconds)
            if self._delta_config.free_cache_engine:
                # Level 1 preserves the baseline needed by sparse updates.
                await self.engine.engine.sleep(level=1)
                await self.engine.engine.wake_up(tags=['weights'])
        self._run_in_loop(begin())

    @remote_function(dispatch='all', lazy_collect=False)
    def publish_delta_sampler(self) -> None:
        async def publish():
            if self._delta_config.free_cache_engine:
                await self.engine.engine.wake_up(tags=['kv_cache'])
                await self._reset_delta_prefix_cache()
            await self.engine.engine.resume_generation()
            self.engine._delta_unavailable = False
        self._run_in_loop(publish())

    @remote_function(dispatch='all', lazy_collect=False)
    def fail_delta_sampler(self) -> None:
        self._delta_job_failed = True
        self.engine._delta_unavailable = True
        self._run_in_loop(self.engine.pause_delta_generation(self._delta_config.timeout_seconds))

    @remote_function(dispatch='all', lazy_collect=False)
    def validate_delta_sampler(self, config: DeltaSyncConfig) -> DeltaWorkerCapabilities:
        self._delta_config = config
        if self._delta_job_failed:
            raise RuntimeError('delta sampler failed; restart the entire training/rollout job before reseeding')
        if config.free_cache_engine and not self.engine.enable_sleep_mode:
            raise ValueError('delta KV cache release requires enable_sleep_mode=True or free_cache_engine=False')
        results = self._run_in_loop(self._delta_rpc('validate_delta_worker'))
        if not results or any(result != results[0] for result in results):
            raise ValueError('inconsistent delta worker capabilities')
        if len(results) != results[0]['tp']:
            raise ValueError('delta topology requires one vLLM worker per TP rank')
        self._delta_device = results[0]['device']
        if self._delta_device == 'npu':
            self._delta_tp = results[0]['tp']
            self._delta_replica_rank = self._delta_actor_replica_rank()
            return dict(results[0], replica_rank=self._delta_replica_rank)
        return results[0]

    @remote_function(dispatch='all', collect='flatten', lazy_collect=False)
    def delta_sampler_metadata(self) -> list[DeltaWorkerMetadata]:
        return self._run_in_loop(self._delta_rpc('delta_receiver_metadata'))

    def _delta_receiver_placements(
        self, metadata: list[DeltaWorkerMetadata], endpoint: DeltaTransportEndpoint,
    ) -> list[DeltaReceiverPlacement]:
        workers = endpoint['rollout_workers']
        ranks = {(item['node_ip'], item['device_uuid']): rank for rank, item in enumerate(workers, start=1)}
        if not metadata or any(item not in workers for item in metadata):
            raise ValueError('local vLLM workers are missing from the delta topology')
        nodes = alive_node_ids()
        return [dict(rank=ranks[item['node_ip'], item['device_uuid']], node_id=nodes[item['node_ip']],
                     device_uuid=item['device_uuid'],
                     cuda_visible_device=item.get('cuda_visible_device', item['device_uuid']),
                     zmq_handle=item['zmq_handle'])
                for item in metadata]

    @remote_function(dispatch='all', lazy_collect=True)
    def init_delta_sampler_transport(self, endpoint: DeltaTransportEndpoint) -> None:
        if self._delta_is_npu:
            # No receiver actors: workers join the trainer's HCCL group on their first receive.
            self._delta_endpoint = endpoint
            self._delta_weight_transfer_initialized = False
            return
        metadata = self._run_in_loop(self._delta_rpc('delta_receiver_metadata'))
        placements = self._delta_receiver_placements(metadata, endpoint)
        # One zero-resource sidecar per rollout device, pinned to its node with the device's
        # CUDA context; the handle is driven below with plain .method.remote() calls.
        self._delta_receivers = [
            spawn_sidecar(
                CudaDeltaReceiver, node_id=placement['node_id'],
                env_vars={'CUDA_VISIBLE_DEVICES': placement.get('cuda_visible_device', placement['device_uuid']),
                          'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES': '1'},
                args=(self._delta_config, endpoint, placement))
            for placement in placements]
        resolve_all([receiver.ready.remote() for receiver in self._delta_receivers],
                    timeout=self._delta_config.timeout_seconds)
        self._delta_weight_transfer_initialized = False

    async def _next_delta_flush(self) -> dict[str, bool] | None:
        flags = await asyncio.gather(*(receiver.next_flush.remote() for receiver in self._delta_receivers))
        if any(flag != flags[0] for flag in flags):
            raise ValueError('delta checkpoint receivers disagree about stream completion')
        return flags[0]

    async def _send_delta_flush(self) -> None:
        receivers = asyncio.create_task(self._delta_rpc('update_twinkle_delta_weights'))
        try:
            await asyncio.gather(*(receiver.send_flush.remote() for receiver in self._delta_receivers))
            await receivers
        finally:
            if not receivers.done():
                receivers.cancel()
            await asyncio.gather(receivers, return_exceptions=True)

    async def _receive_delta_weights_npu(self) -> dict[str, bool]:
        """One NPU round: the vLLM workers pump the whole HCCL session inside ``update_twinkle_delta_weights``.

        That RPC blocks until the session is received and applied, and runs concurrently with the trainer's
        send (the manager launches both before waiting). A zero-change round cannot be told apart here (the
        worker pump just returns on the terminal manifest), so it is reported as changed; the cache clear is
        harmless.
        """
        endpoint = self._delta_endpoint
        if endpoint is None:
            raise RuntimeError('delta NPU transport is not initialized; run a full seed first')
        if not self._delta_weight_transfer_initialized:
            master = endpoint['master_metadata']
            init_info = {
                'dist_ip': master.dist_ip, 'dist_port': master.dist_port,
                'zmq_ip': master.zmq_ip, 'zmq_port': master.zmq_port,
                'topic': 'bucket_metadata', 'world_size': endpoint['world_size']}
            if endpoint.get('hccl_env'):
                init_info['hccl_env'] = dict(endpoint['hccl_env'])
            await self._delta_rpc('init_weight_transfer_engine', {'init_info': init_info})
            self._delta_weight_transfer_initialized = True
        await self._delta_rpc('start_weight_update')
        # Group rank 0 is the trainer; workers are numbered replica-major, TP-minor.
        base = 1 + self._delta_replica_rank * self._delta_tp
        await self._delta_rpc('update_twinkle_delta_weights', {
            'update_info': [{'group_rank': base + index} for index in range(self._delta_tp)]})
        await self._delta_rpc('finish_weight_update')
        await self._clear_delta_cache()
        return {'changed': True}

    @remote_function(dispatch='all', lazy_collect=True)
    def receive_delta_weights(self) -> dict[str, bool]:
        if self._delta_is_npu:
            return self._run_in_loop(self._receive_delta_weights_npu())

        async def receive():
            first = await self._next_delta_flush()
            if first is None:
                return {'changed': False}
            if not self._delta_weight_transfer_initialized:
                await self._delta_rpc('init_weight_transfer_engine', {'init_info': {}})
                self._delta_weight_transfer_initialized = True
            await self._delta_rpc('start_weight_update')

            saw_last = first['is_last']
            await self._send_delta_flush()
            following = await self._next_delta_flush()
            while following is not None:
                if saw_last:
                    raise ValueError('DeltaFlush stream yielded data after is_last=True')
                saw_last = following['is_last']
                await self._send_delta_flush()
                following = await self._next_delta_flush()
            if not saw_last:
                raise ValueError('DeltaFlush stream ended without is_last=True')
            await self._delta_rpc('finish_weight_update')
            await self._clear_delta_cache()
            return {'changed': True}
        return self._run_in_loop(receive())

    @remote_function(dispatch='all', lazy_collect=False)
    def close_delta_sampler(self) -> None:
        self.engine._delta_unavailable = True
        self._run_in_loop(self.engine.pause_delta_generation(self._delta_config.timeout_seconds))
        if self._delta_is_npu:
            try:
                self._run_in_loop(self._delta_rpc('shutdown_delta_weight_transfer'))
            finally:
                self._delta_weight_transfer_initialized = False
                self._delta_endpoint = None
            return
        receivers = getattr(self, '_delta_receivers', [])
        try:
            resolve_all([receiver.close.remote() for receiver in receivers],
                        timeout=self._delta_config.timeout_seconds)
        finally:
            kill_actors(receivers)
            self._delta_receivers = []
