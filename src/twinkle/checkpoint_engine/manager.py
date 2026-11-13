# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/base.py
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Any, Dict, List, Literal, Optional

from twinkle import Platform, get_logger, remote_class, remote_function
from .base import CheckpointEngine
from .delta import DeltaSyncConfig, resolve_delta_engine_cls
from .mixin import CheckpointEngineMixin
from .zmq_auth import generate_auth_key

logger = get_logger()

CheckpointEngineMode = Literal['auto', 'naive', 'colocate', 'standalone']
_VALID_MODES = {'auto', 'naive', 'colocate', 'standalone'}


def _wait(result: Any) -> Any:
    """The value of a lazily collected call (its handle is called), or the value itself."""
    return result() if callable(result) else result


def _together(*calls) -> List[Any]:
    """Run the calls side by side and wait for all of them.

    Rendezvous pairs (a process-group init whose master blocks until every peer connects, a send
    and its receive) must not run one after the other; the first failure is raised after every call
    has ended.
    """
    with ThreadPoolExecutor(len(calls)) as pool:
        futures = [pool.submit(lambda call=call: _wait(call())) for call in calls]
    results, error = [], None
    for future in futures:
        try:
            results.append(future.result())
        except BaseException as failure:  # noqa
            error = error or failure
            results.append(None)
    if error is not None:
        raise error
    return results


@remote_class(execute='first', max_concurrency=3)
class CheckpointEngineManager:
    """Weight synchronization manager for local and Ray deployments; one per model/sampler pair.

    ``mode`` selects one of three synchronization paths:

    * ``naive`` streams a local model's weight generator directly into a local sampler.
    * ``colocate`` connects Ray model and sampler actors sharing GPUs through CUDA IPC.
    * ``standalone`` connects disaggregated Ray actors through NCCL/HCCL.

    ``auto`` resolves local objects to ``naive`` and Ray actor handlers to ``standalone``. It never
    guesses ``colocate`` because actor placement cannot be inferred reliably from the driver.

    The manager owns the policy version: every completed :meth:`sync_weights` adds one, and
    :meth:`version` reads it without waiting for a sync in progress. Built with a ``remote_group``
    it is a single actor every process shares (the training loop syncs, the streaming driver reads
    the version); syncs and :meth:`warm` run one at a time.

    The receivers are the sampler's current replicas. A standalone sync holds their generation
    (``pause_generation``) for the transfer and releases it after, so running requests continue on
    the new weights; when the replica set changed since the last sync the transport is rebuilt for
    the new set. New replicas get the current weights from :meth:`warm` before they serve.

    Architecture (following verl's CheckpointEngineManager):

        Trainer GPU(s)                          Rollout GPU(s)
        ┌──────────────────┐                    ┌──────────────────┐
        │ TransformersModel│                    │   vLLMSampler    │
        │  (Ray actors)    │                    │  (Ray actors)    │
        │        │         │                    │        │         │
        │        ▼         │                    │        ▼         │
        │ CheckpointEngine │ NCCL/HCCL/CUDA IPC │ CheckpointEngine │
        │  send_weights()  │ ─────────────────► │ receive_weights()│
        │                  │                    │        │         │
        │                  │                    │        ▼         │
        │                  │                    │ VLLMEngine       │
        │                  │                    │  update_weights()│
        └──────────────────┘                    └──────────────────┘

    Usage:
        >>> manager = CheckpointEngineManager(model=model, sampler=sampler)
        >>> manager.sync_weights()  # Call after each training step

    With colocated Ray actors the sync also moves the device between the two roles, in the order
    the training loop needs: the sampler's weights are woken (``sleep_level`` > 0), written, the
    trainer offloads (``offload_kwargs`` go to ``model.offload_to_cpu``), and the sampler's KV cache
    is woken, ready to generate. The training loop does the other half after it takes a batch:
    ``sampler.sleep(level)`` and ``model.reload_to_gpu()``.
    """

    def __init__(
        self,
        model: 'CheckpointEngineMixin',
        sampler: 'CheckpointEngineMixin',
        platform: Optional[str] = None,
        mode: CheckpointEngineMode = 'auto',
        sync_backend: Literal['full', 'delta_sharded'] = 'full',
        delta_config: Optional[DeltaSyncConfig] = None,
        channel: str = 'default',
        sleep_level: int = 0,
        offload_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            model: The weight sender (a training model).
            sampler: The weight receiver (a sampler, or a reference model for :meth:`sync_ema`).
            platform: The device platform; picks the transport.
            mode: ``auto`` / ``naive`` / ``colocate`` / ``standalone``, see the class docstring.
            sync_backend: ``full`` sends every weight; ``delta_sharded`` sends only changed ones.
            delta_config: Settings of ``delta_sharded``.
            channel: Name of this manager's transport channel, separating managers that share actors.
            sleep_level: Colocate only: the level the training loop sleeps the sampler at; 0 means it
                never sleeps, so the sync wakes nothing.
            offload_kwargs: Colocate only: keyword arguments of ``model.offload_to_cpu`` after a sync.
        """
        self.model = model
        self.sampler = sampler
        self.requested_mode = mode
        self.mode = self._resolve_mode(mode, model, sampler)
        self.backend_cls = self.decide_backend_engine(platform, self.mode)
        if sync_backend not in ('full', 'delta_sharded'):
            raise ValueError(f'Unknown weight sync backend: {sync_backend}')
        if sync_backend == 'delta_sharded' and self.mode != 'standalone':
            raise ValueError('delta_sharded requires standalone trainer and rollout actors')
        self.sync_backend = sync_backend
        self.delta_config = delta_config or DeltaSyncConfig()
        self.delta_engine_cls = None
        self._delta_platform = None
        if sync_backend == 'delta_sharded':
            # Fail at construction, not mid-sync, when the platform has no delta transport.
            self._delta_platform = Platform.get_platform(platform).__name__
            self.delta_engine_cls = resolve_delta_engine_cls(self._delta_platform)
        self._delta_replica_count = None
        self._delta_round = 0
        self._delta_failed = False
        self.last_delta_metrics = None
        self._sync_lock = threading.RLock()
        self._version = 0
        if not isinstance(channel, str) or not channel:
            raise ValueError(f'checkpoint channel must be a non-empty string, got {channel!r}.')
        self.channel = channel
        self.sleep_level = int(sleep_level)
        self.offload_kwargs = dict(offload_kwargs or {})
        self.backend = None if self.mode == 'naive' else self.backend_cls.backend_name
        # Signs the engine's control frames. Generated here and delivered to both sides through the
        # Ray control path, so a peer that guessed the endpoint path still cannot inject frames.
        self._auth_key = generate_auth_key()
        # The receiver replicas the open channel was built for; None when no channel is open.
        self._members = None

        if self.mode != 'naive':
            try:
                self._members = self._activate_checkpoint_channel()
            except BaseException:
                try:
                    self._close_checkpoint_channel()
                except Exception:
                    logger.warning('Failed to fully close checkpoint channel %s after initialization failure.',
                                   channel, exc_info=True)
                raise

        # LoRA sync state: tracks whether the first full sync has been done.
        # After the first sync, only LoRA adapter weights are transferred.
        self.base_sync_done: bool = False
        self._merge_and_sync = True
        # Cached peft_config dict for LoRA-only sync.
        # Fetched lazily from the model on first LoRA sync.
        self._peft_config: dict | None = None
        self._model_keys: Optional[List[str]] = None

    @staticmethod
    def _resolve_mode(
        mode: CheckpointEngineMode,
        model: 'CheckpointEngineMixin',
        sampler: 'CheckpointEngineMixin',
    ) -> Literal['naive', 'colocate', 'standalone']:
        if mode not in _VALID_MODES:
            valid = ', '.join(sorted(_VALID_MODES))
            raise ValueError(f'Unknown checkpoint engine mode {mode!r}; expected one of: {valid}.')

        model_has_actors = bool(getattr(model, '_actors', None))
        sampler_has_actors = bool(getattr(sampler, '_actors', None))
        if model_has_actors != sampler_has_actors:
            raise ValueError(
                'CheckpointEngineManager requires model and sampler to use the same deployment shape: '
                'both must be local objects or both must be Ray actor handlers.')

        if mode == 'auto':
            return 'standalone' if model_has_actors else 'naive'
        if mode == 'naive' and model_has_actors:
            raise ValueError("mode='naive' requires local model and sampler objects without Ray actors.")
        if mode in ('colocate', 'standalone') and not model_has_actors:
            raise ValueError(f"mode={mode!r} requires model and sampler to be Ray actor handlers.")
        return mode

    @staticmethod
    def decide_backend_engine(
        platform: Optional[str] = None,
        mode: Literal['naive', 'colocate', 'standalone'] = 'standalone',
    ) -> Optional['CheckpointEngine']:
        if mode == 'naive':
            return None

        if mode == 'colocate':
            from twinkle.checkpoint_engine import IPCCheckpointEngine
            return IPCCheckpointEngine
        if mode != 'standalone':
            raise ValueError(f'Cannot select a backend for unresolved mode {mode!r}.')
        # The platform decides its own engine; a literal 'GPU' would mask NPU/XPU
        # because ``Platform.get_platform('GPU')`` always resolves to the NVIDIA
        # platform, so the driver would build an NCCL topology while the workers
        # instantiate the HCCL/XCCL engine.
        return Platform.get_platform(platform).checkpoint_engine()[0]

    # ── Channel ──────────────────────────────────────────────────────────────

    def _activate_checkpoint_channel(self, receiver: Any = None, channel: Optional[str] = None) -> tuple:
        """Select the channel on both sides; returns the ids of the receiver replicas it reached."""
        receiver = self.sampler if receiver is None else receiver
        channel = channel or self.channel
        self.model.activate_checkpoint_engine(self.backend, channel, auth_key=self._auth_key)
        members = receiver.activate_checkpoint_engine(self.backend, channel, auth_key=self._auth_key)
        return tuple(members if isinstance(members, list) else [members])

    def _close_checkpoint_channel(self, receiver: Any = None, channel: Optional[str] = None) -> None:
        receiver = self.sampler if receiver is None else receiver
        channel = channel or self.channel
        try:
            self.model.reset_checkpoint_engine(channel, self.backend)
        finally:
            receiver.reset_checkpoint_engine(channel, self.backend)

    @remote_function(collect='first', lazy_collect=False)
    def close(self) -> None:
        """Tear down this manager's channel; the next sync rebuilds it."""
        with self._sync_lock:
            if self.mode == 'naive' or self._members is None:
                return
            try:
                self._close_checkpoint_channel()
            finally:
                self._members = None
                self.base_sync_done = False

    # ── Version ──────────────────────────────────────────────────────────────

    @remote_function(collect='first', lazy_collect=False)
    def version(self) -> int:
        """How many syncs have completed; read without waiting for one in progress."""
        return self._version

    # ── Sync ─────────────────────────────────────────────────────────────────

    @remote_function(collect='first', lazy_collect=False)
    def sync_weights(self, merge_and_sync=True):
        """Push the model's current weights to every sampler replica, then advance the version.

        Args:
            merge_and_sync (bool, optional): Whether to merge and sync the weights.
                - If True: LoRA weights are merged into the base model, then the
                combined weights are synchronized to the sampler on every call.
                - If False: On the first call, base model weights are synced to the
                sampler. On subsequent calls, only the LoRA adapter weights are
                synced incrementally.
                Defaults to True.
        """
        with self._sync_lock:
            self._merge_and_sync = merge_and_sync
            if self.sync_backend == 'delta_sharded':
                # The delta session holds and releases generation itself (begin/publish).
                self._sync_delta_weights(merge_and_sync)
            elif self.mode == 'colocate':
                self._sync_colocate(merge_and_sync)
            else:
                self.sampler.pause_generation()
                try:
                    if self.mode == 'naive':
                        self._sync_weights_naive(merge_and_sync)
                    else:
                        self._sync_channel(merge_and_sync)
                finally:
                    self.sampler.resume_generation()
            self._version += 1

    def _wake_sampler(self, tags: List[str]) -> None:
        if self.sleep_level > 0:
            self.sampler.wake_up(tags=tags)

    def _sync_colocate(self, merge_and_sync):
        """Weights in, trainer out, KV cache in: the sampler is ready to generate when this returns."""
        self._wake_sampler(['weights'])
        if not self._sync_weights_direct_vllm(merge_and_sync):
            self._sync_channel(merge_and_sync)
        self.model.offload_to_cpu(**self.offload_kwargs)
        self._wake_sampler(['kv_cache'])

    def _sync_channel(self, merge_and_sync):
        """Checkpoint-engine sync over the manager's channel, rebuilt when the replicas changed."""
        try:
            members = self._activate_checkpoint_channel()
            if members != self._members:
                if self._members is not None:
                    logger.info('Sampler replicas changed (%s -> %s); rebuilding checkpoint channel %s.',
                                len(self._members), len(members), self.channel)
                    self._close_checkpoint_channel()
                    members = self._activate_checkpoint_channel()
                self._members = members
                self.base_sync_done = False
            self._transfer(self.sampler, len(self._members), merge_and_sync, self.base_sync_done)
        except BaseException:
            # A failed sync can leave the transport half-torn -- a NCCL group in error state, a
            # dead socket. Close here so the next attempt rebuilds the channel instead of reusing
            # the broken one.
            try:
                self.close()
            except Exception:
                # A failing close (e.g. the remote actor already died) must not replace the
                # original sync failure -- warn and re-raise the root cause.
                logger.warning('Failed to fully close checkpoint channel %s after sync failure.',
                               self.channel, exc_info=True)
            raise
        if not self.base_sync_done:
            self.base_sync_done = True
            if not merge_and_sync:
                logger.info('Base model sync completed, subsequent syncs will be LoRA-only')

    def _transfer(self, receiver, receiver_count: int, merge_and_sync: bool, base_sync_done: bool) -> None:
        """One checkpoint-engine round from the model to ``receiver``'s ``receiver_count`` replicas."""
        model_world_size = self.model.device_mesh.world_size
        model_metadata = self.model.prepare_checkpoint_engine([True] + [False] * (model_world_size - 1))
        receiver.prepare_checkpoint_engine(False)
        model_kwargs, sampler_kwargs = self.backend_cls.build_topology(
            model_world_size,
            receiver_count,
            [model_metadata],
        )
        # The TCPStore server (model rank 0) blocks until all clients (sampler ranks) connect.
        _together(lambda: self.model.init_checkpoint_process_group(**model_kwargs),
                  lambda: receiver.init_checkpoint_process_group(**sampler_kwargs))

        peft_config = self._adapter_config(base_sync_done, merge_and_sync)

        self._ensure_model_keys()
        _together(
            lambda: self.model.send_weights(
                base_sync_done=base_sync_done, merge_and_sync=merge_and_sync, model_keys=self._model_keys),
            lambda: receiver.receive_weights(base_sync_done=base_sync_done, peft_config=peft_config))

        self.model.finalize_checkpoint_engine()
        receiver.finalize_checkpoint_engine()

    def _sync_weights_direct_vllm(self, merge_and_sync):
        """Colocated trainer bucket -> vLLM TP workers, for both full and LoRA-only syncs.

        Non-CUDA platforms and other sampler backends retain the checkpoint-engine
        protocol. Physical UUIDs, rather than inferred rank strides, pair each receiver
        with its trainer device.
        """
        if (os.environ.get('TWINKLE_DISABLE_DIRECT_VLLM_SYNC', '0') == '1'
                or not hasattr(self.sampler, 'direct_weight_sync_metadata')
                or not hasattr(self.model, 'direct_weight_sync_device')):
            return False
        replicas = self.sampler.direct_weight_sync_metadata()
        if not isinstance(replicas, list) or not replicas or any(target is None for target in replicas):
            return False
        targets = [target for replica in replicas for target in (replica if isinstance(replica, list) else [replica])]
        devices = self.model.direct_weight_sync_device()
        receiver_devices = [target['device_uuid'] for target in targets]
        if len(set(receiver_devices)) != len(receiver_devices):
            raise ValueError('Direct vLLM receivers must have unique physical device UUIDs.')
        if any(devices.count(device) != 1 for device in receiver_devices):
            raise ValueError('Each direct vLLM receiver must share exactly one trainer device.')
        peft_config = self._adapter_config(self.base_sync_done, merge_and_sync)
        self._ensure_model_keys()
        _together(
            lambda: self.model.send_weights_to_vllm(
                targets, channel=self.channel, auth_key=self._auth_key,
                base_sync_done=self.base_sync_done, merge_and_sync=merge_and_sync,
                peft_config=peft_config, model_keys=self._model_keys),
            lambda: self.sampler.receive_weights_direct(
                targets, auth_key=self._auth_key, base_sync_done=self.base_sync_done,
                peft_config=peft_config))
        if not self.base_sync_done:
            self.base_sync_done = True
            if not merge_and_sync:
                logger.info('Base model sync completed, subsequent syncs will be LoRA-only')
        logger.info('Direct colocated weight sync completed: %s vLLM workers (per-device trainer -> worker IPC).',
                    len(targets))
        return True

    # ── New replicas ─────────────────────────────────────────────────────────

    @remote_function(collect='first', lazy_collect=False)
    def warm(self, sub) -> None:
        """Load the current weights into new sampler replicas before they take requests.

        ``sub`` is a handle holding only the new replicas (``add_workers`` passes it to its
        ``warmup``). The transfer runs on a channel of its own, closed afterwards; the next
        :meth:`sync_weights` rebuilds the main channel for the grown replica set.
        """
        if self.mode != 'standalone' or self.sync_backend != 'full':
            raise NotImplementedError(
                f'warm needs mode="standalone" with sync_backend="full" (got mode={self.mode!r}, '
                f'sync_backend={self.sync_backend!r}): colocated replicas share the trainer\'s devices '
                'and the delta session has a fixed replica set.')
        with self._sync_lock:
            channel = f'{self.channel}-warm-{uuid.uuid4().hex[:8]}'
            try:
                count = len(self._activate_checkpoint_channel(sub, channel))
                # LoRA-only syncs need the base first, then the current adapter on top of it.
                self._transfer(sub, count, self._merge_and_sync, False)
                if not self._merge_and_sync and self.base_sync_done:
                    self._transfer(sub, count, self._merge_and_sync, True)
            finally:
                self._close_checkpoint_channel(sub, channel)

    # ── Delta ────────────────────────────────────────────────────────────────

    def _prepare_delta_transport(self):
        capabilities = self.sampler.validate_delta_sampler(self.delta_config)
        replicas = capabilities if isinstance(capabilities, list) else [capabilities]
        if self._delta_platform == 'NPU':
            return self._prepare_delta_transport_npu(replicas)
        workers = self.sampler.delta_sampler_metadata()
        if len(workers) != sum(replica['tp'] for replica in replicas):
            raise ValueError('delta topology does not include every rollout TP worker')
        devices = {(worker['node_ip'], worker['device_uuid']) for worker in workers}
        if len(devices) != len(workers):
            raise ValueError('delta checkpoint receivers must occupy distinct rollout GPUs')
        metadata = self.model.prepare_delta_sync(self.delta_config)
        return dict(metadata, world_size=len(workers) + 1, rollout_workers=workers), len(replicas)

    def _prepare_delta_transport_npu(self, replicas):
        """NPU rendezvous: workers join the trainer's HCCL group themselves, so there is no worker list.

        Group rank 0 is the trainer; rollout workers follow replica-major, TP-minor, so the world size is
        every replica's TP size plus the trainer.
        """
        metadata = self.model.prepare_delta_sync(self.delta_config)
        for replica in replicas:
            if replica['device'] != metadata['device']:
                raise ValueError(f"delta device mismatch: trainer is {metadata['device']!r}, "
                                 f"rollout worker is {replica['device']!r}")
        ranks = [replica.get('replica_rank') for replica in replicas]
        if sorted(ranks, key=lambda rank: (rank is None, rank)) != list(range(len(replicas))):
            raise ValueError(f'delta replica ranks {ranks} must be exactly 0..{len(replicas) - 1} without duplicates')
        world_size = sum(replica['tp'] for replica in replicas) + 1
        return dict(metadata, world_size=world_size), len(replicas)

    def _sync_delta_weights(self, merge_and_sync):
        if not merge_and_sync:
            raise ValueError('delta_sharded does not support LoRA synchronization')
        if self._delta_failed:
            raise RuntimeError('delta session failed; restart the entire training/rollout job before full seed')
        started = perf_counter()
        seed = self._delta_replica_count is None
        if seed:
            endpoint, replica_count = self._prepare_delta_transport()
        try:
            self.sampler.begin_delta_sync()
            if seed:
                _together(lambda: self.model.init_delta_transport(endpoint),
                          lambda: self.sampler.init_delta_sampler_transport(endpoint))
                self._delta_replica_count = replica_count
            metrics, worker_metrics = _together(self.model.send_delta_weights, self.sampler.receive_delta_weights)
            results = worker_metrics if isinstance(worker_metrics, list) else [worker_metrics]
            if len(results) != self._delta_replica_count or any(
                    result['changed'] != results[0]['changed'] for result in results):
                raise ValueError('delta rollout replicas disagree about update completion')
            self.sampler.publish_delta_sampler()
            self._delta_round += 1
            self.base_sync_done = True
            self.last_delta_metrics = {'trainers': metrics, 'workers': worker_metrics,
                                       'total_seconds': perf_counter() - started, 'seed': seed}
        except Exception:
            self._delta_failed = True
            try:
                self.sampler.fail_delta_sampler()
            except Exception:
                logger.exception('Could not pause failed delta sampler; restart rollout actors before serving')
            raise

    @remote_function(collect='first', lazy_collect=False)
    def close_sync_session(self):
        """Release delta snapshots and communicators; keep rollout paused until a new full seed."""
        with self._sync_lock:
            if self.sync_backend == 'delta_sharded':
                self._close_delta_session()

    def _close_delta_session(self):
        errors = []
        close_methods = (self.sampler.close_delta_sampler, self.model.close_delta_trainer)
        for close in close_methods:
            try:
                close()
            except Exception as error:
                errors.append(error)
        if errors:
            self._delta_failed = True
            raise RuntimeError('delta cleanup failed; restart the affected actors') from errors[0]
        self._delta_replica_count = None
        self._delta_round = 0
        # Cleanup cannot make an unknown partially updated model safe for a new session.
        self.base_sync_done = False

    # ── Shared helpers ───────────────────────────────────────────────────────

    def _adapter_config(self, base_sync_done: bool, merge_and_sync: bool) -> Optional[Dict[str, Any]]:
        """The cached PEFT config for a LoRA-only sync; None for a full transfer.

        ``base_sync_done`` is a parameter, not ``self.base_sync_done``: :meth:`warm` stages a
        new replica's base first (False) and only then the adapter (True), so the two can
        differ within one manager.
        """
        if not base_sync_done or merge_and_sync:
            return None
        if self._peft_config is None:
            self._peft_config = self.model.get_peft_config_dict()
        return self._peft_config

    def _ensure_model_keys(self):
        if self._model_keys is not None:
            return

        if hasattr(self.sampler, 'get_state_keys'):
            self._model_keys = self.sampler.get_state_keys()

        if self._model_keys is None:
            self._model_keys = []

        # vLLM may have grouped params - use word boundaries to avoid substring matches
        import re
        _STACKED_MAPPINGS = [
            (re.compile(r'\bqkv_proj\b'), ('q_proj', 'k_proj', 'v_proj', 'q', 'k', 'v')),
            (re.compile(r'\bgate_up_proj\b'), ('gate_proj', 'up_proj')),
            (re.compile(r'\bin_proj_qkvz\b'), ('in_proj_qkv', 'in_proj_z')),
            (re.compile(r'\bin_proj_ba\b'), ('in_proj_b', 'in_proj_a')),
            (re.compile(r'\blanguage_model\.model\b'), ('model.language_model', )),
            (re.compile(r'^visual\.'), ('model.visual.', )),
        ]

        def _expand_keys(keys):
            result = set(keys)
            for key in keys:
                for pattern, individuals in _STACKED_MAPPINGS:
                    if pattern.search(key):
                        for ind in individuals:
                            result.add(pattern.sub(ind, key))
            return result

        # Two passes for chain expansion (e.g., language_model.model + qkv_proj)
        expanded = _expand_keys(self._model_keys)
        expanded = _expand_keys(expanded)
        self._model_keys = list(expanded)

    def _sync_weights_naive(self, merge_and_sync):
        """Stream model weights directly into a local sampler."""
        peft_config = self._adapter_config(self.base_sync_done, merge_and_sync)

        self._ensure_model_keys()
        weights = self.model._get_weight_generator(
            base_sync_done=self.base_sync_done,
            merge_and_sync=merge_and_sync,
            model_keys=self._model_keys,
        )
        self.sampler.receive_weights(
            weights=weights,
            base_sync_done=self.base_sync_done,
            peft_config=peft_config,
        )

        if not self.base_sync_done:
            self.base_sync_done = True
            if not merge_and_sync:
                logger.info('Base model sync completed, subsequent syncs will be LoRA-only')

    # ── Paired EMA sync (policy → reference) ────────────────────────────────

    @remote_function(collect='first', lazy_collect=False)
    def sync_ema(self, alpha: float):
        """Synchronize reference weights as an EMA of policy weights.

        Each policy rank k and reference rank k form an independent 2-rank NCCL group:
        policy sends its LOCAL shard parameters, reference receives and applies
        ``param.mul_(alpha).add_(received, alpha=1-alpha)``.  Peak memory per rank is two engine
        buckets -- the engine allocates its staging buffers before the send/receive roles split,
        so the unused direction's buffer is paid for too (configurable, default 2x3 GB),
        independent of model size.

        Requires policy and reference to have the SAME parallel_spec (identical sharding),
        so local parameter names and shapes match one-to-one across the pair. ``sampler`` is the
        reference model here.

        Args:
            alpha: EMA decay factor.  ``ref = alpha * ref + (1 - alpha) * policy``.
                   Closer to 1 means slower tracking.
        """
        with self._sync_lock:
            if self.mode == 'naive':
                self._sync_ema_naive(alpha)
                return

            self._members = self._activate_checkpoint_channel()
            world_size = self.model.device_mesh.world_size
            ref_world_size = self.sampler.device_mesh.world_size
            if world_size != ref_world_size:
                raise ValueError(
                    f'Paired EMA requires policy and reference to have the same world_size '
                    f'(policy={world_size}, reference={ref_world_size}); use the same parallel_spec.')

            # Phase 1: all policy ranks prepare as master (each gets a unique TCPStore port).
            all_metadata = self.model.prepare_ema_sender()
            if not isinstance(all_metadata, list):
                all_metadata = [all_metadata]
            # Reference ranks prepare as non-master (allocate receive buffers).
            # Scalar False: dispatch='slice' broadcasts it identically to every rank.
            self.sampler.prepare_checkpoint_engine(False)

            # Phase 2: init paired 2-rank groups (policy rank k = NCCL rank 0, ref rank k = rank 1).
            _together(
                lambda: self.model.init_checkpoint_process_group(
                    rank=[0] * world_size, world_size=[2] * world_size, master_metadata=all_metadata),
                lambda: self.sampler.init_checkpoint_process_group(
                    rank=[1] * world_size, world_size=[2] * world_size, master_metadata=all_metadata))

            # Phase 3: concurrent send (local shard) / receive+EMA.
            _together(lambda: self.model.send_weights(local=True), lambda: self.sampler.receive_weights_ema(alpha))

            # Phase 4: finalize.
            self.model.finalize_checkpoint_engine()
            self.sampler.finalize_checkpoint_engine()

    def _sync_ema_naive(self, alpha: float):
        """In-process EMA: iterate policy's local params and blend into reference's."""
        import torch
        policy_state = dict(self.model._get_local_state_dict())
        ref_state = dict(self.sampler._get_local_state_dict())
        with torch.no_grad():
            for name, policy_param in policy_state.items():
                ref_param = ref_state[name]
                # A local pair may straddle devices (a policy on the GPU, a frozen
                # reference kept on CPU): blend on the reference's device. A
                # same-device pair pays only a no-op ``to``.
                policy_data = policy_param.data.to(ref_param.device)
                ref_param.data.mul_(alpha).add_(policy_data, alpha=1.0 - alpha)
