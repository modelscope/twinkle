# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import threading
from typing import List

from twinkle import Platform, remote_function
from twinkle.checkpoint_engine.base import CheckpointEngine
from twinkle.perf import timed


class CheckpointEngineMixin:

    _checkpoint_engine: CheckpointEngine = None
    _checkpoint_engines = None
    _checkpoint_engine_key = None
    _bucket_size: int = 3072 << 20  # 2 GB
    # Set by CheckpointEngineManager when the roles share a GPU; see set_checkpoint_engine_backend.
    _checkpoint_engine_backend: str = None
    # Per-channel HMAC keys, delivered by the manager over the (trusted) Ray control path.
    _checkpoint_engine_auth_keys = None

    def _select_checkpoint_engine(self, backend: str, channel: str, auth_key: str = '') -> None:
        if not isinstance(channel, str) or not channel:
            raise ValueError(f'checkpoint channel must be a non-empty string, got {channel!r}.')
        if backend not in {'ipc', 'nccl', 'hccl'}:
            raise ValueError(f'checkpoint backend must be ipc/nccl/hccl, got {backend!r}.')
        self._checkpoint_engine_backend = backend
        self._checkpoint_engine_key = (backend, channel)
        if auth_key:
            if self._checkpoint_engine_auth_keys is None:
                self._checkpoint_engine_auth_keys = {}
            self._checkpoint_engine_auth_keys[(backend, channel)] = auth_key
        engines = getattr(self, '_checkpoint_engines', None)
        self._checkpoint_engine = engines.get(self._checkpoint_engine_key) if engines else None

    @remote_function(dispatch='all', lazy_collect=False, exclusive=True)
    def activate_checkpoint_engine(self, backend: str, channel: str = 'default', auth_key: str = ''):
        """Select one persistent transport channel without discarding the other channels.

        ``auth_key`` signs the engine's control frames (IPC) over the unauthenticated socket path;
        it travels through the Ray control path, which the deployment already trusts.
        """
        self._select_checkpoint_engine(backend, channel, auth_key)

    @remote_function(dispatch='all', lazy_collect=False, exclusive=True)
    def set_checkpoint_engine_backend(self, backend: str):
        """Choose the default transport channel for backward-compatible callers."""
        self._select_checkpoint_engine(backend, 'default')

    def _get_or_create_checkpoint_engine(self) -> 'CheckpointEngine':
        """Get or create the engine for the currently active transport channel."""
        backend = self._checkpoint_engine_backend
        if backend is None:
            backend = 'nccl' if Platform.get_platform().__name__ == 'GPU' else 'hccl'
        key = self._checkpoint_engine_key or (backend, 'default')
        engines = getattr(self, '_checkpoint_engines', None)
        if engines is None:
            engines = {}
            self._checkpoint_engines = engines
        engine = engines.get(key)
        if engine is None:
            if backend == 'ipc':
                # Colocated: the sampler is on this GPU, so weights are mapped, not broadcast. NCCL
                # is not an option here -- it refuses two ranks on one device. The channel reaches
                # the engine so the endpoint separates colocate managers sharing a GPU pair; the
                # bucket only passes when the class default was overridden, because the IPC default
                # (512 MB) is deliberately smaller than this mixin's NCCL-tuned 3 GB.
                from twinkle.checkpoint_engine import IPCCheckpointEngine
                bucket = None if type(self)._bucket_size == CheckpointEngineMixin._bucket_size else self._bucket_size
                auth_keys = self._checkpoint_engine_auth_keys or {}
                engine = IPCCheckpointEngine(bucket, channel=key[1], auth_key=auth_keys.get(key, ''))
            elif backend == 'nccl':
                from twinkle.checkpoint_engine import NCCLCheckpointEngine
                engine = NCCLCheckpointEngine(self._bucket_size)
            elif backend == 'hccl':
                from twinkle.checkpoint_engine import HCCLCheckpointEngine

                # Reusing HCCL communicator across sync steps avoids frequent
                # stream/channel allocation and reduces resource exhaustion risk.
                engine = HCCLCheckpointEngine(
                    self._bucket_size,
                    rebuild_group=False,
                )
            engines[key] = engine
        self._checkpoint_engine_key = key
        self._checkpoint_engine = engine
        return engine

    @remote_function(collect='first', lazy_collect=False, exclusive=True)
    def prepare_checkpoint_engine(self, is_master):
        engine = self._get_or_create_checkpoint_engine()
        engine.is_master = is_master
        return engine.prepare()

    @remote_function(dispatch='slice', lazy_collect=True, exclusive=True)
    def init_checkpoint_process_group(self, rank: int, world_size: int, master_metadata):
        """Initialize process group for weight synchronization."""
        if isinstance(rank, list):
            assert len(rank) == 1
            rank = rank[0]
        if isinstance(world_size, list):
            assert len(world_size) == 1
            world_size = world_size[0]
        if isinstance(master_metadata, list):
            assert len(master_metadata) == 1
            master_metadata = master_metadata[0]
        engine = self._get_or_create_checkpoint_engine()
        engine.init_process_group(
            rank=rank,
            world_size=world_size,
            master_metadata=master_metadata,
        )

    @remote_function(dispatch='all', lazy_collect=False, exclusive=True)
    def finalize_checkpoint_engine(self):
        """Finalize the active engine while retaining a reusable communication group."""
        engine = self._get_or_create_checkpoint_engine()
        engine.finalize()

    @remote_function(dispatch='all', lazy_collect=False, exclusive=True)
    def reset_checkpoint_engine(self, channel: str = 'default', backend: str = None):
        """Destroy selected persistent channels when their remote peer is retired."""
        engines = getattr(self, '_checkpoint_engines', None) or {}
        keys = [key for key in engines if key[1] == channel and (backend is None or key[0] == backend)]
        for key in keys:
            engines.pop(key).close()
        if self._checkpoint_engine_key in keys:
            self._checkpoint_engine = None
            self._checkpoint_engine_key = None

    def _get_weight_generator(self, adapter_name=None, base_sync_done=False, merge_and_sync=False,
                              model_keys=None, **kwargs):
        """Yield (name, tensor) of the weights to broadcast to a receiver.

        Implemented by weight SENDERS (the training models). Declared here so a host
        missing it fails with this message, not a bare AttributeError at the send site.
        """
        raise NotImplementedError(
            f'{type(self).__name__} is not a weight sender: implement _get_weight_generator on the '
            'training-model class (see TransformersModel / MegatronModel).')

    def _get_local_weight_generator(self):
        """Yield (name, local_tensor) of this rank's shard, without cross-rank gathering.

        Serves paired EMA (``send_weights(local=True)``); the receiver holds the SAME
        shard layout (identical parallel_spec), so names and shapes match one-to-one.
        """
        raise NotImplementedError(
            f'{type(self).__name__} is not a weight sender: implement _get_local_weight_generator on the '
            'training-model class (see TransformersModel / MegatronModel).')

    def _get_local_state_dict(self):
        """Return {name: param} of this rank's local shard (EMA receiver side).

        Values are the live parameters (or param.data) that ``receive_weights_ema``
        blends in place; a received name missing from this dict raises KeyError.
        """
        raise NotImplementedError(
            f'{type(self).__name__} does not hold local weight shards: implement _get_local_state_dict '
            'on the class that receives EMA updates (see TransformersModel / MegatronModel).')

    @remote_function(dispatch='all', lazy_collect=True, exclusive=True)
    def send_weights(
        self,
        adapter_name: str = None,
        base_sync_done: bool = False,
        merge_and_sync: bool = False,
        model_keys: List[str] = None,
        local: bool = False,
        **kwargs,
    ):
        engine = self._get_or_create_checkpoint_engine()
        if local:
            weight_generator = self._get_local_weight_generator()
        else:
            weight_generator = self._get_weight_generator(
                adapter_name=adapter_name,
                base_sync_done=base_sync_done,
                merge_and_sync=merge_and_sync,
                model_keys=model_keys,
                **kwargs,
            )

        # In local (paired EMA) mode every rank is a sender in its own 2-rank group.
        # In broadcast mode only engine rank 0 actually sends; others consume silently
        # (they still must iterate the generator for TP allgather collectives to complete).
        is_sender = local or (engine.rank is not None and engine.rank == 0)

        if not is_sender:
            for _name, _tensor in weight_generator:
                pass
            return

        # Run async send_weights in a dedicated event loop thread.
        # We cannot use the Ray worker's event loop because it may already
        # be occupied, and send_weights uses run_in_executor internally.
        async def _send():
            await engine.send_weights(weight_generator)

        result_container = {'error': None}

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_send())
                finally:
                    loop.close()
            except Exception as e:
                result_container['error'] = e

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()

        if result_container['error'] is not None:
            raise result_container['error']

    @remote_function(dispatch='all', collect='none', lazy_collect=False, exclusive=True)
    def prepare_ema_sender(self):
        """Prepare this rank's engine as master for a paired 2-rank EMA group.

        Returns this rank's MasterMetadata (unique TCPStore port) so the manager
        can relay it to the paired reference rank.
        """
        engine = self._get_or_create_checkpoint_engine()
        engine.is_master = True
        return engine.prepare()

    @remote_function(dispatch='all', lazy_collect=True, exclusive=True)
    @timed
    def receive_weights_ema(self, alpha: float, **kwargs):
        """Receive policy weights via checkpoint-engine and apply EMA to local params.

        Streaming: processes one bucket at a time, so peak memory is one bucket
        (not the full model).  Each received (name, tensor) is matched to the
        local state dict by name; a mismatch raises KeyError immediately.

        The local state comes from ``_get_local_state_dict``, whose values are the
        live parameters (``param.data``); ``param.data`` also tolerates plain
        tensors, so hosts storing raw tensors keep working.
        """
        engine = self._get_or_create_checkpoint_engine()
        local_state = self._get_local_state_dict()

        async def _receive_and_ema():
            async for name, tensor in engine.receive_weights():
                param = local_state[name]
                param.data.mul_(alpha).add_(tensor, alpha=1.0 - alpha)

        result_container = {'error': None}

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_receive_and_ema())
                finally:
                    loop.close()
            except Exception as e:
                result_container['error'] = e

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()

        if result_container['error'] is not None:
            raise result_container['error']
