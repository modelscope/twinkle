# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/base.py
from typing import List, Optional

from twinkle import Platform, get_logger
from .base import CheckpointEngine
from .mixin import CheckpointEngineMixin

logger = get_logger()


class CheckpointEngineManager:
    """Weight synchronization manager for colocated and standalone deployments.

    When the model and sampler are local objects (the colocated/torchrun case),
    weights are streamed directly to the sampler.  When both objects are Ray
    handlers, the checkpoint engine provides the NCCL/HCCL transport between
    disaggregated workers.

    Architecture (following verl's CheckpointEngineManager):

        Trainer GPU(s)                          Rollout GPU(s)
        ┌──────────────────┐                    ┌──────────────────┐
        │ TransformersModel│                    │   vLLMSampler    │
        │  (Ray actors)    │                    │  (Ray actors)    │
        │        │         │                    │        │         │
        │        ▼         │                    │        ▼         │
        │ CheckpointEngine │   NCCL/HCCL        │ CheckpointEngine │
        │  send_weights()  │ ─────────────────► │ receive_weights()│
        │                  │                    │        │         │
        │                  │                    │        ▼         │
        │                  │                    │ VLLMEngine       │
        │                  │                    │  update_weights()│
        │                  │                    │   (CUDA IPC)     │
        │                  │                    │        │         │
        │                  │                    │        ▼         │
        │                  │                    │ vLLM subprocess  │
        │                  │                    │  load_weights()  │
        └──────────────────┘                    └──────────────────┘

    Usage:
        >>> manager = CheckpointEngineManager(model=model, sampler=sampler)
        >>> manager.sync_weights()  # Call after each training step
    """

    def __init__(
        self,
        model: 'CheckpointEngineMixin',
        sampler: 'CheckpointEngineMixin',
        platform: str = 'GPU',
    ) -> None:
        self.model = model
        self.sampler = sampler

        model_has_actors = bool(getattr(model, '_actors', None))
        sampler_has_actors = bool(getattr(sampler, '_actors', None))
        if model_has_actors != sampler_has_actors:
            raise ValueError(
                'CheckpointEngineManager requires model and sampler to use the same deployment mode: '
                'both must be colocated/local objects (without _actors) or both must be Ray actor handlers.')

        self._naive = not model_has_actors
        # Do not even select/import a transport backend for colocated sync.
        self.backend_cls = None if self._naive else self.decide_backend_engine(platform)

        # LoRA sync state: tracks whether the first full sync has been done.
        # After the first sync, only LoRA adapter weights are transferred.
        self.base_sync_done: bool = False
        # Cached peft_config dict for LoRA-only sync.
        # Fetched lazily from the model on first LoRA sync.
        self._peft_config: dict | None = None
        self._model_keys: Optional[List[str]] = None

    @staticmethod
    def decide_backend_engine(platform: Optional[str] = None) -> 'CheckpointEngine':
        if Platform.get_platform(platform).__name__ == 'GPU':
            from twinkle.checkpoint_engine import NCCLCheckpointEngine
            return NCCLCheckpointEngine
        elif Platform.get_platform(platform).__name__ == 'NPU':
            from twinkle.checkpoint_engine import HCCLCheckpointEngine
            return HCCLCheckpointEngine
        else:
            raise NotImplementedError

    def sync_weights(self, merge_and_sync=True):
        """
        Synchronize the weights between the model and the sampler.

        This method ensures that the sampler's weights are consistent with the model's
        current state. It supports two synchronization modes: full merge-and-sync or
        separate base-and-LoRA sync.

        Args:
            merge_and_sync (bool, optional): Whether to merge and sync the weights.
                - If True: LoRA weights are merged into the base model, then the
                combined weights are synchronized to the sampler on every call.
                - If False: On the first call, base model weights are synced to the
                sampler. On subsequent calls, only the LoRA adapter weights are
                synced incrementally.
                Defaults to True.

        Returns:
            None
        """
        if self._naive:
            self._sync_weights_naive(merge_and_sync)
            return

        model_metadata = self.model.prepare_checkpoint_engine([True]
                                                              + [False] * (self.model.device_mesh.world_size - 1))
        self.sampler.prepare_checkpoint_engine(False)
        model_kwargs, sampler_kwargs = self.backend_cls.build_topology(
            self.model.device_mesh.world_size,
            self.sampler.device_mesh.data_world_size,
            [model_metadata],
        )
        # Launch both init calls concurrently — TCPStore server (model rank 0)
        # blocks until all clients (sampler ranks) connect, so these MUST NOT
        # be serialised.  lazy_collect=True makes them return futures.
        model_init = self.model.init_checkpoint_process_group(**model_kwargs)
        sampler_init = self.sampler.init_checkpoint_process_group(**sampler_kwargs)
        model_init()  # wait for model init to complete
        sampler_init()  # wait for sampler init to complete

        peft_config = None
        if self.base_sync_done and not merge_and_sync:
            if self._peft_config is None:
                self._peft_config = self.model.get_peft_config_dict()
            peft_config = self._peft_config

        self._ensure_model_keys()

        model_result = self.model.send_weights(
            base_sync_done=self.base_sync_done, merge_and_sync=merge_and_sync, model_keys=self._model_keys)
        sampler_result = self.sampler.receive_weights(base_sync_done=self.base_sync_done, peft_config=peft_config)
        model_result()
        sampler_result()

        self.model.finalize_checkpoint_engine()
        self.sampler.finalize_checkpoint_engine()

        if not self.base_sync_done:
            self.base_sync_done = True
            if not merge_and_sync:
                logger.info('Base model sync completed, subsequent syncs will be LoRA-only')

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
        """Stream model weights directly into a colocated sampler.

        No checkpoint engine is created in this path.  ``receive_weights``
        continues to use the sampler-to-vLLM IPC transport, but the model to
        sampler handoff is an ordinary in-process generator.
        """
        peft_config = None
        if self.base_sync_done and not merge_and_sync:
            if self._peft_config is None:
                self._peft_config = self.model.get_peft_config_dict()
            peft_config = self._peft_config

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
