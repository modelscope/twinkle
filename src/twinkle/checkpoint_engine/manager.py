# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/base.py
import time
from typing import Optional

from twinkle import get_logger, Platform
from .base import CheckpointEngine
from .mixin import CheckpointEngineMixin

logger = get_logger()


class CheckpointEngineManager:
    """Weight synchronization manager for Twinkle (STANDALONE mode).

    Coordinates weight synchronization between training model and inference sampler
    when they reside on **different GPUs** (disaggregated / standalone deployment).

    Architecture (following verl's CheckpointEngineManager):

        Trainer GPU(s)                          Rollout GPU(s)
        ┌──────────────────┐                    ┌──────────────────┐
        │ TransformersModel│                    │   VLLMSampler    │
        │  (Ray actors)    │                    │  (Ray actors)    │
        │        │         │                    │        │         │
        │        ▼         │                    │        ▼         │
        │ CheckpointEngine │   NCCL broadcast   │ CheckpointEngine │
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
        model: "CheckpointEngineMixin",
        sampler: "CheckpointEngineMixin",
        platform: str = 'GPU',
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.backend_cls = self.decide_backend_engine(platform)

        # Validate Ray actors
        assert hasattr(model, '_actors') and model._actors, \
            "CheckpointEngineManager requires model to be deployed as Ray actors"
        assert hasattr(sampler, '_actors') and sampler._actors, \
            "CheckpointEngineManager requires sampler to be deployed as Ray actors"

        # LoRA sync state: tracks whether the first full sync has been done.
        # After the first sync, only LoRA adapter weights are transferred.
        self.base_sync_done: bool = False
        # Cached peft_config dict for LoRA-only sync.
        # Fetched lazily from the model on first LoRA sync.
        self._peft_config: dict | None = None

    @staticmethod
    def decide_backend_engine(platform: Optional[str] = None) -> "CheckpointEngine":
        if Platform.get_platform(platform).__name__ == 'GPU':
            from twinkle.checkpoint_engine import NCCLCheckpointEngine
            return NCCLCheckpointEngine
        elif Platform.get_platform(platform).__name__ == 'NPU':
            from twinkle.checkpoint_engine import HCCLCheckpointEngine
            return HCCLCheckpointEngine
        else:
            raise NotImplementedError

    def sync_weights(self, adapter_name: str = ''):
        start_time = time.time()
        is_lora_only = self.base_sync_done and bool(adapter_name)
        model_metadata = self.model.prepare_checkpoint_engine([True] + [False]*(len(self.model.device_mesh.world_size)-1))
        self.sampler.prepare_checkpoint_engine(False)
        model_kwargs, sampler_kwargs = self.backend_cls.build_topology(
            self.model.device_mesh.world_size, self.sampler.device_mesh.world_size, [model_metadata],
        )
        self.model.init_checkpoint_process_group(**model_kwargs)
        self.sampler.init_checkpoint_process_group(**sampler_kwargs)
        peft_config = None
        if self.base_sync_done and adapter_name:
            if self._peft_config is None:
                self._peft_config = self.model.get_peft_config_dict(adapter_name)
            peft_config = self._peft_config

        # ── Step 4: Send / Receive (parallel) ────────────────────────────
        logger.debug("Step 4/5: send & receive weights")
        send_refs = [
            a.send_weights_via_checkpoint_engine.remote(
                adapter_name=adapter_name,
                base_sync_done=self.base_sync_done,
            )
            for a in model_actors
        ]
        recv_refs = [
            a.receive_weights_via_checkpoint_engine.remote(
                base_sync_done=self.base_sync_done,
                peft_config=peft_config,
            )
            for a in sampler_actors
        ]
        ray.get(send_refs + recv_refs)

        self.model.send_weights()
        self.sampler.receive_weights()

        self.model.finalize_checkpoint_engine()
        self.sampler.finalize_checkpoint_engine()

        if not self.base_sync_done:
            self.base_sync_done = True
            logger.info("Base model sync completed, subsequent syncs will be LoRA-only")

        elapsed = time.time() - start_time
        mode = "LoRA-only" if is_lora_only else "full"
        logger.info(f"Weight sync ({mode}) completed in {elapsed:.2f}s")
