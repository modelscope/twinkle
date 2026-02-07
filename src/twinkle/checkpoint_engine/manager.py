# Copyright (c) ModelScope Contributors. All rights reserved.
# Adapted from https://github.com/volcengine/verl/blob/main/verl/checkpoint_engine/base.py
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

import logging
import time
from typing import TYPE_CHECKING

from .base import CheckpointEngineRegistry

if TYPE_CHECKING:
    from twinkle.model.base import TwinkleModel
    from twinkle.sampler.vllm_sampler import VLLMSampler

logger = logging.getLogger(__name__)


class CheckpointEngineManager:
    """Coordinate weight synchronization from model to sampler via NCCL broadcast.

    This manager orchestrates a 5-step weight sync flow between Ray actors:
    1. prepare     — allocate NCCL buffers, start ZMQ metadata server
    2. build_topology — assign NCCL ranks (trainer[0]→rank0, sampler→rank1..N)
    3. init_process_group — create temporary NCCL group across actors
    4. send / receive    — trainer broadcasts, sampler receives (in parallel)
    5. finalize   — release buffers, optionally destroy NCCL group

    LoRA-aware sync (following verl's design):
    - First sync (``base_sync_done=False``): broadcasts ALL weights (base model)
      so that vLLM (loaded with ``load_format='dummy'``) gets real weights.
      On the sampler side, ``.base_layer`` is stripped from PEFT weight names
      if ``enable_lora=False`` in vLLM, so names match vLLM's model structure.
    - Subsequent syncs (``base_sync_done=True``): broadcasts ONLY LoRA adapter
      weights.  On the sampler side (if ``enable_lora=True``), these are loaded
      via ``add_lora()`` as a tensor-based LoRA adapter.

    Args:
        model: Training model with Ray actors (``model._actors``).
        sampler: Inference sampler with Ray actors (``sampler._actors``).
        backend: Checkpoint engine backend (``'nccl'`` or ``'hccl'``).
        bucket_size_mb: Size of each weight-transfer bucket in MB.
    """

    def __init__(
        self,
        model: "TwinkleModel",
        sampler: "VLLMSampler",
        backend: str = 'nccl',
        bucket_size_mb: int = 2048,
    ) -> None:
        self.model = model
        self.sampler = sampler
        self.backend = backend
        self.bucket_size_mb = bucket_size_mb
        self.backend_cls = CheckpointEngineRegistry.get(backend) # nccl, hccl

        # Validate Ray actors
        assert hasattr(model, '_actors') and model._actors, \
            "CheckpointEngineManager requires model to be deployed as Ray actors"
        assert hasattr(sampler, '_actors') and sampler._actors, \
            "CheckpointEngineManager requires sampler to be deployed as Ray actors"

        self.model_actors = model._actors
        self.sampler_actors = sampler._actors

        # LoRA sync state: tracks whether the first full sync has been done.
        # After the first sync, only LoRA adapter weights are transferred.
        self.base_sync_done: bool = False
        # Cached peft_config dict for LoRA-only sync.
        # Fetched lazily from the model on first LoRA sync.
        self._peft_config: dict | None = None

        logger.info(
            f"CheckpointEngineManager: backend={backend}, "
            f"model_workers={len(self.model_actors)}, "
            f"sampler_workers={len(self.sampler_actors)}"
        )

    def sync_weights(self, adapter_name: str = ''):
        """Synchronize weights from model to sampler via NCCL broadcast.

        This is a **blocking** call. It performs:
        1. prepare   → allocate buffers on all workers
        2. topology  → assign NCCL ranks
        3. init_pg   → create temporary NCCL process group
        4. transfer  → model broadcasts, sampler receives (parallel)
        5. finalize  → release buffers and process group

        LoRA-aware behaviour:
        - If the model has a LoRA adapter (``adapter_name`` is not empty) and
          this is NOT the first sync, only LoRA weights are sent.
        - The sampler side knows via ``peft_config`` / ``base_sync_done``
          whether to use ``load_weights()`` or ``add_lora()`` to apply them.

        Args:
            adapter_name: Adapter name for LoRA weight sync.  When non-empty
                and ``base_sync_done`` is True, only LoRA weights are sent.
        """
        import ray

        start_time = time.time()
        is_lora_only = self.base_sync_done and bool(adapter_name)
        model_actors = self.model_actors
        sampler_actors = self.sampler_actors

        # ── Step 1: Prepare ──────────────────────────────────────────────
        # All workers allocate buffers. Model actor[0] is designated as
        # the master (is_master=True) and starts a ZMQ PUB server.
        logger.debug("Step 1/5: prepare checkpoint engines")
        model_prep = [
            a.prepare_checkpoint_engine.remote(is_master=(i == 0))
            for i, a in enumerate(model_actors)
        ]
        sampler_prep = [
            a.prepare_checkpoint_engine.remote(is_master=False)
            for a in sampler_actors
        ]
        all_metadata = ray.get(model_prep + sampler_prep)
        model_metadata = all_metadata[:len(model_actors)]

        # ── Step 2: Build topology ───────────────────────────────────────
        # trainer[0] → NCCL rank 0 (source)
        # trainer[1..] → rank -1 (not participating)
        # sampler[0..N-1] → NCCL rank 1..N (receivers)
        logger.debug("Step 2/5: build topology")
        model_kwargs, sampler_kwargs = self.backend_cls.build_topology(
            len(model_actors), len(sampler_actors), model_metadata,
        )

        # ── Step 3: Init process group ───────────────────────────────────
        logger.debug("Step 3/5: init process group")
        init_refs = []
        for i, actor in enumerate(model_actors):
            init_refs.append(actor.init_checkpoint_process_group.remote(
                rank=model_kwargs["rank"][i],
                world_size=model_kwargs["world_size"][i],
                master_metadata=model_kwargs["master_metadata"][i],
            ))
        for i, actor in enumerate(sampler_actors):
            init_refs.append(actor.init_checkpoint_process_group.remote(
                rank=sampler_kwargs["rank"][i],
                world_size=sampler_kwargs["world_size"][i],
                master_metadata=sampler_kwargs["master_metadata"][i],
            ))
        ray.get(init_refs)

        # ── Step 3.5: Fetch peft_config if needed for LoRA-only sync ──────
        # On the first LoRA-only sync, fetch the peft_config from the model
        # and cache it.  This is needed by the sampler's add_lora() path.
        peft_config = None
        if self.base_sync_done and adapter_name:
            if self._peft_config is None:
                self._peft_config = ray.get(
                    model_actors[0].get_peft_config_dict.remote(adapter_name)
                )
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

        # ── Step 5: Finalize ─────────────────────────────────────────────
        logger.debug("Step 5/5: finalize")
        fin_refs = [a.finalize_checkpoint_engine.remote() for a in model_actors]
        fin_refs += [a.finalize_checkpoint_engine.remote() for a in sampler_actors]
        ray.get(fin_refs)

        # Mark base sync as done after first successful full sync
        if not self.base_sync_done:
            self.base_sync_done = True
            logger.info("Base model sync completed, subsequent syncs will be LoRA-only")

        elapsed = time.time() - start_time
        mode = "LoRA-only" if is_lora_only else "full"
        logger.info(f"Weight sync ({mode}) completed in {elapsed:.2f}s")
