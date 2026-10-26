# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sparse delta weight synchronization.

Layout:
    config.py      DeltaSyncConfig and the TypedDicts exchanged between trainer and rollout
    platform.py    resolve_delta_engine_cls: GPU -> NCCL engine, NPU -> HCCL engine (lazy import)
    protocol/      wire format shared by both sides
        wire.py        DeltaFlush / DeltaParam structures and checksum
        decode.py      device-agnostic decode of one received flush (no vLLM dependency)
    trainer/       trainer side: what changed and how it is exported
        mixin.py       DeltaTrainerMixin: trainer-side Ray lifecycle
        spec.py        ShardSpec / BlockPlacement: where a local shard sits in the full tensor
        export.py      snapshot, diff and HF-coordinate delta export helpers
        backends/      per-training-backend export hooks (base.py contract, fsdp.py implementation)
    transport/     sender pipeline and its transports
        pipeline.py    DeltaSyncMixin: transport-agnostic bucket/flush pipeline and seed/steady state machine
        gather.py      sparse gather of per-rank deltas to rank 0
        nccl_engine.py DeltaShardedCheckpointEngine: NCCL transport (hooks + rollout receive)
        hccl_engine.py DeltaHCCLCheckpointEngine: HCCL (Ascend NPU) sender transport
    rollout/       rollout side
        mixin.py       DeltaSamplerMixin: rollout-side lifecycle RPCs driven by vLLMSampler
        cuda_receiver.py  CudaDeltaReceiver: Ray actor per vLLM worker, NCCL receive + CUDA IPC relay (GPU)
        hccl_receiver.py  HCCL receive pump that runs inside each vLLM worker process (NPU)
    ../zmq_pubsub.py   zmq PUB/SUB manifest-channel helpers shared by the NCCL and HCCL engines
"""
from .config import DeltaSyncConfig
from .transport.nccl_engine import DeltaShardedCheckpointEngine
from .platform import resolve_delta_engine_cls

__all__ = ['DeltaSyncConfig', 'DeltaShardedCheckpointEngine', 'resolve_delta_engine_cls']
