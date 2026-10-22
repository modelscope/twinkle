# Copyright (c) ModelScope Contributors. All rights reserved.
"""Configuration for the delta checkpoint engine."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, Union
from typing_extensions import NotRequired

if TYPE_CHECKING:
    from ..hccl_checkpoint_engine import MasterMetadata as HCCLMasterMetadata
    from ..nccl_checkpoint_engine import MasterMetadata

    DeltaMasterMetadata = Union[MasterMetadata, HCCLMasterMetadata]
else:
    DeltaMasterMetadata = object

SPARSE_ELEMENT_BYTES = 6


@dataclass(frozen=True)
class DeltaSyncConfig:
    bucket_bytes: int = 64 << 20
    timeout_seconds: float = 300
    gather_batch_size: int = 32
    delta_pin_snapshots: bool = True
    free_cache_engine: bool = True

    def __post_init__(self):
        if not isinstance(self.gather_batch_size, int) or self.gather_batch_size < 1:
            raise ValueError('delta gather batch size must be a positive integer')
        if not isinstance(self.bucket_bytes, int) or self.bucket_bytes < SPARSE_ELEMENT_BYTES:
            raise ValueError('delta bucket must hold at least one int32 index and BF16 value')
        if not 0 < self.timeout_seconds < float('inf'):
            raise ValueError('delta timeout must be finite and positive')


class DeltaTransportEndpoint(TypedDict):
    world_size: int
    master_metadata: DeltaMasterMetadata
    # CUDA only: NPU workers join the trainer's HCCL group themselves, so there is no worker list.
    rollout_workers: NotRequired[list[DeltaWorkerMetadata]]
    # NPU only: trainer rank 0's HCCL socket env; workers apply it before joining the group.
    hccl_env: NotRequired[dict[str, str]]


class DeltaTrainerMetadata(TypedDict):
    device: str
    master_metadata: DeltaMasterMetadata | None
    hccl_env: NotRequired[dict[str, str]]


class DeltaWorkerCapabilities(TypedDict):
    tp: int
    device: str
    # NPU only: data-parallel index of the reporting sampler replica.
    replica_rank: NotRequired[int]


class DeltaWorkerMetadata(TypedDict):
    node_ip: str
    device_uuid: str
    cuda_visible_device: NotRequired[str]
    zmq_handle: str


class DeltaReceiverPlacement(TypedDict):
    rank: int
    node_id: str
    device_uuid: str
    cuda_visible_device: NotRequired[str]
    zmq_handle: str
