# Copyright (c) Model Scope Contributors. All rights reserved.
"""Archived TQ control-plane vocabulary: the run descriptor and partition/group/batch descriptors.

Moved with the TransferQueue runtime that consumes them; the live control plane keeps only
``RolloutPolicy`` / ``PartitionAdmission`` in ``async_rl/types.py`` -- the multi-tenant run
identity (``RLContext`` below) has no live consumer and lives here with its archived callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

from twinkle_agentic.async_rl.types import PartitionAdmission


@dataclass(frozen=True)
class RLContext:
    """What one archived TQ run trains: a single LoRA adapter, or the whole model."""

    tenant_id: str
    training_run_id: str
    base_model_id: str
    adapter_name: str | None = None
    tool_profile: str = 'default'

    def __post_init__(self) -> None:
        if self.adapter_name is not None and not (isinstance(self.adapter_name, str) and self.adapter_name):
            raise ValueError(
                f'adapter_name must be None (full-param) or a non-empty string, got {self.adapter_name!r}')

    @property
    def policy_slot(self) -> str:
        """The key/path component identifying the trained policy: the adapter name, or ``full_param``."""
        return self.adapter_name or 'full_param'

    @property
    def key(self) -> str:
        return f'{self.tenant_id}/{self.training_run_id}/{self.policy_slot}'

    def partition_id(self, step: int) -> str:
        return f'{self.key}/train_{step}'


# Backward-compatible alias archived with its only callers (the TQ pipeline).
LoraContext = RLContext


@dataclass
class PromptGroup:
    """A prompt and the BatchMeta descriptor reserved for its generated samples."""

    context: RLContext
    partition: PartitionAdmission
    group_id: str
    prompt: dict[str, Any]
    batch_meta: Any

    @property
    def partition_id(self) -> str:
        return self.partition.partition_id

    @property
    def num_samples(self) -> int:
        return self.partition.num_generations


@dataclass
class PreparedPartition:
    """Partition prepared in TQ and ready for sampler submission."""

    admission: PartitionAdmission
    groups: tuple[PromptGroup, ...]
    sampling_params: Any


@dataclass
class ClaimedBatch:
    """A TQ-consumed batch and the BatchMeta descriptor selecting its samples."""

    admission: PartitionAdmission
    data: Any
    batch_meta: Any
    sample_tags: tuple[dict[str, Any], ...] = ()


class RolloutOutput(TypedDict, total=False):
    logprobs: list[float]
    rewards: float
    completion_length: int
    generation_idx: int
    rollout_policy_version: int
    rollout_adapter_path: str | None
    rollout_policy_versions: list[int]
    initial_policy_version: int
    final_policy_version: int
    policy_version_span: int
