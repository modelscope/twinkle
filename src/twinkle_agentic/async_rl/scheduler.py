# Copyright (c) ModelScope Contributors. All rights reserved.
"""Context selection policies used independently by rollout/advantage/training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Sequence

from .types import LoraContext, PartitionAdmission


class ContextSchedulePolicy(StrEnum):
    ROUND_ROBIN = 'round_robin'
    STICKY = 'sticky'
    OLDEST_PARTITION = 'oldest_partition'


@dataclass(frozen=True)
class SchedulerConfig:
    policy: ContextSchedulePolicy = ContextSchedulePolicy.ROUND_ROBIN
    max_consecutive_units: int | None = 1


@dataclass(frozen=True)
class ScheduleCandidate:
    context: LoraContext
    partition: PartitionAdmission | None = None


class ContextScheduler:

    def __init__(self, config: SchedulerConfig):
        self.config = config
        self._cursor = 0
        self._sticky_key: str | None = None
        self._consecutive = 0

    def choose(self, candidates: Sequence[ScheduleCandidate]) -> ScheduleCandidate | None:
        if not candidates:
            return None
        if self.config.policy is ContextSchedulePolicy.OLDEST_PARTITION:
            return min(
                candidates,
                key=lambda item: (item.partition.created_order if item.partition else float('inf'), item.context.key))
        if self.config.policy is ContextSchedulePolicy.STICKY and self._sticky_key is not None:
            cap = self.config.max_consecutive_units
            if cap is None or self._consecutive < cap:
                for candidate in candidates:
                    if candidate.context.key == self._sticky_key:
                        return candidate
            else:
                for candidate in candidates:
                    if candidate.context.key != self._sticky_key:
                        return candidate
        index = self._cursor % len(candidates)
        return candidates[index]

    def on_success(self, candidate: ScheduleCandidate) -> None:
        if self.config.policy is ContextSchedulePolicy.STICKY and candidate.context.key == self._sticky_key:
            self._consecutive += 1
            return
        self._sticky_key = candidate.context.key
        self._consecutive = 1
        if self.config.policy is ContextSchedulePolicy.ROUND_ROBIN:
            self._cursor += 1

    def on_blocked(self, candidate: ScheduleCandidate) -> None:
        if candidate.context.key == self._sticky_key:
            self._sticky_key = None
            self._consecutive = 0
        if self.config.policy is ContextSchedulePolicy.ROUND_ROBIN:
            self._cursor += 1
