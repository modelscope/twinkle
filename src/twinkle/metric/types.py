# Copyright (c) ModelScope Contributors. All rights reserved.
"""Transport-neutral metric value types."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

METRIC_STAGES = frozenset({
    'rollout',
    'advantage',
    'train',
    'evaluation',
    'partition',
    'policy',
    'run',
})
METRIC_STATUSES = frozenset({'submitted', 'completed', 'failed'})


@dataclass(frozen=True)
class MetricRecord:
    stage: str
    values: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    sequence: int | None = None
    context_key: str | None = None
    partition_id: str | None = None
    partition_index: int | None = None
    optimizer_step: int | None = None
    policy_version: int | None = None
    status: str = 'completed'
    attributes: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in METRIC_STAGES:
            raise ValueError(f'unsupported metric stage: {self.stage!r}')
        if self.status not in METRIC_STATUSES:
            raise ValueError(f'unsupported metric status: {self.status!r}')
        object.__setattr__(self, 'values', dict(self.values))
        object.__setattr__(self, 'attributes', dict(self.attributes))
