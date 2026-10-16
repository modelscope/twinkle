# Copyright (c) ModelScope Contributors. All rights reserved.
"""Worker-local metric buffering."""

from __future__ import annotations

import threading

from .types import MetricRecord


class MetricBuffer:
    """Thread-safe, destructive worker-local metric buffer."""

    def __init__(self):
        self._records: list[MetricRecord] = []
        self._lock = threading.Lock()

    def record(self, record: MetricRecord) -> None:
        with self._lock:
            self._records.append(record)

    def drain(self) -> list[MetricRecord]:
        with self._lock:
            records = self._records
            self._records = []
        return records
