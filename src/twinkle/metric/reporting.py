# Copyright (c) ModelScope Contributors. All rights reserved.
"""Asynchronous JSONL and SwanLab metric reporting."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import time
from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .types import MetricRecord

logger = logging.getLogger(__name__)


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        match = re.fullmatch(r'\s*tensor\(([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\)\s*', value)
        candidate = match.group(1) if match else value
        try:
            number = float(candidate)
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    try:
        scalar = value.item()
    except (AttributeError, RuntimeError, ValueError):
        return None
    return _finite_number(scalar)


def _safe_name(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', value).strip('_') or 'default'


@dataclass
class _ScalarSummary:
    count: int = 0
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf
    last: float = 0.0

    def add(self, value: float | int) -> None:
        number = float(value)
        self.count += 1
        self.total += number
        self.minimum = min(self.minimum, number)
        self.maximum = max(self.maximum, number)
        self.last = number

    def as_dict(self) -> dict[str, float | int]:
        return {
            'count': self.count,
            'last': self.last,
            'mean': self.total / self.count,
            'min': self.minimum,
            'max': self.maximum,
        }


class _SummaryReducer:

    def __init__(self, run_id: str, started_at: float):
        self.run_id = run_id
        self.started_at = started_at
        self.record_counts: Counter[str] = Counter()
        self.context_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.context_rollout_groups: Counter[str] = Counter()
        self.context_rollout_samples: Counter[str] = Counter()
        self.context_trained_samples: Counter[str] = Counter()
        self.context_optimizer_steps: dict[str, int] = {}
        self.context_policy_versions: dict[str, int] = {}
        self.metric_summaries: dict[str, _ScalarSummary] = defaultdict(_ScalarSummary)
        self.run_status = 'running'
        self.result: dict[str, Any] = {}

    def add(self, record: MetricRecord) -> None:
        record_key = f'{record.stage}:{record.status}'
        self.record_counts[record_key] += 1
        if record.context_key:
            counts = self.context_counts[record.context_key]
            counts[record_key] += 1
            if record.optimizer_step is not None:
                self.context_optimizer_steps[record.context_key] = record.optimizer_step
            if record.policy_version is not None:
                previous_version = self.context_policy_versions.get(record.context_key)
                self.context_policy_versions[record.context_key] = (
                    record.policy_version if previous_version is None else max(previous_version, record.policy_version))
            sample_count = _finite_number(record.values.get('sample_count'))
            if sample_count is not None and record.status == 'completed':
                if record.stage == 'rollout' and record.attributes.get('scope', 'group') == 'group':
                    self.context_rollout_groups[record.context_key] += 1
                    self.context_rollout_samples[record.context_key] += int(sample_count)
                elif record.stage == 'train':
                    self.context_trained_samples[record.context_key] += int(sample_count)
        summarize_values = (
            record.status == 'completed'
            and (record.stage != 'rollout' or record.attributes.get('scope', 'group') == 'group'))
        if summarize_values:
            for name, value in record.values.items():
                number = _finite_number(value)
                if number is not None:
                    self.metric_summaries[f'{record.stage}/{name}'].add(number)
        if record.stage == 'run':
            self.run_status = record.status
            self.result = {**record.values, **record.attributes}

    def as_dict(self, backend_health: Mapping[str, Any]) -> dict[str, Any]:
        wall_time = _finite_number(self.result.get('wall_time_s'))
        if wall_time is None:
            wall_time = time.time() - self.started_at
        rollout_groups = sum(self.context_rollout_groups.values())
        train_steps = sum(counts['train:completed'] for counts in self.context_counts.values())
        trained_partitions = sum(counts['partition:completed'] for counts in self.context_counts.values())
        terminal_partitions = _finite_number(self.result.get('trained_partitions'))
        if terminal_partitions is not None:
            trained_partitions = int(terminal_partitions)
        rollout_samples = sum(self.context_rollout_samples.values())
        trained_samples = sum(self.context_trained_samples.values())
        dropped_records = sum(int(item.get('dropped_records', 0)) for item in backend_health.values())
        backend_write_latency_s = sum(float(item.get('write_latency_s', 0.0)) for item in backend_health.values())
        contexts = {}
        for context_key, counts in self.context_counts.items():
            contexts[context_key] = {
                'rollout_groups': self.context_rollout_groups[context_key],
                'rollout_samples': self.context_rollout_samples[context_key],
                'train_steps': counts['train:completed'],
                'trained_samples': self.context_trained_samples[context_key],
                'trained_partitions': counts['partition:completed'],
                'optimizer_step': self.context_optimizer_steps.get(context_key),
                'policy_version': self.context_policy_versions.get(context_key),
            }
        return {
            'run_id': self.run_id,
            'status': self.run_status,
            'wall_time_s': wall_time,
            'record_counts': dict(sorted(self.record_counts.items())),
            'rollout_groups': rollout_groups,
            'rollout_samples': rollout_samples,
            'train_steps': train_steps,
            'trained_samples': trained_samples,
            'trained_partitions': trained_partitions,
            'rollout_groups_per_sec': rollout_groups / wall_time if wall_time > 0 else 0.0,
            'rollout_samples_per_sec': rollout_samples / wall_time if wall_time > 0 else 0.0,
            'train_steps_per_hour': train_steps * 3600 / wall_time if wall_time > 0 else 0.0,
            'trained_samples_per_sec': trained_samples / wall_time if wall_time > 0 else 0.0,
            'train_partitions_per_hour': trained_partitions * 3600 / wall_time if wall_time > 0 else 0.0,
            'dropped_records': dropped_records,
            'backend_write_latency_s': backend_write_latency_s,
            'per_context': contexts,
            'metrics': {
                name: summary.as_dict()
                for name, summary in sorted(self.metric_summaries.items())
            },
            'backends': dict(backend_health),
            'result': self.result,
        }


class _QueuedBackend:

    def __init__(
        self,
        name: str,
        *,
        queue_capacity: int,
        batch_size: int,
        flush_interval_s: float,
    ):
        if queue_capacity <= 0:
            raise ValueError('queue_capacity must be positive')
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        if flush_interval_s <= 0:
            raise ValueError('flush_interval_s must be positive')
        self.name = name
        self.queue_capacity = queue_capacity
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._queue: deque[dict[str, Any]] = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._flush_requested = False
        self._closed = False
        self._disabled = False
        self._in_flight = False
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._write_batches = 0
        self._write_latency_s = 0.0
        self._failures = 0
        self._last_error: str | None = None
        self._warning_emitted = False
        self._thread = threading.Thread(target=self._run, name=f'twinkle-metrics-{name}', daemon=True)
        self._thread.start()

    def submit(self, payload: dict[str, Any]) -> None:
        with self._condition:
            if self._closing or self._disabled:
                self._dropped += 1
                return
            if len(self._queue) >= self.queue_capacity:
                self._queue.popleft()
                self._dropped += 1
            self._queue.append(payload)
            self._submitted += 1
            self._condition.notify()

    def flush(self, timeout_s: float | None = None) -> bool:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._condition:
            self._flush_requested = True
            self._condition.notify_all()
            while (self._queue or self._in_flight) and not self._disabled:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining)
            self._flush_requested = False
        return True

    def close(self, timeout_s: float | None = None) -> bool:
        with self._condition:
            if self._closed:
                return True
            self._closing = True
            self._condition.notify_all()
        self._thread.join(timeout_s)
        closed = not self._thread.is_alive()
        if closed:
            self._closed = True
        return closed

    def health(self) -> dict[str, Any]:
        with self._condition:
            return {
                'enabled': not self._disabled,
                'queue_size': len(self._queue),
                'submitted_records': self._submitted,
                'written_records': self._written,
                'dropped_records': self._dropped,
                'write_batches': self._write_batches,
                'write_latency_s': self._write_latency_s,
                'failure_count': self._failures,
                'last_error': self._last_error,
            }

    def _run(self) -> None:
        last_write = time.monotonic()
        while True:
            with self._condition:
                while True:
                    if self._closing and not self._queue:
                        batch = []
                        break
                    elapsed = time.monotonic() - last_write
                    should_write = bool(self._queue) and (self._closing or self._flush_requested or len(self._queue)
                                                          >= self.batch_size or elapsed >= self.flush_interval_s)
                    if should_write:
                        batch = [self._queue.popleft() for _ in range(min(len(self._queue), self.batch_size))]
                        break
                    wait_s = (max(0.0, self.flush_interval_s - elapsed) if self._queue else self.flush_interval_s)
                    self._condition.wait(wait_s)
                if not batch:
                    break
                self._in_flight = True
            started = time.perf_counter()
            try:
                self._write_batch(batch)
            except Exception as exc:
                with self._condition:
                    self._failures += 1
                    self._last_error = f'{type(exc).__name__}: {exc}'
                    self._dropped += len(batch) + len(self._queue)
                    self._queue.clear()
                    self._disabled = True
                if not self._warning_emitted:
                    logger.warning('Metrics backend %s failed and was disabled: %s', self.name, exc)
                    self._warning_emitted = True
            else:
                elapsed = time.perf_counter() - started
                with self._condition:
                    self._written += len(batch)
                    self._write_batches += 1
                    self._write_latency_s += elapsed
            finally:
                last_write = time.monotonic()
                with self._condition:
                    self._in_flight = False
                    self._condition.notify_all()
            if self._disabled:
                break
        try:
            self._close_sink()
        except Exception as exc:
            with self._condition:
                self._failures += 1
                self._last_error = f'{type(exc).__name__}: {exc}'
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def _write_batch(self, batch: Sequence[dict[str, Any]]) -> None:
        raise NotImplementedError

    def _close_sink(self) -> None:
        pass


class _JSONLBackend(_QueuedBackend):

    def __init__(self, path: str | Path, **kwargs: Any):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open('w', encoding='utf-8')
        super().__init__('jsonl', **kwargs)

    def _write_batch(self, batch: Sequence[dict[str, Any]]) -> None:
        self._stream.writelines(json.dumps(payload, ensure_ascii=True, default=str) + '\n' for payload in batch)
        self._stream.flush()

    def _close_sink(self) -> None:
        self._stream.close()


class _SwanLabBackend(_QueuedBackend):

    def __init__(
        self,
        *,
        project: str,
        experiment_name: str,
        log_dir: str | Path,
        mode: str,
        **kwargs: Any,
    ):
        import swanlab

        self._swanlab = swanlab
        self._swanlab_run = swanlab.init(
            project=project,
            experiment_name=experiment_name,
            logdir=str(log_dir),
            mode=mode,
        )
        super().__init__('swanlab', **kwargs)

    def _write_batch(self, batch: Sequence[dict[str, Any]]) -> None:
        for payload in batch:
            prefix = (f'context/{_safe_name(payload["context_key"])}' if payload.get('context_key') else 'global')
            stage = _safe_name(payload['stage'])
            values = {}
            for name, value in payload['values'].items():
                metric_name = str(name)
                if metric_name.startswith(f'{stage}/'):
                    metric_name = metric_name[len(stage) + 1:]
                values[f'{prefix}/{stage}/{_safe_name(metric_name)}'] = value
            if payload.get('optimizer_step') is not None:
                values[f'{prefix}/train/optimizer_step'] = payload['optimizer_step']
            if payload.get('policy_version') is not None:
                values[f'{prefix}/policy/version'] = payload['policy_version']
            if payload.get('partition_index') is not None:
                values[f'{prefix}/partition/index'] = payload['partition_index']
            if values:
                self._swanlab_run.log(values, step=payload['sequence'])

    def _close_sink(self) -> None:
        self._swanlab.finish()


class MetricsReporter:
    """Assign ordering and asynchronously fan metric records out to backends."""

    def __init__(
        self,
        *,
        run_id: str,
        backends: Sequence[_QueuedBackend] = (),
        summary_path: str | Path | None = None,
        close_timeout_s: float = 10.0,
    ):
        self.run_id = run_id
        self.started_at = time.time()
        self.close_timeout_s = close_timeout_s
        self.summary_path = Path(summary_path) if summary_path is not None else None
        self._backends = tuple(backends)
        self._lock = threading.Lock()
        self._sequence = 0
        self._closed = False
        self._reducer = _SummaryReducer(run_id, self.started_at)
        self._initial_errors: dict[str, str] = {}

    def add_backend_error(self, name: str, error: BaseException) -> None:
        self._initial_errors[name] = f'{type(error).__name__}: {error}'

    def record(self, record: MetricRecord) -> None:
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            normalized = self._normalize(record, self._sequence)
            self._reducer.add(normalized)
            payload = self._payload(normalized)
            for backend in self._backends:
                backend.submit(payload)

    def record_many(self, records: Sequence[MetricRecord]) -> None:
        for record in records:
            self.record(record)

    def flush(self, timeout_s: float | None = None) -> None:
        timeout = self.close_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + timeout
        for backend in self._backends:
            backend.flush(max(0.0, deadline - time.monotonic()))

    def close(self, timeout_s: float | None = None) -> None:
        timeout = self.close_timeout_s if timeout_s is None else timeout_s
        with self._lock:
            if self._closed:
                return
            self._closed = True
        deadline = time.monotonic() + timeout
        for backend in self._backends:
            backend.close(max(0.0, deadline - time.monotonic()))
        self._write_summary()

    def health(self) -> dict[str, Any]:
        backend_health = {backend.name: backend.health() for backend in self._backends}
        for name, error in self._initial_errors.items():
            backend_health[name] = {
                'enabled': False,
                'failure_count': 1,
                'last_error': error,
                'dropped_records': self._sequence,
            }
        return {
            'record_count': self._sequence,
            'dropped_records': sum(int(item.get('dropped_records', 0)) for item in backend_health.values()),
            'backends': backend_health,
        }

    def summary(self) -> dict[str, Any]:
        return self._reducer.as_dict(self.health()['backends'])

    def _normalize(self, record: MetricRecord, sequence: int) -> MetricRecord:
        values: dict[str, float | int] = {}
        non_numeric = {}
        for name, value in record.values.items():
            number = _finite_number(value)
            if number is None:
                non_numeric[name] = value
            else:
                values[name] = number
        attributes = dict(record.attributes)
        if non_numeric:
            attributes['non_numeric_values'] = non_numeric
        return replace(record, sequence=sequence, values=values, attributes=attributes)

    def _payload(self, record: MetricRecord) -> dict[str, Any]:
        return {
            'timestamp': record.timestamp,
            'elapsed_s': record.timestamp - self.started_at,
            'sequence': record.sequence,
            'run_id': self.run_id,
            'stage': record.stage,
            'context_key': record.context_key,
            'partition_id': record.partition_id,
            'partition_index': record.partition_index,
            'optimizer_step': record.optimizer_step,
            'policy_version': record.policy_version,
            'status': record.status,
            'values': record.values,
            'attributes': record.attributes,
        }

    def _write_summary(self) -> None:
        if self.summary_path is None:
            return
        try:
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = self.summary_path.with_suffix(f'{self.summary_path.suffix}.tmp')
            with temporary_path.open('w', encoding='utf-8') as stream:
                json.dump(self.summary(), stream, ensure_ascii=True, indent=2, default=str)
                stream.write('\n')
            os.replace(temporary_path, self.summary_path)
        except Exception as exc:
            logger.warning('Failed to write metrics summary %s: %s', self.summary_path, exc)


def create_metrics_reporter(config: Mapping[str, Any] | None, *, run_id: str) -> MetricsReporter | None:
    if config is None:
        return None
    config = dict(config or {})
    if not bool(config.get('enabled', True)):
        return None
    queue_capacity = int(config.get('queue_capacity', 10000))
    close_timeout_s = float(config.get('close_timeout_s', 10.0))
    jsonl_config = dict(config.get('jsonl') or {})
    swanlab_config = dict(config.get('swanlab') or {})
    backends: list[_QueuedBackend] = []
    backend_errors: dict[str, BaseException] = {}
    backend_defaults = {
        'queue_capacity': queue_capacity,
        'batch_size': int(jsonl_config.get('batch_size', 64)),
        'flush_interval_s': float(jsonl_config.get('flush_interval_s', 2.0)),
    }
    if bool(jsonl_config.get('enabled', True)):
        try:
            backends.append(_JSONLBackend(jsonl_config['path'], **backend_defaults))
        except Exception as exc:
            backend_errors['jsonl'] = exc
            logger.warning('JSONL metrics backend could not start: %s', exc)
    if bool(swanlab_config.get('enabled', False)) and swanlab_config.get('mode') != 'disabled':
        try:
            backends.append(
                _SwanLabBackend(
                    project=str(swanlab_config.get('project', 'twinkle-rl')),
                    experiment_name=str(swanlab_config.get('name', run_id)),
                    log_dir=swanlab_config.get('log_dir', 'outputs/swanlab'),
                    mode=str(swanlab_config.get('mode', 'local')),
                    queue_capacity=queue_capacity,
                    batch_size=int(swanlab_config.get('batch_size', 16)),
                    flush_interval_s=float(swanlab_config.get('flush_interval_s', 1.0)),
                ))
        except Exception as exc:
            backend_errors['swanlab'] = exc
            logger.warning('SwanLab metrics backend could not start: %s', exc)
    summary_path = jsonl_config.get('summary_path')
    if summary_path is None and jsonl_config.get('path') is not None:
        summary_path = Path(jsonl_config['path']).with_name('summary.json')
    reporter = MetricsReporter(
        run_id=run_id,
        backends=backends,
        summary_path=summary_path,
        close_timeout_s=close_timeout_s,
    )
    for name, error in backend_errors.items():
        reporter.add_backend_error(name, error)
    return reporter
