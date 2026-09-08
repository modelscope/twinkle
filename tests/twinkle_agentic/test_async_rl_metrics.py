from __future__ import annotations

import json
import sys
import threading
import time
import types

import pytest

from twinkle.metric import (
    CompletionRewardMetric,
    MetricBuffer,
    MetricRecord,
    create_metrics_reporter,
)
from twinkle_agentic.async_rl.metrics import advantage_signal_metrics, rollout_metrics
from twinkle.metric.reporting import MetricsReporter, _QueuedBackend


def test_advantage_signal_metrics_report_zero_and_nonzero_groups():
    metrics = advantage_signal_metrics(
        rewards=[1.0, 1.0, 0.0, 1.0],
        advantages=[0.0, 0.0, -1.0, 1.0],
        num_generations=2,
    )
    assert metrics['group_count'] == 2
    assert metrics['group_reward_std_mean'] == pytest.approx(0.25)
    assert metrics['zero_advantage_group_ratio'] == pytest.approx(0.5)
    assert metrics['positive_advantage_ratio'] == pytest.approx(0.25)


def test_advantage_signal_metrics_reject_incomplete_groups():
    with pytest.raises(ValueError, match='complete groups'):
        advantage_signal_metrics([1.0, 0.0, 1.0], [1.0, -1.0, 0.0], num_generations=2)


def test_rollout_metrics_include_rewards_tokens_and_truncation():
    metrics = rollout_metrics(
        rewards={'accuracy': [1.0, 0.0]},
        completion_lengths=[3, 7],
        stop_reasons=['stop', 'length'],
        rollout_latency_s=2.0,
    )
    assert metrics == {
        'sample_count': 2,
        'completion_length_mean': 5.0,
        'completion_length_p95': 7,
        'completion_length_max': 7,
        'completion_truncated_count': 1,
        'completion_truncated_ratio': 0.5,
        'output_tokens': 10,
        'rollout_latency_s': 2.0,
        'output_tokens_per_s': 5.0,
        'accuracy_reward': 0.5,
        'accuracy_reward_std': pytest.approx(2**-0.5),
    }


def test_completion_reward_metric_preserves_model_metric_contract():
    metric = CompletionRewardMetric()
    metric.accumulate(
        rewards={'accuracy': [1.0, 0.0]},
        completion_lengths=[3, 7],
        generate_time=2.0,
        weight_sync_time=0.25,
    )
    result = metric.calculate()
    assert result == {
        'profiling/Time taken: move_model_to_sampler': 0.25,
        'profiling/Time taken: generate': 2.0,
        'train/accuracy_reward': 0.5,
        'train/accuracy_reward_std': pytest.approx(2**-0.5),
        'train/completion_length': 5.0,
    }


def test_metric_buffer_drain_is_atomic_and_destructive():
    buffer = MetricBuffer()

    def produce(start):
        for value in range(start, start + 50):
            buffer.record(MetricRecord(stage='train', values={'loss': value}))

    threads = [threading.Thread(target=produce, args=(index * 50,)) for index in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = buffer.drain()
    assert len(records) == 200
    assert buffer.drain() == []


def test_reporter_writes_new_jsonl_schema_and_summary(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    summary_path = tmp_path / 'summary.json'
    reporter = create_metrics_reporter({
        'queue_capacity': 100,
        'jsonl': {
            'path': path,
            'summary_path': summary_path,
            'batch_size': 2,
            'flush_interval_s': 60,
        },
    }, run_id='test')
    reporter.record(MetricRecord(
        stage='rollout',
        status='completed',
        context_key='tenant/run/adapter',
        partition_id='tenant/run/adapter/train_5',
        partition_index=5,
        policy_version=0,
        values={'sample_count': 4, 'reward': 0.5},
        attributes={'scope': 'group', 'group_id': 'group_0'},
    ))
    reporter.record(MetricRecord(
        stage='rollout',
        status='completed',
        context_key='tenant/run/adapter',
        partition_id='tenant/run/adapter/train_5',
        partition_index=5,
        policy_version=0,
        values={'sample_count': 4, 'reward': 0.25},
        attributes={'scope': 'partition'},
    ))
    reporter.record(MetricRecord(
        stage='train',
        context_key='tenant/run/adapter',
        partition_id='tenant/run/adapter/train_5',
        partition_index=5,
        optimizer_step=1,
        policy_version=0,
        values={'sample_count': 4, 'loss': '0.25'},
    ))
    reporter.record(MetricRecord(
        stage='partition',
        context_key='tenant/run/adapter',
        partition_index=5,
        policy_version=1,
        values={},
    ))
    reporter.record(MetricRecord(stage='run', values={'trained_partitions': 1, 'wall_time_s': 10.0}))
    reporter.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record['sequence'] for record in records] == [1, 2, 3, 4, 5]
    assert records[2]['stage'] == 'train'
    assert records[2]['optimizer_step'] == 1
    assert records[2]['values']['loss'] == 0.25
    assert 'event' not in records[2]
    summary = json.loads(summary_path.read_text())
    assert summary['status'] == 'completed'
    assert summary['trained_partitions'] == 1
    assert summary['per_context']['tenant/run/adapter']['optimizer_step'] == 1
    assert summary['metrics']['train/loss']['mean'] == 0.25
    assert summary['metrics']['rollout/reward']['count'] == 1
    assert summary['metrics']['rollout/reward']['mean'] == 0.5


def test_summary_policy_version_does_not_regress_for_out_of_order_records():
    reporter = MetricsReporter(run_id='test')
    reporter.record(MetricRecord(
        stage='policy',
        context_key='tenant/run/adapter',
        policy_version=5,
        values={},
    ))
    reporter.record(MetricRecord(
        stage='rollout',
        context_key='tenant/run/adapter',
        policy_version=3,
        values={'sample_count': 4},
        attributes={'scope': 'group'},
    ))

    assert reporter.summary()['per_context']['tenant/run/adapter']['policy_version'] == 5
    reporter.close()


def test_jsonl_backend_waits_for_batch_threshold(tmp_path):
    path = tmp_path / 'metrics.jsonl'
    reporter = create_metrics_reporter({
        'jsonl': {
            'path': path,
            'summary_path': tmp_path / 'summary.json',
            'batch_size': 2,
            'flush_interval_s': 60,
        },
    }, run_id='test')
    reporter.record(MetricRecord(stage='train', values={'loss': 1.0}))
    time.sleep(0.05)
    assert path.read_text() == ''
    reporter.record(MetricRecord(stage='train', values={'loss': 2.0}))
    deadline = time.monotonic() + 1
    while not path.read_text() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert len(path.read_text().splitlines()) == 2
    reporter.close()


def test_reporter_assigns_monotonic_sequence_across_threads(tmp_path):
    reporter = create_metrics_reporter({
        'jsonl': {
            'path': tmp_path / 'metrics.jsonl',
            'summary_path': tmp_path / 'summary.json',
        },
    }, run_id='test')

    def produce():
        for _ in range(25):
            reporter.record(MetricRecord(stage='train', values={'loss': 1.0}))

    threads = [threading.Thread(target=produce) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    reporter.close()
    records = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [record['sequence'] for record in records] == list(range(1, 101))


def test_swanlab_backend_uses_sequence_and_context_metric_names(monkeypatch, tmp_path):
    logged = []

    class Run:
        def log(self, values, step):
            logged.append((values, step))

    fake_swanlab = types.SimpleNamespace(init=lambda **_kwargs: Run(), finish=lambda: None)
    monkeypatch.setitem(sys.modules, 'swanlab', fake_swanlab)
    reporter = create_metrics_reporter({
        'jsonl': {'enabled': False},
        'swanlab': {
            'enabled': True,
            'mode': 'local',
            'project': 'test',
            'name': 'test-run',
            'log_dir': tmp_path,
            'batch_size': 1,
        },
    }, run_id='test')
    reporter.record(MetricRecord(
        stage='train',
        context_key='tenant/run/adapter',
        optimizer_step=3,
        policy_version=2,
        partition_index=1,
        values={'loss': 0.5},
    ))
    reporter.close()
    values, sequence = logged[0]
    assert sequence == 1
    assert values['context/tenant_run_adapter/train/loss'] == 0.5
    assert values['context/tenant_run_adapter/train/optimizer_step'] == 3
    assert values['context/tenant_run_adapter/policy/version'] == 2
    assert values['context/tenant_run_adapter/partition/index'] == 1


def test_backend_queue_drops_oldest_record_without_blocking_reporter():
    release = threading.Event()

    class BlockingBackend(_QueuedBackend):
        def _write_batch(self, batch):
            release.wait(2)

    backend = BlockingBackend(
        'blocking',
        queue_capacity=2,
        batch_size=2,
        flush_interval_s=60,
    )
    reporter = MetricsReporter(run_id='test', backends=[backend])
    for index in range(5):
        reporter.record(MetricRecord(stage='train', values={'loss': index}))
    assert reporter.health()['backends']['blocking']['dropped_records'] >= 1
    release.set()
    reporter.close()


def test_backend_failure_is_nonfatal_and_reported():
    class FailedBackend(_QueuedBackend):
        def _write_batch(self, batch):
            raise OSError('disk unavailable')

    backend = FailedBackend(
        'failed',
        queue_capacity=4,
        batch_size=1,
        flush_interval_s=1,
    )
    reporter = MetricsReporter(run_id='test', backends=[backend])
    reporter.record(MetricRecord(stage='train', values={'loss': 1.0}))
    reporter.flush()
    reporter.record(MetricRecord(stage='train', values={'loss': 2.0}))
    health = reporter.health()['backends']['failed']
    assert health['enabled'] is False
    assert health['failure_count'] == 1
    assert 'disk unavailable' in health['last_error']
    reporter.close()


def test_swanlab_failure_does_not_prevent_jsonl(monkeypatch, tmp_path):
    class FailedRun:
        def log(self, values, step):
            raise RuntimeError('swanlab unavailable')

    fake_swanlab = types.SimpleNamespace(init=lambda **_kwargs: FailedRun(), finish=lambda: None)
    monkeypatch.setitem(sys.modules, 'swanlab', fake_swanlab)
    path = tmp_path / 'metrics.jsonl'
    reporter = create_metrics_reporter({
        'jsonl': {
            'path': path,
            'summary_path': tmp_path / 'summary.json',
            'batch_size': 1,
        },
        'swanlab': {
            'enabled': True,
            'mode': 'local',
            'log_dir': tmp_path,
            'batch_size': 1,
        },
    }, run_id='test')
    reporter.record(MetricRecord(stage='train', values={'loss': 1.0}))
    reporter.close()
    assert len(path.read_text().splitlines()) == 1
    assert reporter.health()['backends']['swanlab']['failure_count'] == 1


def test_jsonl_startup_failure_is_nonfatal(tmp_path):
    reporter = create_metrics_reporter({
        'jsonl': {
            'path': tmp_path,
            'summary_path': tmp_path / 'summary.json',
        },
    }, run_id='test')
    reporter.record(MetricRecord(stage='train', values={'loss': 1.0}))
    health = reporter.health()
    assert health['record_count'] == 1
    assert health['backends']['jsonl']['enabled'] is False
    assert health['backends']['jsonl']['failure_count'] == 1
    reporter.close()
