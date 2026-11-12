"""Existing metric output supports resumed runs and raw observation attributes."""

import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

from twinkle.metric import MetricRecord, create_metrics_reporter
from twinkle.metric.reporting import MetricsReporter, _QueuedBackend


def reporter(path, run_id):
    return create_metrics_reporter({"jsonl": {
        "path": path, "summary_path": path.with_name("summary.json"),
        "append": True, "batch_size": 1,
    }}, run_id=run_id)


def test_existing_reporter_flushes_raw_observations_and_nonfinite_values(tmp_path):
    path = tmp_path / "metrics.jsonl"
    output = reporter(path, "run-1")
    output.record(MetricRecord(
        stage="train", optimizer_step=2, values={"tokens": 19, "loss": float("nan")},
        attributes={"memory_by_rank": [{"allocated_bytes": 4096}], "timing_spans": [(1, 2, 3)]},
    ))
    output.flush()
    row = json.loads(path.read_text())
    assert row["values"] == {"tokens": 19}
    assert row["attributes"]["non_numeric_values"]["loss"] is None
    assert row["attributes"]["memory_by_rank"] == [{"allocated_bytes": 4096}]
    assert row["attributes"]["timing_spans"] == [[1, 2, 3]]
    assert row["run_id"] == "run-1" and row["optimizer_step"] == 2
    output.close()


def test_append_does_not_overwrite_previous_run(tmp_path):
    path = tmp_path / "metrics.jsonl"
    for run_id in ("first", "second"):
        output = reporter(path, run_id)
        output.record(MetricRecord(stage="train", optimizer_step=1, values={"loss": 0.5}))
        output.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["run_id"] for row in rows] == ["first", "second"]


def test_existing_reporter_orders_concurrent_records_and_flushes_on_close(tmp_path):
    path = tmp_path / "metrics.jsonl"
    output = reporter(path, "run-1")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda step: output.record(MetricRecord(stage="train", optimizer_step=step, values={"x": step})), range(32)))
    output.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["sequence"] for row in rows] == list(range(1, 33))
    assert sorted(row["optimizer_step"] for row in rows) == list(range(32))
    assert output.health()["dropped_records"] == 0


def test_terminal_totals_are_used_when_producer_reports_batch_scopes(tmp_path):
    output = reporter(tmp_path / "metrics.jsonl", "run-1")
    output.record(MetricRecord(stage="train", context_key="run-1", optimizer_step=1, values={"loss": 0.5}))
    output.record(MetricRecord(
        stage="rollout", context_key="run-1", values={"sample_count": 64}, attributes={"scope": "iteration"},
    ))
    output.record(MetricRecord(
        stage="run", context_key="run-1", values={"rollout_groups": 8, "rollout_samples": 64, "trained_samples": 128},
    ))
    output.close()
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["train_steps"] == 1
    assert summary["rollout_groups"] == 8
    assert summary["rollout_samples"] == 64
    assert summary["trained_samples"] == 128


@pytest.mark.parametrize("config", [None, {"enabled": False}])
def test_disabled_configuration_returns_no_reporter(config):
    assert create_metrics_reporter(config, run_id="run-1") is None


def test_reporter_without_backends_counts_records_and_builds_summary():
    output = create_metrics_reporter({"jsonl": {"enabled": False}}, run_id="run-1")
    assert output is not None
    output.record_many([
        MetricRecord(stage="train", values={"loss": 1.0}),
        MetricRecord(stage="train", values={"loss": 3.0}),
    ])
    assert output.health()["record_count"] == 2
    summary = output.summary()
    assert summary["metrics"]["train/loss"]["count"] == 2
    assert summary["metrics"]["train/loss"]["mean"] == 2.0
    output.close()


def test_close_is_idempotent_and_records_after_close_are_ignored():
    output = MetricsReporter(run_id="run-1")
    output.record(MetricRecord(stage="train", values={"loss": 1.0}))
    output.close()
    output.close()
    output.record(MetricRecord(stage="train", values={"loss": 2.0}))
    assert output.health()["record_count"] == 1


def test_flush_times_out_while_a_backend_write_is_blocked():
    class BlockingBackend(_QueuedBackend):

        def __init__(self, **kwargs):
            self.started = threading.Event()
            self.release = threading.Event()
            super().__init__("blocking", **kwargs)

        def _write_batch(self, batch):
            self.started.set()
            self.release.wait(5)

    backend = BlockingBackend(queue_capacity=4, batch_size=1, flush_interval_s=0.01)
    backend.submit({"sequence": 1})
    assert backend.started.wait(2)
    assert backend.flush(timeout_s=0.05) is False
    backend.release.set()
    assert backend.close(timeout_s=5) is True


@pytest.mark.parametrize("configuration", [
    {"queue_capacity": 0, "batch_size": 1, "flush_interval_s": 1.0},
    {"queue_capacity": 1, "batch_size": 0, "flush_interval_s": 1.0},
    {"queue_capacity": 1, "batch_size": 1, "flush_interval_s": 0.0},
])
def test_backend_configuration_must_be_positive(configuration):
    with pytest.raises(ValueError):
        _QueuedBackend("bad", **configuration)


def test_swanlab_disabled_mode_starts_no_backend(tmp_path):
    output = create_metrics_reporter({
        "jsonl": {"path": tmp_path / "metrics.jsonl"},
        "swanlab": {"enabled": True, "mode": "disabled"},
    }, run_id="run-1")
    assert "swanlab" not in output.health()["backends"]
    output.close()


def test_summary_write_failure_is_nonfatal(tmp_path):
    # summary_path pointing at a directory makes the atomic replace fail; the reporter must
    # swallow it (the JSONL stream is the durable artifact) and keep its in-memory summary.
    output = MetricsReporter(run_id="run-1", summary_path=tmp_path)
    output.record(MetricRecord(stage="train", values={"loss": 1.0}))
    output.close()
    assert output.summary()["record_counts"] == {"train:completed": 1}


def test_close_times_out_while_a_backend_write_is_blocked():
    class BlockingBackend(_QueuedBackend):

        def __init__(self, **kwargs):
            self.started = threading.Event()
            self.release = threading.Event()
            super().__init__("blocking", **kwargs)

        def _write_batch(self, batch):
            self.started.set()
            self.release.wait(5)

    backend = BlockingBackend(queue_capacity=4, batch_size=1, flush_interval_s=0.01)
    backend.submit({"sequence": 1})
    assert backend.started.wait(2)
    assert backend.close(timeout_s=0.05) is False
    backend.release.set()
    assert backend.close(timeout_s=5) is True


def test_swanlab_startup_failure_is_nonfatal(monkeypatch, tmp_path):
    def explode(**_kwargs):
        raise RuntimeError("swanlab unavailable")

    monkeypatch.setitem(sys.modules, "swanlab", types.SimpleNamespace(init=explode))
    output = create_metrics_reporter({
        "jsonl": {"path": tmp_path / "metrics.jsonl"},
        "swanlab": {"enabled": True, "mode": "local"},
    }, run_id="run-1")
    health = output.health()["backends"]["swanlab"]
    assert health["enabled"] is False
    assert health["failure_count"] == 1
    assert "swanlab unavailable" in health["last_error"]
    output.close()


def test_backend_close_failure_is_recorded():
    class BrokenCloseBackend(_QueuedBackend):

        def _write_batch(self, batch):
            pass

        def _close_sink(self):
            raise OSError("sink flush failed")

    backend = BrokenCloseBackend("broken", queue_capacity=4, batch_size=1, flush_interval_s=1.0)
    assert backend.close(timeout_s=5) is True
    health = backend.health()
    assert health["failure_count"] == 1
    assert "sink flush failed" in health["last_error"]


def test_swanlab_uses_global_prefix_and_skips_empty_payloads(monkeypatch):
    logged = []

    class Run:

        def log(self, values, step):
            logged.append((values, step))

    monkeypatch.setitem(
        sys.modules, "swanlab", types.SimpleNamespace(init=lambda **_kwargs: Run(), finish=lambda: None))
    output = create_metrics_reporter({
        "jsonl": {"enabled": False},
        "swanlab": {"enabled": True, "mode": "local", "batch_size": 1},
    }, run_id="run-1")
    output.record(MetricRecord(stage="train", values={"train/loss": 0.5}))
    output.record(MetricRecord(stage="train", values={}))
    output.close()
    assert len(logged) == 1
    assert logged[0][0] == {"global/train/loss": 0.5}
