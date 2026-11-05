"""Existing metric output supports resumed runs and raw observation attributes."""

import json
from concurrent.futures import ThreadPoolExecutor

from twinkle.metric import MetricRecord, create_metrics_reporter


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
