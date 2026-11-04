"""Framework-independent rollout iteration recording and future observation."""

import json
from types import SimpleNamespace as NS

import pytest

from twinkle.perf import RLPerformanceRecorder, reduce_memory, sample_lengths


def test_plain_lengths_and_update_numbers_need_no_training_objects(tmp_path):
    now = [0.0]
    perf = RLPerformanceRecorder(tmp_path, [("train", [0, 1]), ("shared", [0])], clock=lambda: now[0])
    perf.begin(4)
    perf.received_lengths([3, 4], [2, 3], truncated=[False, True], assistant_turns=[1, 2])
    perf.record_training([0, 1, 0, 1])
    now[0] = 2.0
    perf.finish(6, publications=1, resource_metrics={"resource/replicas": 2})
    perf.write_summary(6)
    perf.close()
    row = json.loads((tmp_path / "performance.jsonl").read_text())
    assert row["data"]["perf/throughput"] == 3
    assert row["data"]["training/sample_visits"] == 4
    assert row["data"]["training/unique_samples_trained"] == 2
    assert row["data"]["training/optimizer_steps_this_iteration"] == 2
    assert row["data"]["training/weight_publications_this_iteration"] == 1
    assert row["data"]["response/truncated_ratio"] == 0.5
    assert row["data"]["resource/replicas"] == 2


def test_plain_snapshot_publications_use_actual_counter_deltas(tmp_path):
    now = [0.0]
    perf = RLPerformanceRecorder(tmp_path, [("train", [0])], clock=lambda: now[0], track_batch_wait=True)
    for step, publish_time, publications in [(1, 0.5, 1), (2, 0.5, 1)]:
        perf.begin(step - 1)
        now[0] += 1
        perf.received_lengths([3], [2])
        now[0] += 1
        perf.finish(step, stream_stats={"publish_time": publish_time, "publishes": publications})
    perf.close()
    rows = [json.loads(line)["data"] for line in (tmp_path / "performance.jsonl").read_text().splitlines()]
    assert [row["training/weight_publications_this_iteration"] for row in rows] == [1, 0]
    assert [row["timing_s/update_weights"] for row in rows] == [0.5, 0]
    assert all(row["timing_s/gen"] == 1 for row in rows)


def sample(prompt=3, response=2, *, tool=0):
    ids = list(range(prompt + response + tool))
    labels = [-100] * prompt + ids[prompt : prompt + response] + [-100] * tool
    return NS(encoded={"input_ids": ids, "labels": labels[1:] + labels[:1]}, truncated=False)


def test_lengths_include_tool_observations_but_not_padding():
    s = sample(tool=4)
    assert sample_lengths(s) == (3, 6)
    s.encoded["input_ids"] += [0, 0]
    # Output-order labels for 3 prompt, 2 assistant, 4 observation and 2 padding tokens.
    labels = [-100] * 3 + [3, 4] + [-100] * 6
    s.encoded["labels"] = labels[1:] + labels[:1]
    s.encoded["attention_mask"] = [1] * 9 + [0, 0]
    assert sample_lengths(s) == (3, 6)
    assert sample_lengths(sample(response=0)) == (3, 0)


def test_iteration_counts_tokens_once_across_optimizer_replay_and_flushes(tmp_path):
    now = [10.0]
    perf = RLPerformanceRecorder(
        tmp_path, [("trainer", [0, 1]), ("shared", [0]), ("rollout", [2, 3])], mode="colocate", clock=lambda: now[0]
    )
    perf.begin(7)
    perf.received([sample(), sample(prompt=4, response=3)])
    perf.sample_visits = 8  # four epochs over two samples
    perf.trained_indices = {0, 1}
    now[0] = 12.0
    perf.finish(11)
    row = json.loads((tmp_path / "performance.jsonl").read_text())  # before close
    assert row["step_unit"] == "rl_iteration"
    assert row["data"]["perf/total_num_tokens"] == 12
    assert row["data"]["perf/throughput"] == 1.5  # 12 tokens / 2s / 4 physical GPUs
    assert row["data"]["training/optimizer_steps_this_iteration"] == 4
    assert row["data"]["training/sample_visits"] == 8
    perf.close()


def test_remote_stage_waits_without_collecting_payload(tmp_path):
    now = [0.0]
    perf = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="colocate", clock=lambda: now[0])
    perf.begin(0)

    class Future:
        _is_lazy_collect = True

        def wait(self):
            now[0] += 3.0

        def __call__(self):
            pytest.fail("must not transfer tensor payloads to the driver")

    future = Future()
    assert perf.call("ref", lambda: future) is future
    assert perf.timings["ref"] == 3.0


def test_memory_reduction_is_rank_max_for_peaks_host_mean_for_ram():
    result = reduce_memory(
        [
            {"perf/max_memory_allocated_gb": 3, "perf/cpu_memory_used_gb": 8},
            {"perf/max_memory_allocated_gb": 5, "perf/cpu_memory_used_gb": 10},
        ]
    )
    assert result == {"perf/max_memory_allocated_gb": 5, "perf/cpu_memory_used_gb": 9}


def test_side_effect_future_failure_is_not_a_successful_update(tmp_path):
    perf = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="colocate")
    perf.begin(0)

    class FailedUpdate:
        _is_lazy_collect = True

        def wait(self):
            return self

        def result(self):
            raise RuntimeError("optimizer failed")

    with pytest.raises(RuntimeError, match="optimizer failed"):
        perf.call("optimizer", FailedUpdate, collect_result=True)


def test_reporter_time_excluded_but_full_wall_time_is_saved(tmp_path):
    now = [0.0]
    perf = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="colocate", clock=lambda: now[0])
    perf.begin(0)
    perf.received([sample()])
    now[0] = 2.0
    with perf.exclude_logging():
        now[0] = 5.0
    now[0] = 7.0
    perf.finish(1)
    data = json.loads((tmp_path / "performance.jsonl").read_text())["data"]
    assert data["perf/time_per_step"] == 4.0
    assert data["observability/iteration_wall_seconds"] == 7.0
    assert data["observability/excluded_logging_seconds"] == 3.0
    perf.close()


def test_auxiliary_forwards_keep_original_parallel_submission(tmp_path):
    import threading

    ready = threading.Event()
    launches = []
    perf = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="colocate")
    perf.begin(0)

    class Forward:
        _is_lazy_collect = True

        def wait(self):
            assert ready.wait(2), "forwards were serialized before all lanes could launch"

        def __call__(self):
            pytest.fail("do not collect tensor payloads")

    def submit(name):
        launches.append(name)
        if len(launches) == 3:
            ready.set()
        return Forward()

    try:
        for name in ("ref", "teacher", "old_log_prob"):
            perf.launch_forward(name, submit, name)
        perf.join_forwards()
        assert set(perf.timings) == {"ref", "teacher", "old_log_prob"}
        assert all(value > 0 for value in perf.timings.values())
    finally:
        ready.set()
        perf.close()


def test_summary_records_terminal_driver_counters_after_teardown(tmp_path):
    perf = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="adaptive", parameter_sync_step=2)
    perf.begin(0)
    perf.received([sample()])
    stats = {"publish_time": 0.0, "publishes": 0, "dropped_incomplete": 0, "total_time": 0.0, "final_version": 0}
    perf.finish(1, stream_stats=stats)
    stats.update(dropped_incomplete=3, total_time=12.0, final_version=2)
    perf.write_summary(1, stream_stats=stats, status="loop_completed")
    summary = json.loads((tmp_path / "performance_summary.jsonl").read_text())
    row = json.loads((tmp_path / "performance.jsonl").read_text())
    assert summary["run_id"] == row["run_id"]
    assert summary["data"]["stream/dropped_incomplete"] == 3
    assert summary["data"]["stream/final_version"] == 2
    assert summary["data"]["stream/total_time"] == 12.0
    assert "stream/total_time" not in row["data"]
    assert "stream/dropped_incomplete" not in row["data"]
    assert row["parameter_sync_step_unit"] == "optimizer_update"
    assert row["iteration_scope"] == "one_consumed_rollout_batch"
    perf.close()


def test_assistant_turn_count_has_an_explicit_unit(tmp_path):
    perf = RLPerformanceRecorder(tmp_path, [("trainer", [0])], mode="colocate")
    perf.begin(0)
    s = sample()
    s.rollout_infos = {"turns": 1}  # Count assistant generations independently of user/tool messages.
    perf.received([s])
    perf.finish(1)
    data = json.loads((tmp_path / "performance.jsonl").read_text())["data"]
    assert data["rollout/assistant_turns/mean"] == 1
    assert "num_turns/mean" not in data
    perf.close()
