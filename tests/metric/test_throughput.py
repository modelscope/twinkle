"""Numerical contracts for rollout accounting, padding and durable records."""

import math

import pytest
from twinkle.metric import ThroughputMetric, memory_metrics


def compute_performance_metrics(prompts, responses, timings, devices, **widths):
    metric = ThroughputMetric()
    metric.accumulate(prompt_lengths=prompts, response_lengths=responses, timings=timings, num_devices=devices)
    return metric.calculate(**widths)


@pytest.mark.parametrize("widths,clip,non_aborted_clip", [((None, None), 1 / 3, 0.5), ((12, 8), 0.0, 0.0)])
def test_variable_lengths_and_explicit_padded_width(widths, clip, non_aborted_clip):
    timings = {
        "step": 7.25,
        "gen": 3.5,
        "ref": 0.25,
        "old_log_prob": 0.4,
        "adv": 0.01,
        "update_actor": 2,
        "update_weights": 0.5,
    }
    actual = compute_performance_metrics(
        [3, 6, 4], [0, 2, 5], timings, 8, prompt_width=widths[0], response_width=widths[1]
    )
    expected = {
        "perf/total_num_tokens": 20,
        "perf/num_prompt_tokens": 13,
        "perf/num_response_tokens": 7,
        "perf/num_samples": 3,
        "perf/num_gpus": 8,
        "perf/time_per_step": 7.25,
        "perf/throughput": 0.3448275862068966,
        "timing_per_token_ms/gen": 500,
        "timing_per_token_ms/ref": 12.5,
        "timing_per_token_ms/adv": 0.5,
        "timing_per_token_ms/update_actor": 100,
        "prompt_length/mean": 13 / 3,
        "prompt_length/min": 3,
        "prompt_length/max": 6,
        "prompt_length/clip_ratio": clip,
        "response_length/mean": 7 / 3,
        "response_length/min": 0,
        "response_length/max": 5,
        "response_length/clip_ratio": clip,
        "response/aborted_ratio": 1 / 3,
        "response_length_non_aborted/mean": 3.5,
        "response_length_non_aborted/min": 2,
        "response_length_non_aborted/max": 5,
        "response_length_non_aborted/clip_ratio": non_aborted_clip,
    }
    for key, value in expected.items():
        assert actual[key] == pytest.approx(value), key
    assert "timing_per_token_ms/old_log_prob" not in actual


def test_all_empty_responses_have_no_non_aborted_population():
    actual = compute_performance_metrics([2, 3], [0, 0], {"step": 7.25, "gen": 3.5, "update_actor": 2}, 8)
    assert actual["perf/total_num_tokens"] == 5
    assert actual["perf/throughput"] == pytest.approx(0.08620689655172414)
    assert actual["response/aborted_ratio"] == 1
    assert actual["response_length/clip_ratio"] == 1
    assert actual["timing_per_token_ms/gen"] == 0
    assert actual["timing_per_token_ms/update_actor"] == 400
    assert math.isnan(actual["response_length_non_aborted/mean"])


def test_invalid_allocation_or_timing_fails_instead_of_faking_throughput():
    for n_gpus, seconds in [(0, 1), (2, 0), (2, float("nan"))]:
        with pytest.raises(ValueError):
            compute_performance_metrics([3], [2], {"step": seconds}, n_gpus)


def test_window_throughput_uses_sum_tokens_over_sum_seconds():
    metric = ThroughputMetric()
    metric.accumulate(prompt_lengths=[2], response_lengths=[8], timings={"step": 1}, num_devices=2)
    metric.accumulate(prompt_lengths=[2], response_lengths=[18], timings={"step": 4}, num_devices=2)
    assert metric.calculate()["perf/throughput"] == 3  # 30 / 5 / 2, not mean(5, 2.5)
    metric.reset()
    assert metric.calculate() == {}


def test_memory_reduction_uses_rank_max_for_peaks_and_mean_for_host_ram():
    assert memory_metrics(
        [
            {"max_memory_allocated_bytes": 3 * 1024**3, "cpu_memory_used_bytes": 8 * 1024**3},
            {"max_memory_allocated_bytes": 5 * 1024**3, "cpu_memory_used_bytes": 10 * 1024**3},
        ]
    ) == {"perf/max_memory_allocated_gb": 5, "perf/cpu_memory_used_gb": 9}
