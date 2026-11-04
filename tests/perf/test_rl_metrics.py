"""Numerical contracts for rollout accounting, padding and durable records."""

import json
import math

import pytest

from twinkle.perf import JsonlMetricWriter, compute_rl_performance_metrics


@pytest.mark.parametrize("widths,clip,non_aborted_clip", [((None, None), 1 / 3, 0.5), ((12, 8), 0.0, 0.0)])
def test_variable_lengths_and_explicit_padded_width(widths, clip, non_aborted_clip):
    timings = dict(step=7.25, gen=3.5, ref=0.25, old_log_prob=0.4, adv=0.01, update_actor=2, update_weights=0.5)
    actual = compute_rl_performance_metrics(
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
    actual = compute_rl_performance_metrics([2, 3], [0, 0], {"step": 7.25, "gen": 3.5, "update_actor": 2}, 8)
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
            compute_rl_performance_metrics([3], [2], {"step": seconds}, n_gpus)


def test_jsonl_flush_and_nonfinite(tmp_path):
    writer = JsonlMetricWriter(tmp_path / "metrics.jsonl")
    writer.write(1, {"good": 1.5, "nan": float("nan")})
    assert json.loads(writer.path.read_text())["data"] == {"good": 1.5, "nan": None}
    writer.close()
