"""Observe actual post-commit publication, including terminal and error paths."""

from types import SimpleNamespace

import pytest

from twinkle_agentic.async_rl.streaming_driver import StreamingDriver


def make_driver(**kwargs):
    return StreamingDriver(
        prompt_stream=list(range(12)),
        max_staleness=0,
        num_generations=1,
        groups_per_partition=1,
        buffer_depth=4,
        need_rows=4,
        group_pull=True,
        version_control=True,
        max_steps=2,
        submit=lambda ref, index, policy: (ref.ordinal, index),
        poll=lambda handles: list(handles),
        collect=lambda handle: SimpleNamespace(),
        consume=None,
        **kwargs,
    )


def test_observer_runs_after_real_publication_and_before_next_batch():
    events = []
    driver = make_driver(on_iteration_start=lambda: events.append("start"))
    driver.on_iteration_end = lambda terminal: events.append(("end", terminal, driver.stats.publishes))
    try:
        for samples in driver:
            events.append("train")
            driver.request_publish(lambda window: events.append("publish"))
            driver.commit(1)
    finally:
        driver.abort()
    assert events == ["start", "train", "publish", ("end", False, 1), "start", "train", ("end", True, 1)]


def test_failed_uncommitted_batch_is_not_a_completed_iteration():
    completions = []
    driver = make_driver(on_iteration_end=lambda terminal: completions.append(terminal))
    next(iter(driver))
    driver.abort()  # recipe threw before commit
    assert completions == []


def test_observer_failure_propagates_to_recipe():
    def failed(terminal):
        raise ValueError("cannot persist iteration")

    driver = make_driver(on_iteration_end=failed)
    try:
        with pytest.raises(ValueError, match="cannot persist"):
            for samples in driver:
                driver.commit(1)
    finally:
        driver.abort()


def test_performance_snapshot_is_plain_detached_data():
    driver = make_driver()
    driver.stats.publishes = 2
    snapshot = driver.performance_snapshot()
    assert snapshot["stream_stats"]["publishes"] == 2
    assert snapshot["resource_metrics"] == {}
    snapshot["stream_stats"]["publishes"] = 99
    assert driver.stats.publishes == 2
