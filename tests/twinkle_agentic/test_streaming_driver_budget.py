# Copyright (c) ModelScope Contributors. All rights reserved.
"""A reported optimizer step of 0 must not be rewritten into a budget step."""

import types

from twinkle_agentic.async_rl.streaming_driver import BufferRecord, StreamingDriver


def _driver(**overrides):
    kwargs = dict(
        prompt_stream=[],
        max_staleness=0,
        num_generations=1,
        groups_per_partition=1,
        buffer_depth=1,
        submit=lambda prompt, index, policy: None,
        poll=lambda handles: [],
        collect=lambda handle: None,
        assembly_ready=lambda ready: None,
        consume=lambda records: 0,
        max_steps=3,
    )
    kwargs.update(overrides)
    return StreamingDriver(**kwargs)


def test_commit_zero_does_not_exhaust_a_three_step_budget():
    driver = _driver()
    for reported in (0, 1, 1):
        driver.commit(reported)
    assert driver._committed_steps == 1
    assert driver._budget_steps == 2
    assert driver._default_reached_max() is False
    driver.commit(1)
    assert driver._budget_steps == 3
    assert driver._default_reached_max() is True


def test_omitted_commit_still_counts_as_one_step():
    driver = _driver()
    driver.commit()
    driver.commit(None)
    assert driver._budget_steps == 2
    assert driver._committed_steps == 1


def test_train_pull_keeps_explicit_zero_out_of_the_stats():
    driver = _driver()
    record = BufferRecord(sample=object(), version=0, group_key='g')
    ready = [record]
    driver._consume = lambda records: 0
    driver._train_pull([record], ready, {'g': 1})
    assert ready == []
    assert driver._last_pull_steps == 0
    assert driver.stats.optimizer_steps == 0
    assert driver.stats.train_pulls == 0
    assert driver.stats.consumed == 0


def test_train_pull_missing_report_defaults_to_one_step():
    driver = _driver()
    record = BufferRecord(sample=object(), version=0, group_key='g')
    ready = [record]
    driver._consume = lambda records: None
    driver._train_pull([record], ready, {'g': 1})
    assert ready == []
    assert driver._last_pull_steps == 1
    assert driver.stats.optimizer_steps == 1
    assert driver.stats.train_pulls == 1
    assert driver.stats.consumed == 1


def test_iterator_runs_until_three_real_updates():
    """The disaggregated loop reports [0, 1, 1, 1]; only the real updates fill max_steps=3."""
    active = {}
    next_handle = 0

    def submit(prompt, index, policy):
        nonlocal next_handle
        next_handle += 1
        active[next_handle] = policy.version
        return next_handle

    def collect(handle):
        active.pop(handle)
        return types.SimpleNamespace()

    driver = StreamingDriver(
        prompt_stream=list(range(32)),
        max_staleness=0,
        num_generations=1,
        groups_per_partition=1,
        buffer_depth=4,
        submit=submit,
        poll=lambda handles: list(handles),
        collect=collect,
        group_pull=True,
        need_rows=4,
        version_control=True,
        max_steps=3,
        consume=None,
    )
    reported = iter((0, 1, 1, 1))
    seen = []
    try:
        for batch in driver:
            assert len(batch) == 4
            steps = next(reported)
            seen.append(steps)
            driver.commit(steps)
    finally:
        driver.abort()
    assert seen == [0, 1, 1, 1]
    assert driver._budget_steps == 3
    assert driver.stats.optimizer_steps == 3
    assert driver.stats.consumed == 12
