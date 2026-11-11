# Copyright (c) ModelScope Contributors. All rights reserved.
"""A reported optimizer step of 0 must not be rewritten into a budget step."""

import types

import pytest

from twinkle_agentic.async_rl.streaming_driver import BufferRecord, RolloutPolicy, StreamingDriver


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
    # ``consumed`` counts records HANDED TO ``consume`` (DriverStats:82), not optimizer steps: the
    # 0-step batch's one record was still pulled and consumed, so it counts. What zero-step budgeting
    # keeps out of the stats is ``optimizer_steps``/``train_pulls``/``_budget_steps`` (all 0 above).
    assert driver.stats.consumed == 1


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
    # 4 batches were pulled and handed to ``consume`` (seen == [0, 1, 1, 1]); ``consumed`` counts all
    # 16 records (4 batches x need_rows 4), including the 0-step batch -- it tracks records handed to
    # consume (DriverStats:82), while ``optimizer_steps`` (3) tracks only the real updates.
    assert driver.stats.consumed == 16


# --- per-group bookkeeping cleanup (E9): a terminal group must drop ALL of its driver-side entries ---
#
# The driver keeps three per-group maps keyed by group_key: ``group_counts`` (intra-group barrier count),
# ``group_prompt`` (the regenerate re-admission prompt) and ``resolved_groups`` (the once-per-group barrier
# guard). A group is TERMINAL once it is consumed (``_train_pull``) or fully stale-dropped (``_drop_stale``):
# it can never be re-collected, regenerated or barrier-checked again. Only ``group_counts`` used to be
# pruned, so ``group_prompt``/``resolved_groups`` grew one entry per group for the whole run (an unbounded
# within-run accumulation; E9 "history scale" arm). Each test below reverse-verifies: delete the matching
# prune line in streaming_driver.py and its assert goes red.


def test_train_pull_prunes_all_per_group_bookkeeping():
    driver = _driver()
    record = BufferRecord(sample=object(), version=0, group_key='g')
    ready = [record]
    group_counts = {'g': 1}
    group_prompt = {'g': object()}
    resolved_groups = {'g'}
    driver._consume = lambda records: 1
    driver._train_pull([record], ready, group_counts, group_prompt, resolved_groups)
    assert ready == []
    assert group_counts == {}
    assert group_prompt == {}          # leaked before the fix
    assert resolved_groups == set()    # leaked before the fix


def test_train_pull_prunes_only_consumed_groups_not_unrelated():
    driver = _driver()
    r1 = BufferRecord(sample=object(), version=0, group_key='g1')
    r2 = BufferRecord(sample=object(), version=0, group_key='g2')
    ready = [r1, r2]
    group_counts = {'g1': 1, 'g2': 1, 'g3': 1}
    group_prompt = {'g1': object(), 'g2': object(), 'g3': object()}
    resolved_groups = {'g1', 'g2', 'g3'}
    driver._consume = lambda records: 1
    driver._train_pull([r1, r2], ready, group_counts, group_prompt, resolved_groups)
    assert ready == []
    assert group_counts == {'g3': 1}
    assert set(group_prompt) == {'g3'}    # g3 was not pulled -> untouched
    assert resolved_groups == {'g3'}


def test_train_pull_same_group_siblings_prune_once():
    """num_generations>1: two records of one group prune its single entry once (idempotent, no KeyError)."""
    driver = _driver()
    r1 = BufferRecord(sample=object(), version=0, group_key='g')
    r2 = BufferRecord(sample=object(), version=0, group_key='g')
    ready = [r1, r2]
    group_counts = {'g': 2}
    group_prompt = {'g': object()}
    resolved_groups = {'g'}
    driver._consume = lambda records: 1
    driver._train_pull([r1, r2], ready, group_counts, group_prompt, resolved_groups)
    assert ready == []
    assert group_counts == {}
    assert group_prompt == {}
    assert resolved_groups == set()


def test_train_pull_none_bookkeeping_is_backward_compatible():
    """The pre-fix 3-arg call (group_prompt/resolved_groups omitted) still works: only group_counts prunes."""
    driver = _driver()
    record = BufferRecord(sample=object(), version=0, group_key='g')
    ready = [record]
    group_counts = {'g': 1}
    driver._consume = lambda records: 1
    driver._train_pull([record], ready, group_counts)
    assert ready == []
    assert group_counts == {}


def test_drop_stale_prunes_fully_dropped_group_bookkeeping():
    """A group whose last sibling is stale-dropped is terminal: group_prompt/resolved_groups drop too."""
    driver = _driver()
    driver._policy = RolloutPolicy(None, 5, None)   # current 5, record version 0 -> lag 5 > max_staleness 0
    record = BufferRecord(sample=object(), version=0, group_key='g')
    ready = [record]
    group_counts = {'g': 1}
    group_prompt = {'g': object()}
    resolved_groups = {'g'}
    driver._drop_stale(ready, group_counts, group_prompt, resolved_groups)
    assert ready == []
    assert group_counts == {}
    assert group_prompt == {}          # leaked before the fix
    assert resolved_groups == set()    # leaked before the fix
    assert driver.stats.dropped_stale == 1


def test_drop_stale_partial_drop_keeps_group_bookkeeping():
    """Branch guard: when a drop leaves siblings buffered (remaining>0), the group is NOT terminal yet, so
    group_prompt/resolved_groups must survive. group_counts is deliberately ahead of the buffer to drive the
    remaining>0 path; reverse-verify by moving the prune out of the else branch -> this assert goes red."""
    driver = _driver()
    driver._policy = RolloutPolicy(None, 5, None)
    record = BufferRecord(sample=object(), version=0, group_key='g')   # one stale sibling
    ready = [record]
    group_counts = {'g': 2}        # count 2, buffer 1 -> after drop remaining=1>0 (not terminal)
    group_prompt = {'g': object()}
    resolved_groups = {'g'}
    driver._drop_stale(ready, group_counts, group_prompt, resolved_groups)
    assert ready == []
    assert group_counts == {'g': 1}        # decremented, not popped
    assert set(group_prompt) == {'g'}      # NOT pruned (a sibling is still buffered)
    assert resolved_groups == {'g'}        # NOT pruned


@pytest.mark.parametrize('group_pull,cap,expected', [(True, 12, 12), (False, 4, 4)])
def test_capped_pull_retains_prefetched_records(group_pull, cap, expected):
    records = [BufferRecord(sample=object(), version=0, group_key=str(i // 3)) for i in range(24)]
    driver = _driver(
        num_generations=3, need_rows=4, group_pull=group_pull,
        assembly_ready=None, max_rows_per_pull=cap, buffer_depth=24,
    )
    selected = driver._assembly_ready(records)
    assert len(selected) == expected
    assert len(records) == 24
    uncapped = _driver(num_generations=3, need_rows=4, group_pull=group_pull, assembly_ready=None)
    assert len(uncapped._assembly_ready(records)) == 24


@pytest.mark.parametrize('cap', [0, 8, 13])
def test_group_pull_cap_requires_complete_training_and_generation_quanta(cap):
    with pytest.raises(ValueError, match='pull quantum 12'):
        _driver(num_generations=3, need_rows=4, group_pull=True, assembly_ready=None, max_rows_per_pull=cap)


def test_custom_pull_rule_cannot_silently_ignore_a_cap():
    with pytest.raises(ValueError, match='built-in group_pull'):
        _driver(max_rows_per_pull=4)


@pytest.mark.parametrize('cap', [True, 12.0, '12'])
def test_pull_cap_rejects_non_integer_counts_before_iteration(cap):
    with pytest.raises(TypeError, match='integer trajectory count'):
        _driver(num_generations=3, need_rows=4, group_pull=True, assembly_ready=None, max_rows_per_pull=cap)


def test_capped_group_pull_preserves_siblings_and_pending_score_barrier():
    # Completion order is interleaved; one otherwise complete group is still being scored.
    records = [BufferRecord(sample=object(), version=0, group_key=key)
               for key in ('a', 'b', 'c', 'a', 'b', 'c')]
    records[0].score_ref = object()
    driver = _driver(num_generations=2, need_rows=2, group_pull=True, assembly_ready=None,
                     max_rows_per_pull=2, buffer_depth=6)
    selected = driver._assembly_ready(records)
    assert selected == [records[1], records[4]]
    assert len(records) == 6
    remaining = [record for record in records if record not in selected]
    assert driver._assembly_ready(remaining) == [records[2], records[5]]
    records[0].score_ref = None
    assert driver._assembly_ready(remaining) == [records[0], records[3]]
