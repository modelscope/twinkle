"""Streaming delivery drops tails that cannot fill the normal optimizer batch rule."""

from types import SimpleNamespace

import pytest

from twinkle_agentic.async_rl.streaming_driver import StreamingDriver


def driver(*, prompts=10, generations=1, grouped=False, buffer_depth=32, consume=None, **kwargs):
    return StreamingDriver(
        prompt_stream=[list(range(prompts))],
        max_staleness=0,
        num_generations=generations,
        groups_per_partition=1,
        buffer_depth=buffer_depth,
        submit=lambda ref, index, policy: (ref.ordinal, index),
        poll=lambda handles: list(handles),
        collect=lambda handle: SimpleNamespace(policy_version=0),
        group_pull=grouped,
        need_rows=4,
        consume=consume,
        **kwargs,
    )


@pytest.mark.parametrize('prompts,generations,grouped,consumed,dropped,updates', [
    (10, 1, False, 8, 2, 2),
    (5, 1, False, 4, 1, 1),
    (6, 3, True, 12, 6, 3),
    (2, 3, True, 0, 6, 0),
    (4, 3, True, 12, 0, 3),
    (5, 3, True, 12, 3, 3),
])
def test_natural_drain_drops_incomplete_optimizer_batches(prompts, generations, grouped, consumed, dropped, updates):
    batches = []

    def consume(batch):
        batches.append(list(batch))
        return len(batch) // 4

    stream = driver(prompts=prompts, generations=generations, grouped=grouped, consume=consume)
    stream.run()
    assert stream.stats.consumed == consumed
    assert stream.stats.dropped_incomplete == dropped
    assert stream.stats.optimizer_steps == updates
    assert all(len(batch) % 4 == 0 for batch in batches)
    if grouped:
        for batch in batches:
            assert len(batch) % generations == 0
            for start in range(0, len(batch), generations):
                assert len({record.group_key for record in batch[start:start + generations]}) == 1


def test_iterator_drops_short_tail_and_tears_down_its_thread():
    stream = driver(prompts=5)
    sizes = []
    try:
        for batch in stream:
            sizes.append(len(batch))
            stream.commit(len(batch) // 4)
    finally:
        stream.abort()
    assert sizes == [4]
    assert stream.stats.optimizer_steps == 1
    assert stream.stats.consumed == 4 and stream.stats.dropped_incomplete == 1
    assert not stream._iter_thread.is_alive()


def test_zero_step_commit_does_not_consume_the_optimizer_budget():
    stream = driver(prompts=9, buffer_depth=4, max_steps=1)
    sizes = []
    try:
        for batch in stream:
            sizes.append(len(batch))
            stream.commit(0 if len(sizes) == 1 else 1)
    finally:
        stream.abort()
    assert sizes == [4, 4]
    assert stream.stats.optimizer_steps == stream._budget_steps == 1
    assert stream.stats.train_pulls == 1
    assert stream.stats.consumed == 8


def test_blocking_consumer_can_buffer_a_pull_without_reporting_an_update():
    sizes = []

    def consume(batch):
        sizes.append(len(batch))
        return 0 if len(sizes) == 1 else 1

    stream = driver(prompts=9, buffer_depth=4, consume=consume)
    stream.run()
    assert sizes == [4, 4]
    assert stream.stats.optimizer_steps == 1
    assert stream.stats.consumed == 8 and stream.stats.dropped_incomplete == 1


def test_explicit_budget_stop_does_not_flush_a_tail():
    stream = driver(prompts=5, max_steps=1)
    sizes = []
    try:
        for batch in stream:
            sizes.append(len(batch))
            stream.commit(1)
    finally:
        stream.abort()
    assert sizes == [4]
    assert stream.stats.optimizer_steps == 1
