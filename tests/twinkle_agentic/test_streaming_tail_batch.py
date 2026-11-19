"""Streaming delivery hands out whole batch_size rows of complete groups; an incomplete tail is dropped, never padded."""

from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event, current_thread

import pytest

from twinkle_agentic.async_rl.streaming_driver import StreamingDriver
from twinkle_agentic.rollout.multi_turn import RolloutStatus


def _completed(value):
    future = Future()
    future.set_result(value)
    return future


class _Rollout:
    """Single-turn rollout: every step finishes its episode at once, returning (rows, RolloutStatus)."""

    def __init__(self):
        self.stopped = []

    def step(self, traj, episode_id=None, session_wait=None):
        sample = RolloutStatus(episode_id=episode_id, done=True, turns=1, token_lengths=(4, 6))
        return _completed(([dict(traj)], sample))

    def stop(self, traj, episode_id=None):
        self.stopped.append(episode_id)


def _orm(trajs):
    return _completed([1.0])


def make_driver(prompts, *, batch_size=4, generations=1, **kwargs):
    dataloader = [[{'prompt': i} for i in range(prompts)]]
    return StreamingDriver(dataloader, _Rollout(), batch_size=batch_size, num_generations=generations,
                           orms=[_orm], max_staleness=0, **kwargs)


def sizes_of(driver):
    return [len(rows) for rows, _facts in driver]


@pytest.mark.parametrize('prompts,generations,batch_size,sizes,leftover', [
    (10, 1, 4, [4, 4], 2),
    (5, 1, 4, [4], 1),
    (8, 1, 4, [4, 4], 0),
    (2, 3, 12, [], 6),
    (12, 3, 12, [12, 12, 12], 0),
])
def test_only_whole_batches_delivered_incomplete_tail_dropped(prompts, generations, batch_size, sizes, leftover):
    driver = make_driver(prompts, batch_size=batch_size, generations=generations)
    try:
        assert sizes_of(driver) == sizes
        stats = driver.stats()
        assert stats['handed'] == len(sizes)
        stats['handed'] = -1
        assert driver.stats()['handed'] == len(sizes)
        assert stats['ready_rows'] == leftover
    finally:
        driver.close()


def test_groups_delivered_contiguously_not_split_across_batches():
    driver = make_driver(12, batch_size=12, generations=3)
    try:
        batches = list(driver)
        assert [len(rows) for rows, _ in batches] == [12, 12, 12]
        for rows, facts in batches:
            keys = [row['group_key'] for row in rows]
            for start in range(0, len(rows), 3):
                assert len(set(keys[start:start + 3])) == 1
            assert facts['num_rows'] == len(rows)
            assert len(facts['scores']) == len(rows)
    finally:
        driver.close()


@pytest.mark.parametrize('prompts,max_steps,sizes', [
    (10, 1, [4]),
    (8, 2, [4, 4]),
    (5, 1, [4]),
])
def test_explicit_budget_stop_does_not_flush_a_tail(prompts, max_steps, sizes):
    driver = make_driver(prompts, batch_size=4, max_steps=max_steps)
    try:
        assert sizes_of(driver) == sizes
        assert driver.stats()['handed'] == len(sizes)
    finally:
        driver.close()


def test_close_tears_down_the_background_thread():
    driver = make_driver(8)
    list(driver)
    assert driver._thread is not None and driver._thread.is_alive()
    driver.close()
    assert not driver._thread.is_alive()


def test_second_iteration_is_rejected():
    driver = make_driver(8)
    try:
        iter(driver)
        with pytest.raises(RuntimeError, match='ONE iteration'):
            iter(driver)
    finally:
        driver.close()


def test_batch_size_must_be_a_multiple_of_num_generations():
    with pytest.raises(ValueError, match='multiple of num_generations'):
        make_driver(8, batch_size=5, generations=3)


def test_skipping_updates_does_not_stall_a_finite_stream():
    # No optimizer update or policy sync happens as the caller consumes the batches.
    class Checkpoints:
        def version(self):
            return 1

    stream = StreamingDriver([[{}] * 3], _Rollout(), batch_size=2, num_generations=2,
                             checkpoint_manager=Checkpoints())
    with ThreadPoolExecutor(1) as executor:
        try:
            batches = executor.submit(list, stream).result(timeout=5)
            assert [facts['policy_versions'] for _, facts in batches] == [[1], [1], [1]]
        finally:
            stream.close()


def test_close_wakes_the_iterator_and_waits_for_step_and_stop():
    submitted, stopping = Event(), Event()
    step, stop = Future(), Future()

    class DelayedRollout:

        def step(self, trajectory, *, episode_id, session_wait):
            submitted.set()
            return step

        def stop(self, trajectory, *, episode_id):
            assert step.done(), 'an in-flight reset must finish before its session is closed'
            stopping.set()
            return stop

    driver = StreamingDriver([[{}]], DelayedRollout(), batch_size=1)
    with ThreadPoolExecutor(2) as pool:
        waiting = pool.submit(list, driver)
        assert submitted.wait(timeout=5)
        closing = pool.submit(driver.close)
        try:
            assert waiting.result(timeout=5) == []
            assert not closing.done()
            assert not stopping.is_set()
            step.set_result(([{}], RolloutStatus(episode_id='episode')))
            assert stopping.wait(timeout=5)
            assert not closing.done(), 'the stop request still owns its resources'
        finally:
            if not step.done():
                step.set_result(([{}], RolloutStatus(episode_id='episode')))
            stop.set_result(None)
        closing.result(timeout=5)
    driver.close()
    assert driver.stats()['inflight'] == 0


def test_stale_group_regeneration_runs_on_the_driver_thread():
    class Rollout(_Rollout):
        def step(self, trajectory, **kwargs):
            assert current_thread().name == 'twinkle-streaming-driver'
            return super().step(trajectory, **kwargs)

    class Checkpoints:
        current = 0

        def version(self):
            return self.current

    checkpoints = Checkpoints()
    driver = StreamingDriver([[{}, {}]], Rollout(), batch_size=1, sync_interval=2,
                             checkpoint_manager=checkpoints, max_group_regenerations=1)
    try:
        batches = iter(driver)
        first, _ = next(batches)
        # Wait for both old-policy rows to reach delivery state before publishing.
        with driver._lock:
            assert driver._lock.wait_for(lambda: driver._ready_rows == 1, timeout=5)
        checkpoints.current += 1
        second, _ = next(batches)
        assert first[0]['policy_version'] == 0
        assert second[0]['policy_version'] == second[0]['group_attempt'] == 1
    finally:
        driver.close()
