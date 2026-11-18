"""Streaming delivers only full batches of prompt groups."""

from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from twinkle_agentic.async_rl.streaming_driver import StreamingDriver
from twinkle_agentic.rollout.multi_turn import RolloutStatus


class ImmediateRollout:

    def step(self, trajectory, *, episode_id, session_wait):
        result = Future()
        result.set_result(([trajectory], RolloutStatus(episode_id=episode_id, done=True)))
        return result

    def stop(self, _trajectory, *, episode_id):
        pass


@pytest.mark.parametrize('prompts,generations,batch_size,expected', [
    (10, 1, 4, 2),
    (3, 2, 4, 1),
    (1, 2, 4, 0),
    (4, 3, 6, 2),
])
def test_natural_drain_drops_short_tail(prompts, generations, batch_size, expected):
    stream = StreamingDriver(
        [[{'prompt': index} for index in range(prompts)]], ImmediateRollout(),
        batch_size=batch_size, num_generations=generations)
    try:
        batches = list(stream)
        assert len(batches) == stream.stats()['handed'] == expected
        for rows, facts in batches:
            assert len(rows) == facts['num_rows'] == batch_size
            assert len({row['episode_id'] for row in rows}) == batch_size
            for start in range(0, len(rows), generations):
                assert len({row['group_key'] for row in rows[start:start + generations]}) == 1
    finally:
        stream.close()
    assert not stream._thread.is_alive()


def test_batch_budget_counts_deliveries_and_stats_are_detached():
    stream = StreamingDriver([[{}] * 5], ImmediateRollout(), batch_size=2, max_steps=1)
    try:
        assert [len(rows) for rows, _ in stream] == [2]
        snapshot = stream.stats()
        snapshot['handed'] = 99
        assert stream.stats()['handed'] == 1
        with pytest.raises(RuntimeError, match='ONE iteration'):
            iter(stream)
    finally:
        stream.close()



def test_skipping_updates_does_not_stall_a_finite_stream():
    # No optimizer update or policy sync happens as the caller consumes the batches.
    class Checkpoints:
        def version(self):
            return 1

    stream = StreamingDriver([[{}] * 3], ImmediateRollout(), batch_size=2, num_generations=2,
                             checkpoint_manager=Checkpoints())
    with ThreadPoolExecutor(1) as executor:
        try:
            batches = executor.submit(list, stream).result(timeout=5)
            assert [facts['policy_versions'] for _, facts in batches] == [[1], [1], [1]]
        finally:
            stream.close()


def test_close_wakes_the_iterator_and_waits_for_step_and_stop():
    from threading import Event

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
    from threading import current_thread

    class Rollout(ImmediateRollout):
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
