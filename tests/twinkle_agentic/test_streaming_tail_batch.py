"""Streaming delivery hands out whole batch_size rows of complete groups; an incomplete tail is dropped, never padded."""

from collections import Counter
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


def test_close_timeout_keeps_the_pending_step_and_stop_owned():
    submitted, stopping = Event(), Event()
    step, stop = Future(), Future()
    stopped = []

    class Rollout:
        def step(self, traj, *, episode_id, session_wait):
            submitted.set()
            return step

        def stop(self, traj, *, episode_id):
            stopped.append((episode_id, traj))
            stopping.set()
            return stop

    driver = StreamingDriver([[{}]], Rollout(), batch_size=1)
    driver._start()
    assert submitted.wait(5)
    try:
        with pytest.raises(TimeoutError, match='step for episode'):
            driver.close(timeout=0.3)
        assert stopped == []
        step.set_result(([{'latest': True}], RolloutStatus(episode_id='episode')))
        assert stopping.wait(5)
        with pytest.raises(TimeoutError, match='stop for episode'):
            driver.close(timeout=0.1)
        assert len(stopped) == 1 and stopped[0][1] == {'latest': True}
    finally:
        if not step.done():
            step.set_result(([{}], RolloutStatus(episode_id='episode')))
        stop.set_result(None)
        driver.close(timeout=5)
    assert len(stopped) == 1 and driver.stats()['inflight'] == 0


def test_close_stops_an_episode_as_soon_as_its_own_step_finishes():
    steps = [Future(), Future()]
    both_submitted, fast_stopped = Event(), Event()
    stopped = []

    class Rollout:
        def step(self, traj, *, episode_id, session_wait):
            index = traj['prompt']
            if index == 1:
                both_submitted.set()
            return steps[index]

        def stop(self, traj, *, episode_id):
            index = traj['prompt']
            assert steps[index].done()
            stopped.append(index)
            if index == 1:
                fast_stopped.set()

    driver = StreamingDriver([[{'prompt': 0}, {'prompt': 1}]], Rollout(), batch_size=2)
    driver._start()
    with ThreadPoolExecutor(1) as pool:
        closing = None
        try:
            assert both_submitted.wait(5)
            closing = pool.submit(driver.close, timeout=5)
            with driver._lock:
                assert driver._lock.wait_for(lambda: driver._closed, timeout=5)
            steps[1].set_result(([{'prompt': 1}], RolloutStatus(episode_id='fast')))
            assert fast_stopped.wait(2)
            assert not steps[0].done() and not closing.done()
            assert stopped == [1]
        finally:
            for index, step in enumerate(steps):
                if not step.done():
                    step.set_result(([{'prompt': index}], RolloutStatus(episode_id=str(index))))
        if closing is not None:
            closing.result(timeout=5)
    driver.close()
    assert stopped == [1, 0]


@pytest.mark.parametrize('synchronous', [False, True])
def test_slow_stop_does_not_block_other_episodes(synchronous, monkeypatch):
    all_submitted, slow_started, fast_stopped = Event(), Event(), Event()
    steps = [Future() for _ in range(4)]
    slow_stop = Future()
    stopped = []
    stop_workers = set()

    class Rollout:
        def step(self, traj, *, episode_id, session_wait):
            prompt = traj['prompt']
            if prompt == len(steps) - 1:
                all_submitted.set()
            return steps[prompt]

        def stop(self, traj, *, episode_id):
            prompt = traj['prompt']
            assert steps[prompt].done()
            stop_workers.add(current_thread())
            stopped.append(prompt)
            if prompt == 0:
                slow_started.set()
                if synchronous:
                    slow_stop.result()
                    return None
                return slow_stop
            if len(set(stopped) - {0}) == len(steps) - 1:
                fast_stopped.set()
            return _completed(None)

    driver = StreamingDriver([[{'prompt': p} for p in range(len(steps))]], Rollout(), batch_size=len(steps))
    monkeypatch.setattr(driver, 'CLOSE_STOP_WORKERS', 2)
    driver._start()
    assert all_submitted.wait(5)
    with ThreadPoolExecutor(1) as pool:
        closing = pool.submit(driver.close, timeout=0.75)
        try:
            with driver._lock:
                assert driver._lock.wait_for(lambda: driver._closed, timeout=5)
            for prompt, step in enumerate(steps):
                step.set_result(([{'prompt': prompt}], RolloutStatus(episode_id=str(prompt))))
            assert slow_started.wait(5)
            assert fast_stopped.wait(2), 'Other episodes could not stop while the first stop was pending'
            with pytest.raises(TimeoutError, match='stop for episode'):
                closing.result(timeout=5)
            assert not slow_stop.done()
        finally:
            for prompt, step in enumerate(steps):
                if not step.done():
                    step.set_result(([{'prompt': prompt}], RolloutStatus(episode_id=str(prompt))))
            slow_stop.set_result(None)
            driver.close(timeout=5)
    assert Counter(stopped) == Counter(range(len(steps)))
    assert len(stop_workers) == 2
    assert driver.stats()['inflight'] == 0


def test_close_drains_slow_scores_from_groups_dropped_for_no_slot(monkeypatch):
    scores = []

    class Rollout(_Rollout):
        def step(self, traj, *, episode_id, session_wait):
            if traj['sample_index'] == 1:
                return _completed(([traj], RolloutStatus(episode_id=episode_id, no_slot=True)))
            return super().step(traj, episode_id=episode_id, session_wait=session_wait)

    def slow_orm(_trajs):
        future = Future()
        scores.append(future)
        return future

    prompts = 5
    rollout = Rollout()
    driver = StreamingDriver([[{} for _ in range(prompts)]], rollout, batch_size=2, num_generations=2,
                             max_steps=1, max_no_slot_wait=0, orms=[slow_orm])
    monkeypatch.setattr(driver, 'NO_SLOT_BASE_DELAY', 0)
    monkeypatch.setattr(driver, 'NO_SLOT_JITTER', 0)
    with ThreadPoolExecutor(1) as pool:
        consuming = pool.submit(list, driver)
        try:
            assert consuming.result(timeout=5) == []
            stats = driver.stats()
            assert stats['inflight'] == stats['handed'] == 0
            assert stats['no_slot_dropped'] == len(scores) == prompts
            assert len(driver._pending) > driver._capacity()
            with pytest.raises(TimeoutError, match='orm for episode'):
                driver.close(timeout=0.25)
            assert set(driver._pending) == set(scores)
        finally:
            for future in scores:
                future.set_result([1.0])
            driver.close(timeout=5)
    assert not driver._pending
    assert rollout.stopped == []


def test_close_times_out_while_the_driver_is_blocked_in_admission():
    reading, release = Event(), Event()
    submitted = []

    class Dataloader:
        def __iter__(self):
            reading.set()
            assert release.wait(5)
            yield [{}]

    class Rollout(_Rollout):
        def step(self, *args, **kwargs):
            submitted.append(True)
            return super().step(*args, **kwargs)

    driver = StreamingDriver(Dataloader(), Rollout(), batch_size=1)
    driver._start()
    try:
        assert reading.wait(5)
        with pytest.raises(TimeoutError, match='driver thread'):
            driver.close(timeout=0.1)
    finally:
        release.set()
        driver.close(timeout=5)
    assert submitted == []


def test_close_retries_only_episodes_without_a_successful_stop():
    submitted = Event()
    unfinished = [Future(), Future()]
    calls = []

    class Rollout:
        def step(self, traj, *, episode_id, session_wait):
            if traj['prompt'] == 1:
                submitted.set()
            return unfinished[traj['prompt']]

        def stop(self, traj, *, episode_id):
            prompt = traj['prompt']
            calls.append(prompt)
            if prompt == 1 and calls.count(1) == 1:
                raise RuntimeError('missing stop acknowledgement')

    driver = StreamingDriver([[{'prompt': 0}, {'prompt': 1}]], Rollout(), batch_size=2)
    driver._start()
    assert submitted.wait(5)

    def finish_steps():
        with driver._lock:
            assert driver._lock.wait_for(lambda: driver._closed, timeout=5)
        for prompt, future in enumerate(unfinished):
            future.set_result(([{'prompt': prompt}], RolloutStatus(episode_id=str(prompt))))

    with ThreadPoolExecutor(1) as pool:
        stepping = pool.submit(finish_steps)
        with pytest.raises(ExceptionGroup, match='could not close rollout episodes'):
            driver.close(timeout=5)
        stepping.result(timeout=5)
    driver.close(timeout=5)
    assert Counter(calls) == {0: 1, 1: 2}


def test_sync_during_reclaim_replaces_stale_batch_before_spending_last_step():
    reclaiming, resume = Event(), Event()

    class Checkpoints:
        current = 0

        def version(self):
            return self.current

    class Adapter:
        def lend(self, *args, **kwargs):
            pass

        def reclaim(self):
            if not reclaiming.is_set():
                reclaiming.set()
                assert resume.wait(5)

        def adjust(self):
            pass

    checkpoints = Checkpoints()
    driver = make_driver(1, batch_size=1, max_steps=1, sync_interval=2, checkpoint_manager=checkpoints,
                         cluster_adapter=Adapter(), max_group_regenerations=1)
    with ThreadPoolExecutor(1) as pool:
        consuming = pool.submit(list, driver)
        try:
            assert reclaiming.wait(5)
            with driver._lock:
                checkpoints.current = 1
                assert driver._lock.wait_for(lambda: driver._version == 1, timeout=5)
                assert driver._handed == 0
            resume.set()
            [(rows, facts)] = consuming.result(timeout=5)
            assert len(rows) == 1 and rows[0]['group_attempt'] == 1
            assert facts['policy_versions'] == [1]
            assert driver.stats()['handed'] == driver.state_dict()['prompts_drawn'] == 1
        finally:
            resume.set()
            driver.close(timeout=5)


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
