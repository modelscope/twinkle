"""The remaining-batch limit leaves room to finish or replace admitted groups."""

from concurrent.futures import Future

import pytest

from twinkle_agentic.async_rl.streaming_driver import StreamingDriver
from twinkle_agentic.rollout.multi_turn import EpisodeError, RolloutStatus


class PendingRollout:

    def __init__(self):
        self.pending = {}

    def step(self, traj, *, episode_id, session_wait):
        future = Future()
        self.pending[episode_id] = (traj, future)
        return future

    def finish(self, driver, episode, *, failed=False):
        traj, future = self.pending.pop(episode.episode_id)
        if failed:
            future.set_exception(EpisodeError('generation failed'))
        else:
            future.set_result(([traj], RolloutStatus(done=True)))
        driver._handle(*driver._events.get_nowait())


@pytest.mark.parametrize('max_steps, expected', [(None, 8), (2, 8), (1, 4)])
def test_tail_prefetch_caps_admission(max_steps, expected):
    driver = StreamingDriver([{}] * 4, PendingRollout(), batch_size=4, num_generations=2,
                             max_steps=max_steps, max_staleness=1)
    with driver._lock:
        driver._admit()
        assert not driver._exhausted
    assert driver.stats()['admitted'] == expected


def test_tail_prefetch_reserves_an_incomplete_group_and_replaces_it_if_dropped():
    rollout = PendingRollout()
    driver = StreamingDriver([{}] * 4, rollout, batch_size=4, num_generations=2, max_steps=1,
                             max_staleness=1, max_group_regenerations=1)
    with driver._lock:
        driver._admit()
    first, second = driver._groups.values()
    for episode in first.members:
        rollout.finish(driver, episode)
    rollout.finish(driver, second.members[0], failed=True)
    with driver._lock:
        driver._admit()
    assert driver.stats()['admitted'] == 4

    rollout.finish(driver, second.members[1])
    assert driver.stats()['admitted'] == 5
    rollout.finish(driver, second.members[0], failed=True)
    with driver._lock:
        driver._admit()
    replacement, = driver._groups.values()
    for episode in replacement.members:
        rollout.finish(driver, episode)
    try:
        rows, _ = next(iter(driver))
        assert [row['group_key'] for row in rows] == ['0', '0', '2', '2']
        assert driver.stats()['dropped_group'] == 1
    finally:
        driver.close()
