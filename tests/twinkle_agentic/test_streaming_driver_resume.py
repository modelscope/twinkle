import pickle
import time

import pytest

from twinkle_agentic.async_rl.rollout_resume import ResumeStore
from twinkle_agentic.async_rl.streaming_driver import StreamingDriver
from twinkle_agentic.rollout.multi_turn import RolloutStatus


class _Future:
    def __init__(self, value=None):
        self._value = value

    def exception(self):
        return None

    def result(self):
        return self._value

    def add_done_callback(self, callback):
        callback(self)


class _BlockingFuture:
    def add_done_callback(self, callback):
        self.callback = callback


class _DoneRollout:
    def __init__(self):
        self.calls = []

    def step(self, trajectory, episode_id=None, session_wait=None):
        self.calls.append(episode_id)
        row = dict(trajectory, turns=1)
        sample = RolloutStatus(episode_id=episode_id, turns=1, done=True, token_lengths=(1, 1), stop_reason="stop")
        return _Future(([row], sample))

    def stop(self, trajectory, episode_id=None):
        return None


class _BoundRollout(_DoneRollout):
    def step(self, trajectory, episode_id=None, session_wait=None):
        self.calls.append((episode_id, trajectory.get("policy_version"), trajectory.get("adapter_path")))
        row = dict(trajectory, turns=1)
        sample = RolloutStatus(episode_id=episode_id, turns=1, done=True, token_lengths=(1, 1), stop_reason="stop")
        return _Future(([row], sample))


class _PolicyManager:
    def __init__(self, version, paths):
        self._version = version
        self._paths = dict(paths)

    def version(self):
        return self._version

    def adapter_path(self, version):
        return self._paths.get(version)


class _PartiallyBlockingRollout(_DoneRollout):
    def step(self, trajectory, episode_id=None, session_wait=None):
        self.calls.append(episode_id)
        if episode_id.endswith("member:1"):
            return _BlockingFuture()
        row = dict(trajectory, turns=1)
        sample = RolloutStatus(episode_id=episode_id, turns=1, done=True, token_lengths=(1, 1), stop_reason="stop")
        return _Future(([row], sample))


class _BlockingRollout(_DoneRollout):
    def step(self, trajectory, episode_id=None, session_wait=None):
        self.calls.append(episode_id)
        return _BlockingFuture()


def _prompts(count=2):
    return [{"messages": [{"role": "user", "content": f"prompt-{index}"}]} for index in range(count)]


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


def test_inflight_prompt_and_group_survive_snapshot(tmp_path):
    store = ResumeStore(tmp_path, "run-inflight")
    rollout = _BlockingRollout()
    driver = StreamingDriver(_prompts(), rollout, batch_size=2, num_generations=2, max_staleness=0, resume_store=store)
    driver._admit()
    state = driver.state_dict()

    restored_rollout = _BlockingRollout()
    restored = StreamingDriver(
        _prompts(), restored_rollout, batch_size=2, num_generations=2, max_staleness=0, resume_store=store
    )
    restored.load_state_dict(state)

    assert restored._prompt_seq == 1
    assert list(restored._groups) == ["0"]
    assert restored._groups["0"].row == _prompts()[0]
    assert len(restored._groups["0"].members) == 2
    assert restored._inflight == 2
    assert len(restored_rollout.calls) == 2
    assert [member.episode_id for member in restored._groups["0"].members] == [
        member.episode_id for member in driver._groups["0"].members
    ]
    assert [member.submit_traj["state_key"] for member in restored._groups["0"].members] == [
        member.submit_traj["state_key"] for member in driver._groups["0"].members
    ]


def test_ready_group_round_trip_and_lease_redelivery(tmp_path):
    store = ResumeStore(tmp_path, "run-ready")
    source = StreamingDriver(
        _prompts(), _DoneRollout(), batch_size=2, num_generations=2, max_staleness=0, max_steps=1, resume_store=store
    )
    source._start()
    _wait_until(lambda: source._ready_rows == 2)
    ready_state = source.state_dict()
    source.close()

    restored = StreamingDriver(
        _prompts(), _DoneRollout(), batch_size=2, num_generations=2, max_staleness=0, max_steps=1, resume_store=store
    )
    restored.load_state_dict(ready_state)
    rows, facts = restored._next_batch()
    assert len(rows) == 2
    assert facts["policy_versions"] == [0]
    assert restored._handed == 1

    leased_state = restored.state_dict()
    after_crash = StreamingDriver(
        _prompts(), _DoneRollout(), batch_size=2, num_generations=2, max_staleness=0, max_steps=1, resume_store=store
    )
    after_crash.load_state_dict(leased_state)
    again_rows, again_facts = after_crash._next_batch()
    assert again_rows == rows
    assert again_facts == facts
    with pytest.raises(Exception, match="durable model checkpoint"):
        after_crash.commit_batch("")
    after_crash.commit_batch("checkpoint-1", manifest={"optimizer_step": 7})
    commit = store.journal.records()[-1]
    assert commit.payload["checkpoint"] == "checkpoint-1"
    assert commit.payload["manifest"] == {"optimizer_step": 7}

    stale = StreamingDriver(
        _prompts(), _DoneRollout(), batch_size=2, num_generations=2, max_staleness=0, max_steps=1, resume_store=store
    )
    stale.load_state_dict(leased_state)
    assert stale._lease is None


def test_resumed_group_keeps_its_pinned_adapter_after_policy_advances(tmp_path):
    store = ResumeStore(tmp_path, "run-policy")
    manager = _PolicyManager(1, {1: "/policies/policy-1.adapter"})
    driver = StreamingDriver(
        _prompts(1),
        _BlockingRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=1,
        checkpoint_manager=manager,
        resume_store=store,
    )
    driver._version = manager.version()
    driver._admit()
    state = driver.state_dict()
    assert driver._groups["0"].version == 1
    assert driver._groups["0"].adapter_path == "/policies/policy-1.adapter"
    driver.close()

    manager._version = 2
    manager._paths[2] = "/policies/policy-2.adapter"
    rollout = _BoundRollout()
    restored = StreamingDriver(
        _prompts(1),
        rollout,
        batch_size=2,
        num_generations=2,
        max_staleness=1,
        checkpoint_manager=manager,
        resume_store=store,
    )
    restored.load_state_dict(pickle.loads(pickle.dumps(state)))

    assert restored._groups["0"].version == 1
    assert restored._groups["0"].adapter_path == "/policies/policy-1.adapter"
    assert {call[1:] for call in rollout.calls} == {(1, "/policies/policy-1.adapter")}
    assert restored.stats()["resumed_groups"] == 1


def test_ready_group_over_staleness_is_dropped_whole(tmp_path):
    store = ResumeStore(tmp_path, "run-stale")
    manager = _PolicyManager(1, {1: "/policies/policy-1.adapter"})
    source = StreamingDriver(
        _prompts(1),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        checkpoint_manager=manager,
        resume_store=store,
    )
    source._start()
    _wait_until(lambda: source._ready_rows == 2)
    state = source.state_dict()
    state["exhausted"] = True
    source.close()

    manager._version = 2
    manager._paths[2] = "/policies/policy-2.adapter"
    rollout = _BoundRollout()
    restored = StreamingDriver(
        _prompts(1),
        rollout,
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        checkpoint_manager=manager,
        resume_store=store,
    )
    restored.load_state_dict(state)
    assert restored._next_batch() is None
    assert restoration_stats(restored) == {
        "dropped_stale": 2,
        "dropped_group": 1,
        "ready_rows": 0,
        "live": 0,
    }
    assert rollout.calls == []


def test_resume_retains_a_partially_finished_group_for_training(tmp_path):
    store = ResumeStore(tmp_path, "run-partial")
    manager = _PolicyManager(1, {1: "/policies/policy-1.adapter"})
    rollout = _PartiallyBlockingRollout()
    driver = StreamingDriver(
        _prompts(1),
        rollout,
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        checkpoint_manager=manager,
        resume_store=store,
    )
    driver._version = manager.version()
    driver._admit()
    while not driver._events.empty():
        driver._handle(*driver._events.get_nowait())
    assert driver._groups["0"].members[0].finished
    assert not driver._groups["0"].members[1].finished
    state = driver.state_dict()
    driver.close()

    resumed_rollout = _BoundRollout()
    resumed = StreamingDriver(
        _prompts(1),
        resumed_rollout,
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        checkpoint_manager=manager,
        resume_store=store,
    )
    resumed.load_state_dict(state)
    rows, facts = resumed._next_batch()

    assert len(rows) == 2
    assert facts["policy_versions"] == [1]
    assert resumed.stats()["resumed_groups"] == 1
    assert resumed.stats()["resumed_members"] == 1
    assert resumed_rollout.calls == [("run-partial:g0:a0:member:1", 1, "/policies/policy-1.adapter")]

    lease_id = resumed.commit_batch("checkpoint/policy-1", manifest={"optimizer_step": 1})
    assert lease_id == "run-partial/batch/0"
    assert resumed._next_batch() is None

    after_commit = StreamingDriver(
        _prompts(1),
        _BoundRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        checkpoint_manager=manager,
        resume_store=store,
    )
    after_commit.load_state_dict(resumed.state_dict())
    assert after_commit._lease is None


def test_delivered_groups_release_the_admission_window(tmp_path):
    manager = _PolicyManager(1, {1: "/policies/policy-1.adapter"})
    driver = StreamingDriver(
        _prompts(3),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=2,
        checkpoint_manager=manager,
    )
    try:
        first, _ = driver._next_batch()
        driver.commit_batch("checkpoint/one")
        second, _ = driver._next_batch()
        driver.commit_batch("checkpoint/two")
        assert len(first) == len(second) == 2
        assert driver.stats()["live"] == 0
        assert driver._next_batch() is None
    finally:
        driver.close()


def restoration_stats(driver):
    stats = driver.stats()
    return {key: stats[key] for key in ("dropped_stale", "dropped_group", "ready_rows", "live")}
