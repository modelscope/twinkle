import errno
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from twinkle.data_format.sampling import SamplingParams
from twinkle_agentic.async_rl import rollout_resume
from twinkle_agentic.async_rl.rollout_resume import (
    DurableJournal,
    JournalEvent,
    ResumeStore,
    UnsafeResume,
    atomic_write_bytes,
)
from twinkle_agentic.async_rl.streaming_driver import StreamingDriver
from twinkle_agentic.envs.base import EnvResumeCapability
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout, RolloutStatus, _Episode


class _Future:
    def __init__(self, value=None):
        self._value = value

    def exception(self):
        return None

    def result(self):
        return self._value

    def add_done_callback(self, callback):
        callback(self)


class _DoneRollout:
    def step(self, trajectory, episode_id=None, session_wait=None):
        row = dict(trajectory, turns=1)
        sample = RolloutStatus(
            episode_id=episode_id,
            turns=1,
            done=True,
            token_lengths=(1, 1),
            stop_reason="stop",
        )
        return _Future(([row], sample))

    def stop(self, trajectory, episode_id=None):
        return None


class _PolicyManager:
    def __init__(self, version, paths):
        self._version = version
        self._paths = dict(paths)

    def version(self):
        return self._version

    def adapter_path(self, version):
        return self._paths.get(version)


class _Template:
    max_length = 32
    truncation_strategy = "right"


def _prompts(count=2):
    return [{"messages": [{"role": "user", "content": f"prompt-{index}"}]} for index in range(count)]


def _consume(store, run_id):
    driver = StreamingDriver(
        _prompts(3),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=2,
        run_id=run_id,
        resume_store=store,
    )
    batches = []
    try:
        for batch in range(2):
            rows, facts = driver._next_batch()
            batches.append((rows, facts))
            driver.commit_batch(f"checkpoint-{batch}" if store is not None else "")
    finally:
        driver.close()
    return batches


def test_no_fault_durable_and_memory_drivers_are_equivalent(tmp_path):
    memory_batches = _consume(None, "run-no-fault")
    durable_batches = _consume(ResumeStore(tmp_path, "run-no-fault"), "run-no-fault")
    assert durable_batches == memory_batches


def test_journal_concurrent_instances_share_one_chain(tmp_path):
    path = tmp_path / "rollout.jsonl"
    writers = [DurableJournal(path) for _ in range(4)]

    def append_many(writer, writer_index):
        for item in range(20):
            writer.append(JournalEvent.PROMPT_ASSIGNED, f"run/{writer_index}/{item}")

    with ThreadPoolExecutor(max_workers=len(writers)) as pool:
        list(pool.map(lambda pair: append_many(pair[1], pair[0]), enumerate(writers)))

    reloaded = DurableJournal(path)
    records = list(reloaded.records())
    assert len(records) == 80
    assert [record.seq for record in records] == list(range(80))
    assert len({record.digest for record in records}) == 80


def test_journal_fsync_failure_rolls_back_uncertain_append(tmp_path, monkeypatch):
    path = tmp_path / "rollout.jsonl"
    journal = DurableJournal(path)
    journal.append(JournalEvent.PROMPT_ASSIGNED, "run/stable")
    old_size = path.stat().st_size

    def disk_full(_descriptor):
        raise OSError(errno.ENOSPC, "disk full")

    with monkeypatch.context() as context:
        context.setattr(rollout_resume.os, "fsync", disk_full)
        with pytest.raises(OSError, match="disk full"):
            journal.append(JournalEvent.GROUP_READY, "run/uncertain")

    assert path.stat().st_size == old_size
    reloaded = DurableJournal(path)
    assert [record.state_key for record in reloaded.records()] == ["run/stable"]


def test_atomic_rename_failure_preserves_previous_object(tmp_path, monkeypatch):
    target = tmp_path / "objects" / "state.resume"
    atomic_write_bytes(target, b"old")

    def interrupted_replace(_source, _target):
        raise OSError(errno.EINTR, "replace interrupted")

    with monkeypatch.context() as context:
        context.setattr(rollout_resume.os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="replace interrupted"):
            atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"old"
    assert list(target.parent.glob("*.tmp")) == []


def _ready_driver(tmp_path, run_id):
    store = ResumeStore(tmp_path, run_id)
    driver = StreamingDriver(
        _prompts(1),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        resume_store=store,
    )
    driver._start()
    deadline = time.monotonic() + 2
    while driver._ready_rows < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert driver._ready_rows == 2
    return store, driver


def test_exactly_once_commit_contract_and_stale_snapshot(tmp_path):
    store, driver = _ready_driver(tmp_path, "run-lease")
    try:
        rows, facts = driver._next_batch()
        leased_state = driver.state_dict()
        with pytest.raises(UnsafeResume, match="durable model checkpoint"):
            driver.commit_batch("")
        assert driver.commit_batch("checkpoint/one", {"optimizer_step": 1}) == "run-lease/batch/0"
        assert driver.commit_batch("checkpoint/one", {"optimizer_step": 1}) is None
    finally:
        driver.close()

    recovered = StreamingDriver(
        _prompts(1),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        resume_store=store,
    )
    recovered.load_state_dict(leased_state)
    assert recovered._lease is None
    assert recovered._next_batch() is None
    assert rows
    assert facts["policy_versions"] == [0]


def _episode(rollout, state_key="run/group-0/episode-0"):
    trajectory = {"messages": [{"role": "user", "content": "hello"}], "state_key": state_key}
    episode = _Episode("episode-0", trajectory, rollout._params(SamplingParams()))
    episode.ledger = rollout._new_ledger("episode-0")
    episode.ledger._pif = {
        "input_ids": [1, 2, 3],
        "labels": [-100, 2, 3],
        "completion_mask": [0, 1, 1],
        "messages": trajectory["messages"],
    }
    episode.ledger._turns = 1
    episode.budget = 3
    episode.spent = 1
    episode.state_key = state_key
    return episode


class _NoResumeEnv:
    resume_capability = EnvResumeCapability.NONE

    def __init__(self):
        self.calls = 0

    def checkpoint_episode(self, episode_id, state_key):
        self.calls += 1
        return {"unexpected": True}

    def restore_episode(self, episode_id, state_key, state, trajectory=None):
        self.calls += 1


def test_env_without_resume_capability_fails_closed(tmp_path):
    store = ResumeStore(tmp_path, "run-env")
    env = _NoResumeEnv()
    rollout = MultiTurnRollout(response_callback=lambda *_args: None, template=_Template(), env=env, resume_store=store)
    episode = _episode(rollout)
    episode.env_open = True
    rollout._episodes[episode.episode_id] = episode

    with pytest.raises(UnsafeResume, match="no durable resume capability"):
        rollout.checkpoint_episode(episode.episode_id)
    assert env.calls == 0

    episode.env_open = False
    state = rollout.checkpoint_episode(episode.episode_id)
    state["env_open"] = True
    restored = MultiTurnRollout(
        response_callback=lambda *_args: None,
        template=_Template(),
        env=env,
        resume_store=store,
    )
    with pytest.raises(UnsafeResume, match="no durable resume capability"):
        restored.restore_episode(state)
    assert env.calls == 0
    assert restored._episodes == {}


class _CountingTool:
    def __init__(self):
        self.calls = 0

    def __call__(self, _tool_call):
        self.calls += 1
        return f"side-effect-{self.calls}"


def test_non_idempotent_tool_is_not_replayed_after_crash(tmp_path):
    store = ResumeStore(tmp_path, "run-tool")
    source = MultiTurnRollout(response_callback=lambda *_args: None, template=_Template(), resume_store=store)
    episode = _episode(source)
    source._episodes[episode.episode_id] = episode
    before_tool = source.checkpoint_episode(episode.episode_id)

    tool = _CountingTool()
    calls = [{"name": "charge_card", "arguments": {"amount": 10}}]
    effect_key, cached = source._begin_tool_effect(episode, calls)
    assert effect_key is not None and cached is None
    observations = source._run_tools(tool, calls)
    assert tool.calls == 1
    assert observations == ["side-effect-1"]

    crashed = MultiTurnRollout(response_callback=lambda *_args: None, template=_Template(), resume_store=store)
    with pytest.raises(UnsafeResume, match="pending tool effects"):
        crashed.restore_episode(before_tool)
    assert tool.calls == 1

    source._commit_tool_effect(episode, effect_key, observations)
    safe_state = store.load_episode(episode.state_key)
    recovered = MultiTurnRollout(response_callback=lambda *_args: None, template=_Template(), resume_store=store)
    recovered.restore_episode(safe_state)
    recovered_episode = recovered._episodes[episode.episode_id]
    replay_key, replay = recovered._begin_tool_effect(recovered_episode, calls)
    assert replay_key == effect_key
    assert replay == observations
    assert tool.calls == 1


def test_adapter_lifecycle_deleted_path_and_path_reuse_are_explicit(tmp_path):
    manager = _PolicyManager(1, {})
    driver = StreamingDriver(
        _prompts(1),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=1,
        checkpoint_manager=manager,
    )
    group_state = {
        "key": "0",
        "row": _prompts(1)[0],
        "version": 1,
        "adapter_path": None,
        "attempt": 0,
        "source": [],
        "members": [],
        "baselines": [],
        "missing": 0,
    }
    with pytest.raises(UnsafeResume, match="no resolvable adapter path"):
        driver._restore_group(group_state)

    manager._paths[1] = "/policies/reused-policy.adapter"
    explicit = dict(group_state, adapter_path="/policies/original-policy.adapter")
    restored = driver._restore_group(explicit)
    assert restored.adapter_path == "/policies/original-policy.adapter"


def test_openenv_adapters_declare_no_resume_capability():
    from twinkle_agentic.envs.agentenv import AgentEnv
    from twinkle_agentic.envs.openenv import OpenEnv, OpenEnvClient

    assert AgentEnv.resume_capability is EnvResumeCapability.NONE
    assert OpenEnv.resume_capability is EnvResumeCapability.NONE
    assert OpenEnvClient.resume_capability is EnvResumeCapability.NONE
