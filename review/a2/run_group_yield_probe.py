#!/usr/bin/env python3
"""量化 durable resume 保留的可训练 GRPO 完整组。

场景模拟训练 checkpoint 已把 DataLoader 游标推进到下一个 prompt，但上一条 prompt
仍有成员在采样。普通重启会跳过该 prompt；durable resume 必须恢复半完成组、补齐成员，
并且在训练 checkpoint 提交后拒绝再次交付。
"""
# ruff: noqa: E402, I001, T201

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "review" / "gate0"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import py310_compat  # noqa: E402,F401
import run_gate0  # noqa: E402,F401

from twinkle_agentic.async_rl.rollout_resume import ResumeStore  # noqa: E402
from twinkle_agentic.async_rl.streaming_driver import StreamingDriver  # noqa: E402
from twinkle_agentic.rollout.multi_turn import RolloutStatus  # noqa: E402


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


class _Manager:
    def __init__(self, version, paths):
        self._version = version
        self._paths = dict(paths)

    def version(self):
        return self._version

    def adapter_path(self, version):
        return self._paths.get(version)


class _PartialRollout:
    def __init__(self):
        self.calls = []

    def step(self, trajectory, episode_id=None, session_wait=None):
        del session_wait
        self.calls.append(episode_id)
        if episode_id.endswith("member:1"):
            return _BlockingFuture()
        row = dict(trajectory, turns=1)
        sample = RolloutStatus(episode_id=episode_id, turns=1, done=True, token_lengths=(1, 1), stop_reason="stop")
        return _Future(([row], sample))

    def stop(self, trajectory, episode_id=None):
        del trajectory, episode_id


class _DoneRollout(_PartialRollout):
    def step(self, trajectory, episode_id=None, session_wait=None):
        del session_wait
        self.calls.append(episode_id)
        row = dict(trajectory, turns=1)
        sample = RolloutStatus(episode_id=episode_id, turns=1, done=True, token_lengths=(1, 1), stop_reason="stop")
        return _Future(([row], sample))


def _prompt():
    return {"messages": [{"role": "user", "content": "durable group probe"}]}


def run() -> dict:
    with tempfile.TemporaryDirectory(prefix="group-yield-") as root:
        store = ResumeStore(Path(root), "group-yield")
        manager = _Manager(1, {1: "/policies/policy-1.adapter"})
        source = StreamingDriver(
            [_prompt()],
            _PartialRollout(),
            batch_size=2,
            num_generations=2,
            max_staleness=0,
            max_steps=1,
            checkpoint_manager=manager,
            resume_store=store,
        )
        source._version = manager.version()
        source._admit()
        while not source._events.empty():
            source._handle(*source._events.get_nowait())
        state = source.state_dict()
        source.close()

        # The normal trainer checkpoint restored a cursor past the in-flight prompt.
        cursor_restart = StreamingDriver(
            [],
            _DoneRollout(),
            batch_size=2,
            num_generations=2,
            max_staleness=0,
            max_steps=1,
            checkpoint_manager=manager,
            resume_store=store,
        )
        cursor_batch = cursor_restart._next_batch()
        cursor_restart.close()

        durable = StreamingDriver(
            [_prompt()],
            _DoneRollout(),
            batch_size=2,
            num_generations=2,
            max_staleness=0,
            max_steps=1,
            checkpoint_manager=manager,
            resume_store=store,
        )
        durable.load_state_dict(state)
        rows, facts = durable._next_batch()
        leased_state = durable.state_dict()
        lease_id = durable.commit_batch("/checkpoints/policy-1", manifest={"optimizer_step": 1})
        duplicate = durable.commit_batch("/checkpoints/policy-1", manifest={"optimizer_step": 1})

        after_commit = StreamingDriver(
            [],
            _DoneRollout(),
            batch_size=2,
            num_generations=2,
            max_staleness=0,
            max_steps=1,
            checkpoint_manager=manager,
            resume_store=store,
        )
        after_commit.load_state_dict(leased_state)
        after_commit.close()
        durable.close()

        return {
            "ok": True,
            "cursor_trainable_rows": 0 if cursor_batch is None else len(cursor_batch[0]),
            "durable_trainable_rows": len(rows),
            "durable_groups": len(rows) // 2,
            "policy_versions": facts["policy_versions"],
            "resumed_members": durable.stats()["resumed_members"],
            "commit_id": lease_id,
            "duplicate_commit_id": duplicate,
            "lease_after_commit": after_commit._lease is not None,
        }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
