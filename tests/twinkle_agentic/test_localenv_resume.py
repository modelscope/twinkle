import os
import sys
import types

import pytest

if os.name == "nt" and "resource" not in sys.modules:
    resource = types.ModuleType("resource")
    resource.RLIMIT_AS = 0
    resource.setrlimit = lambda *args, **kwargs: None
    sys.modules["resource"] = resource

if "ray" not in sys.modules:
    ray = types.ModuleType("ray")
    ray.ObjectRef = type("ObjectRef", (), {})
    ray.is_initialized = lambda: False
    sys.modules["ray"] = ray

from twinkle_agentic.envs.localenv import LocalEnv  # noqa: E402
from twinkle_agentic.envs.resume import UnsafeCheckpoint  # noqa: E402


def test_localenv_workspace_round_trip_uses_a_new_session(tmp_path):
    source = LocalEnv(root=str(tmp_path))
    source.reset("episode-1")
    source.step("episode-1", "write_file", {"path": "nested/state.txt", "content": "turn-boundary"})
    state_key = "run-1/group-0/episode-1/attempt-0"

    source.checkpoint_episode("episode-1", state_key)
    snapshot = source.checkpoint_episode("episode-1", state_key)
    key_root = source._resume_root / source._state_dir_name(state_key)
    assert [path.name for path in key_root.iterdir() if path.is_dir()] == [snapshot["generation"]]
    source.close("episode-1")

    restored = LocalEnv(root=str(tmp_path))
    restored.restore_episode("episode-1", state_key, snapshot)
    result = restored.step("episode-1", "read_file", {"path": "nested/state.txt"})
    assert result.observation == "turn-boundary"

    restored.discard_episode_state(state_key)
    assert not restored._resume_root.exists() or not any(restored._resume_root.iterdir())
    restored.close("episode-1")


def test_localenv_resume_preserves_scored_zero_reward(tmp_path):
    source = LocalEnv(root=str(tmp_path))
    source.reset("episode-1")
    session = source._session("episode-1")
    session.rewarded = True
    session.reward = 0.0
    state_key = "run-1/group-0/episode-1/attempt-0"
    snapshot = source.checkpoint_episode("episode-1", state_key)
    source.close("episode-1")

    restored = LocalEnv(root=str(tmp_path))
    restored.restore_episode("episode-1", state_key, snapshot)
    assert restored.close("episode-1") == 0.0


def test_localenv_checkpoint_refuses_a_running_process(tmp_path):
    env = LocalEnv(root=str(tmp_path))
    env.reset("episode-1")
    env._session("episode-1").state.procs.add(object())
    with pytest.raises(UnsafeCheckpoint, match="running process"):
        env.checkpoint_episode("episode-1", "run-1/group-0/episode-1")
    env.close("episode-1")
