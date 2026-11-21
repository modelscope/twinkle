import pytest

from twinkle.data_format import SampleResponse
from twinkle.data_format.sampling import SampledSequence, SamplingParams
from twinkle_agentic.async_rl.rollout_resume import ResumeStore, UnsafeResume
from twinkle_agentic.envs.localenv import LocalEnv
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout, _Episode


class _Template:
    max_length = 32
    truncation_strategy = "right"


def _rollout(store=None):
    return MultiTurnRollout(response_callback=lambda *args: None, template=_Template(), resume_store=store)


def _episode(rollout, state_key="run/group-0/episode-0/attempt-0", adapter_path=None):
    trajectory = {"messages": [{"role": "user", "content": "hello"}], "state_key": state_key}
    if adapter_path is not None:
        trajectory["adapter_path"] = adapter_path
    ep = _Episode("episode-0", trajectory, rollout._params(SamplingParams()))
    ep.ledger = rollout._new_ledger("episode-0")
    ep.ledger._pif = {
        "input_ids": [1, 2, 3],
        "labels": [-100, 2, 3],
        "completion_mask": [0, 1, 1],
        "messages": trajectory["messages"],
    }
    ep.ledger._turns = 1
    ep.budget = 3
    ep.spent = 1
    return ep


def test_multi_turn_episode_round_trip_from_store(tmp_path):
    store = ResumeStore(tmp_path, "run")
    source = _rollout(store)
    ep = _episode(source, adapter_path="/policies/policy-1.adapter")
    source._episodes[ep.episode_id] = ep
    source.checkpoint_episode(ep.episode_id)

    restored = _rollout(store)
    restored.restore_episode_from_store(ep.state_key)
    again = restored._episodes[ep.episode_id]
    assert again.state_key == ep.state_key
    assert again.budget == ep.budget and again.spent == ep.spent
    assert again.ledger.input_feature["input_ids"] == [1, 2, 3]
    assert again.ledger.turns == 1
    assert again.adapter_path == "/policies/policy-1.adapter"


def test_multi_turn_restore_refuses_effect_without_tool_checkpoint(tmp_path):
    store = ResumeStore(tmp_path, "run")
    source = _rollout(store)
    ep = _episode(source)
    source._episodes[ep.episode_id] = ep
    state = source.checkpoint_episode(ep.episode_id)

    pending_effect = ep.state_key + "/effect/1/abc"
    store.begin_effect(pending_effect, {"calls": []})
    with pytest.raises(UnsafeResume, match="pending tool effects"):
        _rollout(store).restore_episode(state)

    ep.completed_effects.add(pending_effect)
    safe_state = source.checkpoint_episode(ep.episode_id)
    restored_episode = _rollout(store).restore_episode(safe_state)
    assert restored_episode == ep.episode_id
    assert store.begin_effect(pending_effect, {"calls": []}) == {"recovered_from_checkpoint": True}

    uncheckpointed_effect = ep.state_key + "/effect/2/def"
    store.begin_effect(uncheckpointed_effect, {"calls": []})
    store.finish_effect(uncheckpointed_effect, {"observations": []})
    with pytest.raises(UnsafeResume, match="without a matching episode checkpoint"):
        _rollout(store).restore_episode(safe_state)

    ep.completed_effects.add(uncheckpointed_effect)
    final_state = source.checkpoint_episode(ep.episode_id)
    restored = _rollout(store).restore_episode_from_store(ep.state_key)
    assert restored == ep.episode_id
    assert pending_effect in final_state["completed_effects"]
    assert uncheckpointed_effect in store.completed_effects(ep.state_key)


def test_multi_turn_refuses_message_ledger_checkpoint():
    rollout = MultiTurnRollout(response_callback=lambda *args: None)
    ep = _episode(rollout)
    rollout._episodes[ep.episode_id] = ep
    with pytest.raises(UnsafeResume, match="lose sampled token ids"):
        rollout.checkpoint_episode(ep.episode_id)


def test_multi_turn_restores_localenv_at_turn_boundary(tmp_path):
    store = ResumeStore(tmp_path / "resume", "run")
    env = LocalEnv(root=str(tmp_path / "workspace"))
    episode_id = "episode-env"
    state_key = "run/group-0/attempt-0/member/0"
    env.reset(episode_id)
    env.step(episode_id, "write_file", {"path": "state.txt", "content": "durable-turn"})

    source = MultiTurnRollout(response_callback=lambda *args: None, template=_Template(), env=env, resume_store=store)
    ep = _episode(source, state_key)
    ep.episode_id = episode_id
    ep.trajectory["episode_id"] = episode_id
    ep.trajectory["state_key"] = state_key
    ep.env_open = True
    source._episodes[episode_id] = ep
    source.checkpoint_episode(episode_id)
    env.close(episode_id)

    restored = MultiTurnRollout(
        response_callback=lambda *args: None,
        template=_Template(),
        env=LocalEnv(root=str(tmp_path / "workspace")),
        resume_store=store,
    )
    restored.restore_episode_from_store(state_key)
    result = restored.env.step(episode_id, "read_file", {"path": "state.txt"})
    assert result.observation == "durable-turn"
    restored.env.close(episode_id)


def test_episode_policy_binding_overrides_the_samplers_mutable_default():
    class _Sampler:
        accepts_rollout_state = False

        def __init__(self):
            self.adapter_paths = []

        def sample(self, _inputs, sampling_params=None, *, adapter_path=None, **_kwargs):
            self.adapter_paths.append(adapter_path)
            return [
                SampleResponse(
                    sequences=[SampledSequence(stop_reason="stop", tokens=[7], logprobs=[[(7, -0.1)]], decoded="x")]
                )
            ]

    sampler = _Sampler()
    rollout = MultiTurnRollout(sampler=sampler)
    ep = _Episode("episode-pinned", {"adapter_path": "/policies/policy-1.adapter"}, rollout._params(SamplingParams()))
    rollout._generate(ep, type("Ledger", (), {"state": None, "input_feature": {}})(), SamplingParams())

    assert sampler.adapter_paths == ["/policies/policy-1.adapter"]


def test_next_turn_persists_the_completed_generation_boundary(tmp_path):
    store = ResumeStore(tmp_path, "run")
    rollout = _rollout(store)
    episode = _episode(rollout)
    rollout._episodes[episode.episode_id] = episode

    rows, status = rollout._next_turn(episode)

    assert not status.done
    assert rows[0]["turns"] == 1
    durable = store.load_episode(episode.state_key)
    assert durable["spent"] == 1
    assert durable["ledger"]["input_feature"]["input_ids"] == [1, 2, 3]
    assert durable["ledger"]["turns"] == 1
