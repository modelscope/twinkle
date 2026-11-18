# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from twinkle.advantage import GRPOAdvantage, StateGraphAdvantage


def _trajectory(pre_state, post_state, success, *, version=1, span=(1, 2)):
    return {
        'labels': [0] * 6,
        'turn_credit_status': {
            'available': True,
            'fallback_reason': None,
        },
        'turn_credits': [{
            'turn_id': 1,
            'action_token_start': span[0],
            'action_token_end': span[1],
            'pre_state_key': pre_state,
            'post_state_key': post_state,
            'policy_version': version,
            'terminal_success': success,
        }],
    }


class TestStateGraphAdvantage:

    def test_adds_state_relative_credit_to_action_tokens(self):
        trajectories = [
            _trajectory('s0', 'success', True),
            _trajectory('s0', 'dead', None),
            _trajectory('s1', 'success', True),
            _trajectory('s1', 'dead', None),
        ]
        advantage = StateGraphAdvantage()
        result = advantage([1.0, 0.0, 1.0, 0.0], trajectories, num_generations=4)

        episode = GRPOAdvantage()(torch.tensor([1.0, 0.0, 1.0, 0.0]), num_generations=4)
        assert torch.allclose(result[:, 0], episode)
        assert torch.all(result[0, 1] > episode[0])
        assert torch.all(result[1, 1] < episode[1])
        assert advantage.last_diagnostics.comparable_state_group_count == 2
        assert advantage.last_diagnostics.transition_count == 4
        assert advantage.last_diagnostics.fallback_counts == {}

    def test_missing_success_is_exact_grpo_fallback(self):
        trajectories = [
            _trajectory('s0', 'dead0', None),
            _trajectory('s0', 'dead1', None),
            _trajectory('s0', 'dead2', None),
            _trajectory('s0', 'dead3', None),
        ]
        rewards = [0.0, 1.0, 0.0, 2.0]
        advantage = StateGraphAdvantage()
        result = advantage(rewards, trajectories, num_generations=4)
        episode = GRPOAdvantage()(torch.tensor(rewards), num_generations=4)

        assert torch.allclose(result, episode.unsqueeze(1).repeat(1, 6))
        assert advantage.last_diagnostics.fallback_counts['terminal_success_missing'] == 1

    def test_policy_version_mismatch_is_exact_grpo_fallback(self):
        trajectories = [
            _trajectory('s0', 'success', True, version=1),
            _trajectory('s0', 'dead', None, version=2),
        ]
        rewards = [1.0, 0.0]
        advantage = StateGraphAdvantage()
        result = advantage(rewards, trajectories, num_generations=2)
        episode = GRPOAdvantage()(torch.tensor(rewards), num_generations=2)

        assert torch.allclose(result, episode.unsqueeze(1).repeat(1, 6))
        assert advantage.last_diagnostics.fallback_counts['policy_version_mismatch'] == 1

    def test_partially_missing_policy_version_is_exact_grpo_fallback(self):
        trajectories = [
            _trajectory('s0', 'success', True, version=1),
            _trajectory('s0', 'dead', None, version=None),
        ]
        rewards = [1.0, 0.0]
        advantage = StateGraphAdvantage()
        result = advantage(rewards, trajectories, num_generations=2)
        episode = GRPOAdvantage()(torch.tensor(rewards), num_generations=2)

        assert torch.allclose(result, episode.unsqueeze(1).repeat(1, 6))
        assert advantage.last_diagnostics.fallback_counts['policy_version_missing'] == 1

    def test_fully_missing_policy_version_is_exact_grpo_fallback(self):
        trajectories = [
            _trajectory('s0', 'success', True, version=None),
            _trajectory('s0', 'dead', None, version=None),
        ]
        rewards = [1.0, 0.0]
        advantage = StateGraphAdvantage()
        result = advantage(rewards, trajectories, num_generations=2)
        episode = GRPOAdvantage()(torch.tensor(rewards), num_generations=2)

        assert torch.allclose(result, episode.unsqueeze(1).repeat(1, 6))
        assert advantage.last_diagnostics.fallback_counts['policy_version_missing'] == 1

    def test_policy_version_zero_is_available(self):
        trajectories = [
            _trajectory('s0', 'success', True, version=0),
            _trajectory('s0', 'dead', None, version=0),
        ]
        advantage = StateGraphAdvantage()
        result = advantage([1.0, 0.0], trajectories, num_generations=2)

        assert result.shape == (2, 6)
        assert advantage.last_diagnostics.fallback_counts == {}

    @pytest.mark.parametrize('field,value,reason', [
        ('pre_state_key', 1, 'state_key_invalid'),
        ('policy_version', [], 'policy_version_invalid'),
        ('terminal_success', 'yes', 'terminal_success_invalid'),
        ('observation_token_start', 1, 'observation_span_invalid'),
    ])
    def test_invalid_contract_field_is_exact_grpo_fallback(self, field, value, reason):
        trajectories = [_trajectory('s0', 'success', True), _trajectory('s0', 'dead', None)]
        trajectories[0]['turn_credits'][0][field] = value
        rewards = [1.0, 0.0]
        advantage = StateGraphAdvantage()
        result = advantage(rewards, trajectories, num_generations=2)
        episode = GRPOAdvantage()(torch.tensor(rewards), num_generations=2)

        assert torch.allclose(result, episode.unsqueeze(1).repeat(1, 6))
        assert advantage.last_diagnostics.fallback_counts[reason] == 1

    def test_insufficient_state_support_leaves_only_episode_credit(self):
        trajectories = [
            _trajectory('s0', 'success', True),
            _trajectory('s1', 'dead', None),
        ]
        rewards = [1.0, 0.0]
        advantage = StateGraphAdvantage()
        result = advantage(rewards, trajectories, num_generations=2)
        episode = GRPOAdvantage()(torch.tensor(rewards), num_generations=2)

        assert torch.allclose(result, episode.unsqueeze(1).repeat(1, 6))
        assert advantage.last_diagnostics.comparable_state_group_count == 0

    def test_invalid_span_skips_that_row(self):
        trajectories = [
            _trajectory('s0', 'success', True, span=(0, 1)),
            _trajectory('s0', 'dead', None, span=(7, 9)),
        ]
        rewards = [1.0, 0.0]
        advantage = StateGraphAdvantage()
        result = advantage(rewards, trajectories, num_generations=2)
        episode = GRPOAdvantage()(torch.tensor(rewards), num_generations=2)

        assert torch.allclose(result, episode.unsqueeze(1).repeat(1, 6))
        assert advantage.last_diagnostics.fallback_counts['action_span_invalid'] == 1

    def test_validation(self):
        with pytest.raises(ValueError, match='at least two'):
            StateGraphAdvantage()([1.0], [_trajectory('s', 't', True)], num_generations=1)
        with pytest.raises(ValueError, match='complete groups'):
            StateGraphAdvantage()([1.0], [_trajectory('s', 't', True)], num_generations=2)
        with pytest.raises(ValueError, match='min_state_support'):
            StateGraphAdvantage(min_state_support=1)
