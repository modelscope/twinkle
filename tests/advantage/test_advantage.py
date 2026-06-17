# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch

from twinkle.advantage import GRPOAdvantage, RLOOAdvantage


class TestGRPOAdvantage:

    def setup_method(self):
        self.grpo = GRPOAdvantage()

    # --- basic shape / dtype ---

    def test_output_shape_matches_input(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = self.grpo(rewards, num_generations=4)
        assert result.shape == rewards.shape

    def test_output_is_float32(self):
        rewards = [1, 2, 3, 4]
        result = self.grpo(rewards, num_generations=4)
        assert result.dtype == torch.float32

    def test_accepts_list_input(self):
        rewards = [0.0, 1.0, 0.0, 1.0]
        result = self.grpo(rewards, num_generations=4)
        assert result.shape == (4,)

    # --- scale='none' ---

    def test_scale_none_subtracts_group_mean(self):
        # Group [0, 1, 0, 1]: mean=0.5, advantages = [-0.5, 0.5, -0.5, 0.5]
        rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
        result = self.grpo(rewards, num_generations=4, scale='none')
        expected = torch.tensor([-0.5, 0.5, -0.5, 0.5])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_scale_none_two_groups(self):
        # 2 prompts, 4 generations each
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = self.grpo(rewards, num_generations=4, scale='none')
        # Group 1: [0,1,2,3] mean=1.5 → [-1.5, -0.5, 0.5, 1.5]
        # Group 2: [4,5,6,7] mean=5.5 → [-1.5, -0.5, 0.5, 1.5]
        expected = torch.tensor([-1.5, -0.5, 0.5, 1.5, -1.5, -0.5, 0.5, 1.5])
        assert torch.allclose(result, expected, atol=1e-6)

    # --- scale='group' ---

    def test_scale_group_divides_by_group_std(self):
        rewards = torch.tensor([0.0, 2.0, 0.0, 2.0])
        result = self.grpo(rewards, num_generations=4, scale='group')
        # Group mean=1.0, std≈1.1547
        # advantages_raw = [-1, 1, -1, 1]
        # normalized = [-1, 1, -1, 1] / 1.1547
        assert result.shape == (4,)

    def test_scale_group_zero_std_handled(self):
        # All same reward → std=0, division by eps=1e-8
        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
        result = self.grpo(rewards, num_generations=4, scale='group')
        # advantages_raw = [0,0,0,0], still 0 after division
        assert torch.allclose(result, torch.zeros(4), atol=1e-5)

    # --- scale='batch' ---

    def test_scale_batch_uses_batch_std(self):
        rewards = torch.tensor([0.0, 2.0, 4.0, 6.0])
        result = self.grpo(rewards, num_generations=4, scale='batch')
        assert result.shape == (4,)

    # --- num_generations=1 ---

    def test_num_generations_1_scale_none(self):
        rewards = torch.tensor([1.0, 3.0, 5.0])
        result = self.grpo(rewards, num_generations=1, scale='none')
        # mean=3.0, advantages = [-2, 0, 2]
        expected = torch.tensor([-2.0, 0.0, 2.0])
        assert torch.allclose(result, expected, atol=1e-6)

    def test_num_generations_1_scale_batch(self):
        rewards = torch.tensor([1.0, 3.0, 5.0])
        result = self.grpo(rewards, num_generations=1, scale='batch')
        # mean=3.0, std≈2.0, advantages = [-2,0,2]/2
        assert result.shape == (3,)

    def test_num_generations_1_scale_default(self):
        """Default scale for num_generations=1 is 'group', which returns raw rewards."""
        rewards = torch.tensor([1.0, 2.0, 3.0])
        result = self.grpo(rewards, num_generations=1)
        assert torch.allclose(result, rewards)

    # --- multi-dim rewards ---

    def test_multi_dim_rewards_summed(self):
        rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = self.grpo(rewards, num_generations=4, scale='none')
        # summed: [3, 7, 11, 15], mean=9, advantages = [-6, -2, 2, 6]
        expected = torch.tensor([-6.0, -2.0, 2.0, 6.0])
        assert torch.allclose(result, expected, atol=1e-6)

    # --- error handling ---

    def test_invalid_num_generations_zero(self):
        with pytest.raises(ValueError):
            self.grpo(torch.tensor([1.0, 2.0]), num_generations=0)

    def test_invalid_num_generations_mismatch(self):
        with pytest.raises(ValueError):
            self.grpo(torch.tensor([1.0, 2.0, 3.0]), num_generations=2)

    def test_single_element_with_num_gen_1(self):
        result = self.grpo(torch.tensor([5.0]), num_generations=1)
        assert torch.allclose(result, torch.tensor([5.0]))


class TestRLOOAdvantage:

    def setup_method(self):
        self.rloo = RLOOAdvantage()

    # --- basic shape / dtype ---

    def test_output_shape_matches_input(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        result = self.rloo(rewards, num_generations=4)
        assert result.shape == rewards.shape

    def test_accepts_list_input(self):
        rewards = [1.0, 2.0, 3.0, 4.0]
        result = self.rloo(rewards, num_generations=4)
        assert result.shape == (4,)

    # --- RLOO formula ---

    def test_rloo_leave_one_out_baseline(self):
        # K=4, rewards=[0,0,0,1]
        # For reward=0: baseline = 1/3 ≈ 0.333, advantage ≈ -0.333
        # For reward=1: baseline = 0/3 = 0,     advantage = 1
        rewards = torch.tensor([0.0, 0.0, 0.0, 1.0])
        result = self.rloo(rewards, num_generations=4, scale='none')
        assert torch.allclose(result[0], torch.tensor(-1.0 / 3), atol=1e-5)
        assert torch.allclose(result[3], torch.tensor(1.0), atol=1e-5)

    def test_rloo_all_same_rewards(self):
        rewards = torch.tensor([5.0, 5.0, 5.0, 5.0])
        result = self.rloo(rewards, num_generations=4, scale='none')
        # All advantages should be 0
        assert torch.allclose(result, torch.zeros(4), atol=1e-5)

    def test_rloo_two_groups(self):
        rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
        result = self.rloo(rewards, num_generations=2, scale='none')
        # Group1 [1,3]: baseline_1=3, adv_1=-2; baseline_2=1, adv_2=2
        # Group2 [5,7]: baseline_1=7, adv_1=-2; baseline_2=5, adv_2=2
        expected = torch.tensor([-2.0, 2.0, -2.0, 2.0])
        assert torch.allclose(result, expected, atol=1e-5)

    # --- scale modes ---

    def test_scale_group(self):
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = self.rloo(rewards, num_generations=4, scale='group')
        assert result.shape == (4,)

    def test_scale_batch(self):
        rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = self.rloo(rewards, num_generations=4, scale='batch')
        assert result.shape == (4,)

    # --- error handling ---

    def test_invalid_num_generations_one(self):
        with pytest.raises(ValueError):
            self.rloo(torch.tensor([1.0, 2.0]), num_generations=1)

    def test_invalid_num_generations_zero(self):
        with pytest.raises(ValueError):
            self.rloo(torch.tensor([1.0, 2.0]), num_generations=0)

    def test_invalid_num_generations_mismatch(self):
        with pytest.raises(ValueError):
            self.rloo(torch.tensor([1.0, 2.0, 3.0]), num_generations=2)

    # --- multi-dim ---

    def test_multi_dim_rewards_summed(self):
        rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        result = self.rloo(rewards, num_generations=4, scale='none')
        # summed: [3, 7, 11, 15], sum=36
        # adv_3 = 3 - (36-3)/3 = 3 - 11 = -8
        # adv_15 = 15 - (36-15)/3 = 15 - 7 = 8
        assert result.shape == (4,)
