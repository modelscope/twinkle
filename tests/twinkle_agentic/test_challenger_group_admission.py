# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-only coverage for group admission in Challenger's refill boundary."""

from collections import Counter

from twinkle.advantage.group_admission import GroupAdmissionConfig, GroupAdmissionPolicy
from twinkle_agentic.challenger.base import Challenger
from twinkle_agentic.envs.base import Env, StepResult


class _Env(Env):

    def step(self, tool_name, arguments):
        return StepResult()


class _Challenger(Challenger):

    def _launch(self):
        return False


def _trajectory(reward, answer='answer'):
    return {
        'rewards': reward,
        'messages': [{'role': 'assistant', 'content': answer}],
    }


def _admit(challenger, group):
    reasons = Counter()
    admitted = challenger._admitted_groups([group], reasons)
    return admitted, reasons


def test_default_keeps_existing_exact_dead_filter():
    challenger = _Challenger(envs=[_Env()])
    try:
        admitted, reasons = _admit(
            challenger,
            [_trajectory(0.5), _trajectory(0.5), _trajectory(0.5)],
        )
        assert admitted == []
        assert reasons == {'exact_dead': 1}
    finally:
        challenger.close()


def test_default_accepts_a_group_with_reward_spread():
    challenger = _Challenger(envs=[_Env()])
    group = [_trajectory(0.2), _trajectory(0.4), _trajectory(0.6)]
    try:
        admitted, reasons = _admit(challenger, group)
        assert admitted == [group]
        assert not reasons
    finally:
        challenger.close()


def test_resolution_aware_policy_rejects_near_tie_atomically():
    policy = GroupAdmissionPolicy(GroupAdmissionConfig(min_reward_range=0.02))
    challenger = _Challenger(envs=[_Env()], group_admission_policy=policy)
    try:
        admitted, reasons = _admit(
            challenger,
            [_trajectory(0.500), _trajectory(0.505), _trajectory(0.510)],
        )
        assert admitted == []
        assert reasons == {'near_tie': 1}
    finally:
        challenger.close()


def test_near_duplicate_gate_reads_assistant_completions():
    policy = GroupAdmissionPolicy(
        GroupAdmissionConfig(max_mean_pairwise_similarity=0.9),
        scorer=lambda left, right: 1.0 if left == right else 0.0,
    )
    challenger = _Challenger(envs=[_Env()], group_admission_policy=policy)
    try:
        admitted, reasons = _admit(
            challenger,
            [_trajectory(0.0, 'same'), _trajectory(0.5, 'same'), _trajectory(1.0, 'same')],
        )
        assert admitted == []
        assert reasons == {'redundant': 1}
    finally:
        challenger.close()
