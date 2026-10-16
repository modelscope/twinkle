# Copyright (c) ModelScope Contributors. All rights reserved.
"""Accuracy reward for the Answer-style format used by DAPO-Math."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from twinkle.data_format import Trajectory, user_data_get
from twinkle.reward.base import Reward
from twinkle.reward.math_reward import MathReward

_ANSWER_LINE = re.compile(r'^\s*Answer\s*:\s*(.+?)\s*$', re.IGNORECASE | re.MULTILINE)


class DAPOMathAccuracyReward(Reward):
    """Compare the final ``Answer:`` or boxed result with the ground truth."""

    def __call__(self, trajectories: list[Trajectory], **kwargs: Any) -> list[float]:
        return [self._score(trajectory) for trajectory in trajectories]

    def metric_payload(
        self,
        trajectories: list[Trajectory],
        *,
        rewards: list[float],
        **kwargs: Any,
    ) -> dict[str, float]:
        return {'accuracy_reward': sum(rewards) / len(rewards)}

    @classmethod
    def _score(cls, trajectory: Trajectory) -> float:
        completion = cls._last_assistant_content(trajectory)
        prediction = cls.extract_answer(completion)
        ground_truth = str(user_data_get(trajectory.get('user_data'), 'ground_truth', '')).strip()
        if not prediction or not ground_truth:
            return 0.0
        if cls._decimal_equal(prediction, ground_truth):
            return 1.0
        return float(MathReward.compare_consecutive(prediction, ground_truth))

    @staticmethod
    def extract_answer(completion: str) -> str:
        if '\\boxed{' in completion:
            return MathReward.extract_boxed_result(completion).strip()
        matches = _ANSWER_LINE.findall(completion)
        if not matches:
            return ''
        answer = matches[-1].strip().rstrip('.')
        if len(answer) >= 2 and answer.startswith('$') and answer.endswith('$'):
            answer = answer[1:-1].strip()
        return answer

    @staticmethod
    def _last_assistant_content(trajectory: Trajectory) -> str:
        for message in reversed(trajectory.get('messages', [])):
            if message.get('role') == 'assistant':
                return str(message.get('content', ''))
        return ''

    @staticmethod
    def _decimal_equal(first: str, second: str) -> bool:
        try:
            return Decimal(first.replace(',', '')) == Decimal(second.replace(',', ''))
        except InvalidOperation:
            return False


class DAPOMathReward(Reward):
    """DAPO math training reward with token-level overlong shaping.

    Accuracy remains a ``0/1`` diagnostic metric, while the optimization score
    is ``+1/-1`` with a linear penalty near the response-length limit.
    """

    def __init__(
        self,
        max_response_length: int,
        overlong_buffer_length: int,
        overlong_penalty_factor: float = 1.0,
        score_tail_chars: int = 300,
    ):
        if max_response_length <= 0:
            raise ValueError('max_response_length must be positive')
        if overlong_buffer_length <= 0 or overlong_buffer_length > max_response_length:
            raise ValueError('overlong_buffer_length must be in [1, max_response_length]')
        if overlong_penalty_factor < 0:
            raise ValueError('overlong_penalty_factor must be non-negative')
        if score_tail_chars <= 0:
            raise ValueError('score_tail_chars must be positive')
        self.max_response_length = max_response_length
        self.overlong_buffer_length = overlong_buffer_length
        self.overlong_penalty_factor = overlong_penalty_factor
        self.score_tail_chars = score_tail_chars

    def components(self, trajectories: list[Trajectory]) -> tuple[list[float], list[float]]:
        accuracy_rewards = [self._accuracy(trajectory) for trajectory in trajectories]
        overlong_rewards = [self._overlong_reward(trajectory) for trajectory in trajectories]
        return accuracy_rewards, overlong_rewards

    def __call__(self, trajectories: list[Trajectory], **kwargs: Any) -> list[float]:
        accuracy_rewards, overlong_rewards = self.components(trajectories)
        return [(1.0 if accuracy else -1.0) + overlong
                for accuracy, overlong in zip(accuracy_rewards, overlong_rewards)]

    def metric_payload(
        self,
        trajectories: list[Trajectory],
        *,
        rewards: list[float],
        **kwargs: Any,
    ) -> dict[str, float]:
        accuracy_rewards, overlong_rewards = self.components(trajectories)
        size = len(trajectories)
        if size == 0:
            return {
                'total_reward': 0.0,
                'accuracy_reward': 0.0,
                'overlong_reward': 0.0,
                'overlong_ratio': 0.0,
            }
        return {
            'total_reward': sum(rewards) / size,
            'accuracy_reward': sum(accuracy_rewards) / size,
            'overlong_reward': sum(overlong_rewards) / size,
            'overlong_ratio': sum(value < 0 for value in overlong_rewards) / size,
        }

    def _accuracy(self, trajectory: Trajectory) -> float:
        completion = DAPOMathAccuracyReward._last_assistant_content(trajectory)
        scored_completion = completion[-self.score_tail_chars:]
        prediction = DAPOMathAccuracyReward.extract_answer(scored_completion)
        ground_truth = str(user_data_get(trajectory.get('user_data'), 'ground_truth', '')).strip()
        if not prediction or not ground_truth:
            return 0.0
        if DAPOMathAccuracyReward._decimal_equal(prediction, ground_truth):
            return 1.0
        return float(MathReward.compare_consecutive(prediction, ground_truth))

    def _overlong_reward(self, trajectory: Trajectory) -> float:
        if 'completion_length' not in trajectory:
            raise ValueError('DAPOMathReward requires token-level completion_length on every trajectory')
        completion_length = int(trajectory['completion_length'])
        expected_length = self.max_response_length - self.overlong_buffer_length
        exceed_length = completion_length - expected_length
        return min(
            -exceed_length / self.overlong_buffer_length * self.overlong_penalty_factor,
            0.0,
        )
