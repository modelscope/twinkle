# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import List

from twinkle.data_format import Trajectory
from twinkle.reward.base import Reward


class FormatReward(Reward):

    @staticmethod
    def format_reward(completion: str) -> float:
        """Format reward: checks <think> and <answer> tags."""
        has_think = bool(re.search(r"<think>.*?</think>", completion, re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", completion, re.DOTALL))
        return 1.0 if (has_think and has_answer) else 0.0

    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ""
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            fmt_reward = self.format_reward(completion)
            rewards.append(fmt_reward)
        return rewards

