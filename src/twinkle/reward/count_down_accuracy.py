# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import List

from twinkle.data_format import Trajectory
from twinkle.reward.base import Reward


class CountDownAccuracy(Reward):

    @staticmethod
    def countdown_accuracy_reward(completion: str, target: int, nums: List[int]) -> float:
        """Accuracy reward: checks if equation is correct."""
        try:
            match = re.search(r'<answer>(.*?)<\/answer>', completion)
            if match is None:
                return 0.0
            equation = match.group(1).strip()
            if '=' in equation:
                equation = equation.split('=')[0]
            used_numbers = [int(n) for n in re.findall(r'\d+', equation)]
            if sorted(used_numbers) != sorted(nums):
                return 0.0
            if not re.match(r'^[\d+\-*/().\s]+$', equation):
                return 0.0
            result = eval(equation, {'__builtins__': None}, {})
            return 1.0 if abs(float(result) - float(target)) < 1e-5 else 0.0
        except Exception: # noqa
            return 0.0

    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ""
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            user_data = trajectory.get('user_data', [{}])
            data = user_data[0] if isinstance(user_data, list) and user_data else {}
            target = data.get('target', 0)
            nums = data.get('nums', [])
            acc_reward = self.countdown_accuracy_reward(completion, target, nums)
            rewards.append(acc_reward)
        return rewards

