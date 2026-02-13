import re
from typing import Any, Dict, List

from twinkle.reward.base import Reward


class GSM8KAccuracyReward(Reward):
    """Accuracy reward for GSM8K: checks if the model's answer matches ground truth.

    Extracts the last '#### <number>' from model output and compares with ground truth.
    Returns 1.0 for correct, 0.0 for incorrect.
    """

    @staticmethod
    def extract_answer(completion: str) -> str:
        """Extract the last #### answer from model completion."""
        # Only check last 500 chars for efficiency
        text = completion[-500:] if len(completion) > 500 else completion
        matches = re.findall(r'####\s*([\-\d,\.\s]+)', text)
        if matches:
            return matches[-1].replace(',', '').replace(' ', '').strip()
        return ''

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            # Get model completion (last assistant message)
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break

            # Get ground truth from user_data
            user_data = trajectory.get('user_data')
            for item in user_data:
                if item[0] == 'ground_truth':
                    gt = item[1]
                    break

            predicted = self.extract_answer(completion)

            # Numeric comparison
            correct = False
            if predicted and gt:
                try:
                    correct = abs(float(predicted) - float(gt)) < 1e-5
                except (ValueError, OverflowError):
                    correct = predicted == gt

            rewards.append(1.0 if correct else 0.0)
        return rewards


class GSM8KFormatReward(Reward):
    """Format reward: checks if output contains <think>...</think> tag.

    Returns 1.0 if format is correct, 0.0 otherwise.
    """

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            messages = trajectory.get('messages', [])
            completion = ''
            for msg in reversed(messages):
                if msg.get('role') == 'assistant':
                    completion = msg.get('content', '')
                    break
            has_think = bool(re.search(r'<think>.*?</think>', completion, re.DOTALL))
            has_answer = bool(re.search(r'####\s*[\-\d,\.]+', completion))
            rewards.append(1.0 if (has_think and has_answer) else 0.0)
        return rewards
