import re
from typing import Any, Dict, List

from twinkle.reward.base import Reward


def _extract_last_boxed(text: str) -> str:
    """Extract content from the last \\boxed{...}, handling nested braces."""
    idx = text.rfind('\\boxed{')
    if idx == -1:
        return ''
    start = idx + len('\\boxed{')
    depth = 1
    j = start
    while j < len(text) and depth > 0:
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
        j += 1
    if depth == 0:
        return text[start:j - 1].strip()
    return ''


def _has_boxed(text: str) -> bool:
    """Check whether *text* contains a valid \\boxed{...} (nested-brace aware)."""
    return bool(_extract_last_boxed(text))


class GSM8KAccuracyReward(Reward):
    """Accuracy reward for GSM8K: checks if the model's answer matches ground truth.

    Extracts the answer from \\boxed{} (preferred) or #### format.
    Returns 1.0 for correct, 0.0 for incorrect.
    """

    @staticmethod
    def extract_answer(completion: str) -> str:
        """Extract the answer from model completion, preferring \\boxed{} over ####."""
        text = completion[-500:] if len(completion) > 500 else completion
        boxed = _extract_last_boxed(text)
        if boxed:
            return boxed.replace(',', '').replace(' ', '').strip()
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
            user_data = trajectory.get('user_data') or []
            gt = ''
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
    """Format reward: checks if output contains \\boxed{} or #### answer format.

    Returns 1.0 if a valid answer format is present, 0.0 otherwise.
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
            has_answer = bool(_has_boxed(completion) or re.search(r'####\s*[\-\d,\.]+', completion))
            rewards.append(1.0 if has_answer else 0.0)
        return rewards
