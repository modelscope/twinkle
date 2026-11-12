import re
from typing import Any, Dict, List

from twinkle.data_format import user_data_get
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


class MathVerifyAccuracyReward(Reward):
    """Use the same math-verify parsing and equivalence check as AReaL."""

    def __init__(self, *, precision: int = 6, try_extract_without_anchor: bool = True):
        self.precision = precision
        self.try_extract_without_anchor = try_extract_without_anchor

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        from math_verify.grader import verify
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse

        extraction_config = (
            ExprExtractionConfig(try_extract_without_anchor=self.try_extract_without_anchor),
            LatexExtractionConfig(),
        )

        rewards = []
        for trajectory in trajectories:
            completion = ''
            for message in reversed(trajectory.get('messages', [])):
                if message.get('role') == 'assistant':
                    completion = str(message.get('content', ''))
                    break
            ground_truth = str(user_data_get(trajectory.get('user_data'), 'ground_truth', ''))
            try:
                # Disable signal-based timeouts because rollout rewards run on
                # the sampler's background event-loop thread.
                gold = parse(
                    ground_truth,
                    extraction_config=extraction_config,
                    parsing_timeout=None,
                )
                answer = parse(
                    completion,
                    extraction_config=extraction_config,
                    parsing_timeout=None,
                )
                if not gold or not answer:
                    rewards.append(0.0)
                    continue
                correct = verify(
                    gold,
                    answer,
                    float_rounding=self.precision,
                    timeout_seconds=None,
                )
                rewards.append(1.0 if correct else 0.0)
            except Exception:
                rewards.append(0.0)
        return rewards

    def metric_payload(
        self,
        trajectories: List[Dict[str, Any]],
        *,
        rewards: List[float],
        **kwargs,
    ) -> Dict[str, float]:
        return {'accuracy_reward': sum(rewards) / len(rewards)}


class GSM8KBrevityReward(Reward):
    """Reward concise completions that contain a parseable final answer."""

    def __init__(self, full_reward_length: int = 300, decay_length: int = 3000):
        self.full_reward_length = full_reward_length
        self.decay_length = decay_length

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = ''
            for message in reversed(trajectory.get('messages', [])):
                if message.get('role') == 'assistant':
                    completion = message.get('content', '')
                    break
            has_answer = _has_boxed(completion) or bool(re.search(r'####\s*[\-\d,\.]+', completion))
            if not has_answer:
                rewards.append(0.0)
                continue
            excess_length = max(0, len(completion) - self.full_reward_length)
            rewards.append(max(0.0, 1.0 - excess_length / self.decay_length))
        return rewards


class GSM8KAccuracyBrevityReward(Reward):
    """Sum GSM8K answer accuracy and brevity rewards."""

    def __init__(self, accuracy_weight: float = 1.0, brevity_weight: float = 1.0):
        self.accuracy_weight = accuracy_weight
        self.brevity_weight = brevity_weight
        self.accuracy_reward = GSM8KAccuracyReward()
        self.brevity_reward = GSM8KBrevityReward()

    def components(self, trajectories: List[Dict[str, Any]]) -> tuple[list[float], list[float]]:
        return self.accuracy_reward(trajectories), self.brevity_reward(trajectories)

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        accuracy_rewards, brevity_rewards = self.components(trajectories)
        return [
            self.accuracy_weight * accuracy + self.brevity_weight * brevity
            for accuracy, brevity in zip(accuracy_rewards, brevity_rewards)
        ]

    def metric_payload(
        self,
        trajectories: List[Dict[str, Any]],
        *,
        rewards: List[float],
        **kwargs,
    ) -> Dict[str, float]:
        accuracy_rewards, brevity_rewards = self.components(trajectories)
        size = len(trajectories)
        return {
            'total_reward': sum(rewards) / size,
            'accuracy_reward': sum(accuracy_rewards) / size,
            'brevity_reward': sum(brevity_rewards) / size,
        }


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
