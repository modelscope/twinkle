# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Dict, List

from twinkle.data_format import last_assistant_text
from twinkle.reward.base import Reward


class MultiModalAccuracyReward(Reward):
    """Accuracy reward for multimodal VQA tasks (e.g. CLEVR).

    Compares the model's answer (inside <answer> tags) against
    the ground truth stored in user_data['ground_truth'].
    Falls back to math_verify symbolic verification when available.
    Returns 1.0 for correct, 0.0 for incorrect.
    """

    @staticmethod
    def extract_answer(text: str) -> str:
        """Extract the answer from <answer>...</answer> tags."""
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
        return match.group(1).strip() if match else ''

    def __call__(self, trajectories: List[Dict[str, Any]], **kwargs) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            completion = last_assistant_text(trajectory)

            user_data = trajectory.get('user_data', [])
            gt = ''
            solution = ''
            for item in user_data:
                if item[0] == 'ground_truth':
                    gt = str(item[1])
                elif item[0] == 'solution':
                    solution = str(item[1])

            predicted = self.extract_answer(completion)
            reward = 0.0

            # Try symbolic math verification first
            try:
                from math_verify import parse, verify
                answer = parse(completion)
                if float(verify(answer, parse(solution or gt))) > 0:
                    reward = 1.0
            except Exception:
                pass

            # Fallback: string matching
            if reward == 0.0 and predicted and gt:
                if predicted.strip().lower() == gt.strip().lower():
                    reward = 1.0
                else:
                    try:
                        if abs(float(predicted) - float(gt)) < 1e-5:
                            reward = 1.0
                    except (ValueError, OverflowError):
                        pass

            rewards.append(reward)
        return rewards
