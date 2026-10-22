# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, List, Union

from twinkle.data_format import Trajectory, ground_truth, last_assistant_text
from twinkle.reward.base import Reward


class MathReward(Reward):
    """Boxed-answer math accuracy: the gold comes from the trajectory's ``user_data``."""

    def __init__(self, ground_truth_key: str = 'solution'):
        self.ground_truth_key = ground_truth_key

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        """Extract content from \\boxed{...}, handling nested braces."""
        idx = text.rfind('\\boxed{')
        if idx == -1:
            return text
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
        return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:  # noqa
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathReward.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathReward.parse_expression(latex) for latex in cleaned_list]
        if parsed_exprs[0] is None or parsed_exprs[1] is None:
            # Fallback to cleaned string comparison when LaTeX parsing fails.
            return cleaned_list[0] == cleaned_list[1]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, trajectories: List[Trajectory], **kwargs: Any) -> List[float]:
        rewards = []
        for trajectory in trajectories:
            prediction = last_assistant_text(trajectory)
            # The gold lives in the trajectory's user_data (or flat column); an absent key falls
            # back to scoring the completion against itself (compare degenerates, never raises).
            # The test is `is None`: a gold of 0 is a real label, not a missing one.
            gold = ground_truth(trajectory, self.ground_truth_key)
            if gold is None:
                gold = prediction
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if isinstance(gold, str) and '# Answer' in gold:
                gold = gold.split('# Answer')[1]
            prediction = prediction.strip()
            gold = str(gold).strip()
            prediction = MathReward.extract_boxed_result(prediction)
            gold = MathReward.extract_boxed_result(gold)
            reward = MathReward.compare_consecutive(prediction, gold)
            rewards.append(float(1.0 if reward else 0.0))
        return rewards
