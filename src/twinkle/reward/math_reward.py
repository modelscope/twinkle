# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import List, Union

from twinkle import remote_class, remote_function
from twinkle.reward.base import Reward
from twinkle.data_format import Trajectory


@remote_class()
class MathReward(Reward):

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
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
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
        except Exception: # noqa
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

    @remote_function()
    def __call__(self, trajectories: List[Trajectory], ground_truths: List[Trajectory]):
        rewards = []
        def _last_content(traj):
            # Trajectories can be dicts after serialization in distributed runs.
            if isinstance(traj, dict):
                return traj['messages'][-1]['content']
            return traj.messages[-1].content

        def _ground_truth_content(traj):
            if isinstance(traj, dict):
                user_data = traj.get('user_data')
                if isinstance(user_data, list):
                    for item in user_data:
                        if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == self.ground_truth_key:
                            return item[1]
            return _last_content(traj)

        predictions = [_last_content(trajectory) for trajectory in trajectories]
        ground_truths = [_ground_truth_content(trajectory) for trajectory in ground_truths]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathReward.extract_boxed_result(prediction)
            ground_truth = MathReward.extract_boxed_result(ground_truth)
            reward = MathReward.compare_consecutive(prediction, ground_truth)
            reward = 1.0 if reward else -1.0
            rewards.append(float(reward))
        return rewards
