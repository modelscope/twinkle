# Copyright (c) ModelScope Contributors. All rights reserved.
import re

from twinkle.data_format import Message, Trajectory
from .base import Preprocessor


class CompetitionMathProcessor(Preprocessor):

    def __call__(self, row) -> Trajectory:
        problem = row['problem']
        solution = row['solution']
        messages = [
            Message(role='user', content=problem),
            Message(role='assistant', content=solution),
        ]
        return Trajectory(messages=messages)


class CompetitionMathGRPOProcessor(Preprocessor):

    def __call__(self, row) -> Trajectory:
        problem = row['problem']
        solution = row['solution']
        messages = [
            Message(
                role='system',
                content='You are a helpful math assistant. Respond with only the final answer in the form '
                '\\boxed{...} and nothing else.'),
            Message(role='user', content=problem),
            Message(role='assistant', content=''),
        ]
        return Trajectory(messages=messages, user_data=[('solution', solution)])


class SelfCognitionProcessor(Preprocessor):

    def __init__(self, model_name, model_author):
        self.model_name = model_name
        self.model_author = model_author

    def __call__(self, row) -> Trajectory:
        problem = row['query'].replace('{{NAME}}', self.model_name).replace('{{AUTHOR}}', self.model_author)
        solution = row['response'].replace('{{NAME}}', self.model_name).replace('{{AUTHOR}}', self.model_author)
        messages = [
            Message(role='system', content='You are a helpful assistant.'),
            Message(role='user', content=problem),
            Message(role='assistant', content=solution),
        ]
        return Trajectory(messages=messages)


class AlpacaProcessor(Preprocessor):

    def __call__(self, row) -> Trajectory:
        instruction = row.get('instruction') or ''
        input_text = row.get('input') or ''
        output_text = row.get('output') or ''
        prompt = instruction if not input_text else f'{instruction}\n{input_text}'
        messages = [
            Message(role='user', content=prompt),
            Message(role='assistant', content=output_text),
        ]
        return Trajectory(messages=messages)


class CountdownProcessor(Preprocessor):
    system_prompt = ('You are a helpful assistant. You first thinks about the reasoning process '
                     'in the mind and then provides the user with the answer.')

    def __call__(self, row) -> Trajectory:
        nums = row.get('nums', [])
        target = row.get('response', row.get('target', 0))

        query = f"""Using the numbers {nums}, create an equation that equals {target}.
You can use basic arithmetic operations (+, -, *, /) and each number can only be used once.
Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags,
for example <answer> (1 + 2) / 3 * 4 = 4 </answer>."""

        messages = [
            Message(role='system', content=self.system_prompt),
            Message(role='user', content=query),
        ]
        return Trajectory(messages=messages, user_data=[{'target': target, 'nums': nums}])


class GSM8KProcessor(Preprocessor):
    """Preprocessor for GSM8K dataset.

    GSM8K fields: question (str), answer (str ending with '#### <number>')
    Extracts the ground truth number and stores it in user_data for reward.
    """

    system_prompt = ('You are a helpful math assistant. Solve the problem step by step. '
                     'Show your reasoning in <think> </think> tags, then give the final '
                     'numerical answer after ####.\n'
                     'For example:\n<think> ... reasoning ... </think>\n#### 42')

    def extract_ground_truth(self, answer_str: str) -> str:
        """Extract the number after '####' from GSM8K answer."""
        match = re.search(r'####\s*([\-\d,\.]+)', answer_str)
        if match:
            return match.group(1).replace(',', '').strip()
        return ''

    def __call__(self, row) -> Trajectory:
        question = row['question']
        answer = row.get('answer', '')
        ground_truth = self.extract_ground_truth(answer)

        messages = [
            Message(role='system', content=self.system_prompt),
            Message(role='user', content=question),
        ]
        return Trajectory(
            messages=messages,
            user_data=[('ground_truth', ground_truth)],
        )
