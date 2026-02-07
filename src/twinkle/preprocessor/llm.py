# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Preprocessor
from twinkle.data_format import Trajectory, Message


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
            Message(role='system', content='You are a helpful math assistant. Respond with only the final answer in the form \\boxed{...} and nothing else.'),
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
        prompt = instruction if not input_text else f"{instruction}\n{input_text}"
        messages = [
            Message(role='user', content=prompt),
            Message(role='assistant', content=output_text),
        ]
        return Trajectory(messages=messages)

class CountdownProcessor(Preprocessor):
    system_prompt = (
        "You are a helpful assistant. You first thinks about the reasoning process "
        "in the mind and then provides the user with the answer."
    )
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
            Message(role='assistant', content=''),
        ]
        return Trajectory(messages=messages, user_data=[{'target': target, 'nums': nums}])