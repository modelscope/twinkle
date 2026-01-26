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


class SelfCognitionProcessor(Preprocessor):

    def __call__(self, row) -> Trajectory:
        problem = row['query']
        solution = row['response']
        messages = [
            Message(role='user', content=problem),
            Message(role='assistant', content=solution),
        ]
        return Trajectory(messages=messages)