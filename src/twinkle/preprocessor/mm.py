# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Dict, List, Optional

from twinkle.data_format import Message, Trajectory
from .base import Preprocessor


class CLEVRProcessor(Preprocessor):
    """Preprocessor for CLEVR-CoGenT visual reasoning dataset (prompt-only, for GRPO).

    Dataset fields: image (PIL.Image or dict), problem (str), solution (str with <answer> tags)
    Produces prompt-only trajectories with image in the user message and
    ground truth stored in user_data for reward computation.

    For fast ``.map()`` performance, call ``dataset.cast_column('image', decode=False)``
    before mapping so that images stay as Arrow-native bytes dicts.
    """

    DEFAULT_SYSTEM = ('A conversation between User and Assistant. The user asks a question, '
                      'and the Assistant solves it. The assistant first thinks about the reasoning '
                      'process in the mind and then provides the user with the answer. The reasoning '
                      'process and answer are enclosed within <think> </think> and <answer> </answer> '
                      'tags, respectively, i.e., <think> reasoning process here </think>'
                      '<answer> answer here </answer>')

    def __init__(self, system: Optional[str] = None):
        self.system = system if system is not None else self.DEFAULT_SYSTEM

    @staticmethod
    def extract_ground_truth(solution: str) -> str:
        """Extract answer text from <answer>...</answer> tags."""
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', solution, re.DOTALL)
        return match.group(1).strip() if match else solution.strip()

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        rows = [self.preprocess(row) for row in rows]
        rows = self.map_row_to_col(rows)
        return rows

    def preprocess(self, row) -> Trajectory:
        image = row['image']
        problem = row['problem']
        solution = row.get('solution', '')
        ground_truth = self.extract_ground_truth(solution)

        messages = [
            Message(role='system', content=[{
                'type': 'text',
                'text': self.system
            }]),
            Message(role='user', content=[
                {
                    'type': 'image',
                    'image': image
                },
                {
                    'type': 'text',
                    'text': problem
                },
            ]),
        ]
        return Trajectory(
            messages=messages,
            user_data=[('ground_truth', ground_truth), ('solution', solution)],
        )
