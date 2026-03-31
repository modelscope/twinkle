# Copyright (c) ModelScope Contributors. All rights reserved.
"""
DPO (Direct Preference Optimization) Data Preprocessors.

These preprocessors convert various preference dataset formats into the standard
format required by Twinkle for DPO training.

DPO output format:
    - positive: Trajectory - chosen response trajectory
    - negative: Trajectory - rejected response trajectory
"""
from typing import Any, Dict, List, Optional, Union

from twinkle.data_format import Message, Trajectory
from .base import Preprocessor


class EmojiDPOProcessor(Preprocessor):
    """Preprocessor for shareAI/DPO-zh-en-emoji dataset format.

    Dataset format:
        {
            'prompt': str,
            'answer_zh': str,  # chosen response (Chinese)
            'answer_en': str,  # rejected response (English)
        }

    Output format:
        - positive: Trajectory with chosen (answer_zh)
        - negative: Trajectory with rejected (answer_en)

    Args:
        system: Optional system prompt.
        chosen_key: Key for chosen response (default: 'answer_zh').
        rejected_key: Key for rejected response (default: 'answer_en').
        prompt_key: Key for prompt (default: 'prompt').
    """

    def __init__(
        self,
        system: Optional[str] = None,
        chosen_key: str = 'answer_zh',
        rejected_key: str = 'answer_en',
        prompt_key: str = 'prompt',
    ):
        self.system = system
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.prompt_key = prompt_key

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Trajectory]:
        """Process a single row."""
        prompt = row.get(self.prompt_key, '')
        chosen = row.get(self.chosen_key, '')
        rejected = row.get(self.rejected_key, '')

        prompt_messages = []
        if self.system:
            prompt_messages.append(Message(role='system', content=self.system))
        prompt_messages.append(Message(role='user', content=prompt))

        chosen_messages = prompt_messages + [Message(role='assistant', content=chosen)]
        rejected_messages = prompt_messages + [Message(role='assistant', content=rejected)]

        return {
            'positive': Trajectory(messages=chosen_messages),
            'negative': Trajectory(messages=rejected_messages),
        }

    def __call__(self, rows: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        rows = self.map_col_to_row(rows)
        results = [self.preprocess(row) for row in rows]
        return {
            'positive': [r['positive'] for r in results],
            'negative': [r['negative'] for r in results],
        }
