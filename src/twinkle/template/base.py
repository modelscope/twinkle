# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from collections.abc import Mapping
from twinkle.hub import HubOperation
from twinkle.data_format import Trajectory, InputFeature, Message
from .utils import tokenize_with_assistant_labels


class Template:

    PLACEHOLDER = "<<<ASSISTANT_PLACEHOLDER_7f3d2a1b>>>"

    def __init__(self, model_id: str, **kwargs):
        model_id = HubOperation.download_model(model_id)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)
        self._test_support_assistant_tokens_mask()

    def _test_support_assistant_tokens_mask(self):
        dummy_inputs = [
            Message(role='user', content='How are you?'),
            Message(role='assistant', content='Fine.'),
        ]
        outputs = self.tokenizer.apply_chat_template(conversation=dummy_inputs,
                                                     return_assistant_tokens_mask=True, return_dict=True)
        assistant_masks = outputs['assistant_masks']
        self._template_support_assistant_tokens_mask = not all(np.array(assistant_masks).flatten())

    def encode(self, trajectory: Trajectory) -> InputFeature:
        if self._template_support_assistant_tokens_mask:
            messages = [dict(message) for message in trajectory['messages']]
            tools = [dict(tool) for tool in trajectory.get('tools', [])]
            outputs = self.tokenizer.apply_chat_template(conversation=messages, tools=tools,
                                               return_assistant_tokens_mask=True, return_dict=True)
            input_ids = outputs['input_ids']
            assistant_masks = outputs['assistant_masks']
            labels = np.where(assistant_masks, input_ids, -100)
        else:
            input_ids, labels = tokenize_with_assistant_labels(self.tokenizer, trajectory)
        return InputFeature(
            input_ids=np.array(input_ids),
            attention_mask=np.ones_like(input_ids),
            position_ids=np.arange(0, len(input_ids)),
            labels=np.roll(labels, -1, axis=-1),
        )

    @staticmethod
    def find_subsequence(seq: List[int], subseq: List[int], start: int = 0) -> int:
        subseq_len = len(subseq)
        for i in range(start, len(seq) - subseq_len + 1):
            if seq[i:i + subseq_len] == subseq:
                return i
        return -1

    @staticmethod
    def map_col_to_row(trajectories: Dict[str, Any]):
        if not trajectories:
            return []
        rows = []
        total_count = len(trajectories[next(iter(list(trajectories.keys())))])
        for i in range(total_count):
            row = {}
            for key in trajectories:
                row[key] = trajectories[key][i]
            rows.append(row)
        return rows

    @staticmethod
    def map_row_to_col(rows: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
        if not rows:
            return {}

        columns: Dict[str, List[Any]] = {}
        keys = rows[0].keys()

        for key in keys:
            columns[key] = [row[key] for row in rows]

        return columns

    def batch_encode(self, trajectories: Dict[str, Any]) -> List[InputFeature]:
        output = []
        _transfer = False
        if isinstance(trajectories, Mapping):
            _transfer = True
            trajectories = self.map_col_to_row(trajectories)
        for trajectory in trajectories:
            output.append(self.encode(trajectory))
        if _transfer:
            output = self.map_row_to_col(output)
        return output

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        encoded = None
        try:
            encoded = self.encode(trajectory)
            if not encoded:
                return None
            else:
                return trajectory
        except Exception as e:  # noqa
            return None
        finally:
            if encoded:
                del encoded

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        output = []
        for trajectory in trajectories:
            output.append(self.check(trajectory))
        return output

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.tokenizer.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids: List[List[int]], **kwargs) -> List[str]:
        return [self.tokenizer.decode(_ids, **kwargs) for _ids in token_ids]