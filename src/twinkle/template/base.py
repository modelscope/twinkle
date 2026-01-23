# Copyright (c) ModelScope Contributors. All rights reserved.
from copy import deepcopy, copy
from typing import List, Optional, Dict, Any, Literal, Union
from transformers import AutoTokenizer, PreTrainedTokenizer
import numpy as np
from collections.abc import Mapping
from twinkle.hub import HubOperation
from twinkle.data_format import Trajectory, InputFeature, Message
from .utils import tokenize_with_assistant_labels


class Template:

    PLACEHOLDER = "<<<ASSISTANT_PLACEHOLDER_7f3d2a1b>>>"

    def __init__(self,
                 model_id: str,
                 use_chat_template: bool = True,
                 max_length: Optional[int] = 8192,
                 truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise',
                 default_system: Optional[str] = None,
                 **kwargs):
        model_id = HubOperation.download_model(model_id)
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)
        self.use_chat_template = use_chat_template
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.default_system = default_system
        self._test_support_assistant_tokens_mask()
        self.pre_pipeline = [
            self._add_default_system, # Add a default system field
        ]
        self.post_pipeline = [
            self._check_max_length, # Check and split input_features
            self._add_attention_fields, # Add useful fields
            self._roll_labels, # roll labels
        ]

    def _test_support_assistant_tokens_mask(self):
        dummy_inputs = [
            Message(role='user', content='How are you?'),
            Message(role='assistant', content='Fine.'),
        ]
        outputs = self.tokenizer.apply_chat_template(conversation=dummy_inputs,
                                                     return_assistant_tokens_mask=True, return_dict=True)
        assistant_masks = outputs['assistant_masks']
        self._template_support_assistant_tokens_mask = (0 < np.array(assistant_masks).sum() < len(assistant_masks))

    def _invoke_pre_pipeline(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        current = trajectories
        for pipeline in self.pre_pipeline:
            next_batch = []
            for trajectory in current:
                next_batch.extend(pipeline(trajectory))
            current = next_batch
        return current

    def _invoke_post_pipeline(self, input_features: List[InputFeature]) -> List[InputFeature]:
        current = input_features
        for pipeline in self.post_pipeline:
            next_batch = []
            for input_feature in current:
                next_batch.extend(pipeline(input_feature))
            current = next_batch
        return current

    def _add_default_system(self, trajectory: Trajectory) -> List[Trajectory]:
        if self.use_chat_template and self.default_system:
            if trajectory['messages'][0]['role'] == 'user':
                trajectory['messages'].insert(0, Message(role='system', content=self.default_system))
            for (_, messages) in trajectory['extend_message']:
                if trajectory['messages'][0]['role'] == 'user':
                    trajectory['messages'].insert(0, Message(role='system', content=self.default_system))
        return [trajectory]

    def _check_max_length(self, input_feature: InputFeature) -> List[InputFeature]:
        if self.max_length and len(input_feature['input_ids']) > self.max_length:
            if self.truncation_strategy == 'raise':
                raise ValueError(f'An input message(length: {len(input_feature["input_ids"])} '
                                 f'exceeds the maximum length({self.max_length})')
            elif self.truncation_strategy == 'left':
                return [InputFeature(**{key: value[-self.max_length:] for key, value in input_feature.items()})]
            elif self.truncation_strategy == 'right':
                return [InputFeature(**{key: value[:self.max_length] for key, value in input_feature.items()})]
            else: # split
                result = []
                total_length = len(input_feature['input_ids'])
                for start in range(0, total_length, self.max_length):
                    end = min(start + self.max_length, total_length)
                    result.append(InputFeature(**{key: value[start:end] for key, value in input_feature.items()}))
                return result
        else:
            return [input_feature]

    def _add_attention_fields(self, input_feature: InputFeature) -> List[InputFeature]:
        input_ids = input_feature['input_ids']
        input_feature['attention_mask'] = np.ones_like(input_ids)
        input_feature['position_ids'] = np.arange(len(input_ids))
        input_feature['length'] = len(input_ids)
        return [input_feature]

    def _roll_labels(self, input_feature: InputFeature) -> List[InputFeature]:
        input_feature['labels'] = np.roll(input_feature['labels'], -1, axis=-1)
        return [input_feature]

    def encode(self, trajectory: Trajectory) -> InputFeature:
        if self.use_chat_template:
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
        else:
            assert len(trajectory['messages']) == 1 and trajectory['messages'][0]['role'] == 'user'
            text = trajectory['messages'][0]['content']
            input_ids = self.tokenizer.encode(text)
            labels = deepcopy(input_ids)
        return InputFeature(
            input_ids=np.array(input_ids),
            labels=labels,
        )

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

    def batch_encode(self, trajectories: Union[Dict[str, Any], List[Trajectory]]) -> List[InputFeature]:
        output = []
        _transfer = False
        if isinstance(trajectories, Mapping):
            _transfer = True
            trajectories = self.map_col_to_row(trajectories)
        trajectories = self._invoke_pre_pipeline(trajectories)
        for trajectory in trajectories:
            output.append(self.encode(trajectory))
        output = self._invoke_post_pipeline(output)
        if _transfer:
            output = self.map_row_to_col(output)
        return output

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        encoded = None
        try:
            encoded = self.batch_encode([trajectory])
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
