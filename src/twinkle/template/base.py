from typing import List
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from twinkle.data_format import Trajectory, InputFeature


class Template:

    def __init__(self, model_id: str, **kwargs):
        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_id)

    def encode(self, trajectory: Trajectory) -> InputFeature:
        conversation = [message.to_dict_clean() for message in trajectory.messages]
        templated = self.tokenizer.apply_chat_template(conversation=conversation, tools=trajectory.tools)
        encoded = self.tokenizer(templated, return_tensors="pt")
        return InputFeature(
            input_ids=encoded['input_ids'],
            attention_mask=torch.ones_like(encoded['input_ids']),
            position_ids=torch.range(0, len(encoded['input_ids'])),
        )

    def batch_encode(self, trajectories: List[Trajectory]) -> List[InputFeature]:
        output = []
        for trajectory in trajectories:
            output.append(self.encode(trajectory))
        return output

    def check(self, trajectory: Trajectory):
        encoded = None
        try:
            encoded = self.encode(trajectory)
            if not encoded:
                return None
            else:
                return trajectory
        except Exception as e: # noqa
            return None
        finally:
            if encoded:
                del encoded
