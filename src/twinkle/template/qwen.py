from typing import List

from .base import Template
from ..trajectory import Trajectory


class Qwen3Template(Template):

    def encode(self, trajectories: List[Trajectory]):
        outputs = []
        for trajectory in trajectories:
            conversation = [message.to_dict_clean() for message in trajectory.messages]
            tools = [tool for tool in trajectory.tools]
            output = self.tokenizer.apply_chat_template(conversation=conversation, tools=tools)
            outputs.append(output)
        return outputs
