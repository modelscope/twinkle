from .base import Template
from ..trajectory import Trajectory


class Qwen3Template(Template):

    def encode(self, trajectory: Trajectory):
        conversation = [message.to_dict_clean() for message in trajectory.messages]
        tools = [tool for tool in trajectory.tools]
        return self.tokenizer.apply_chat_template(conversation=conversation, tools=tools)
