from dataclasses import dataclass, field
from typing import List, Dict, Any

from .message import Message, Tool


@dataclass
class Trajectory:

    messages: List[Message] = field(default_factory=list)

    tools: List[Tool] = field(default_factory=list)

    generation_config: Dict[str, Any] = field(default_factory=dict)

    experts: Dict[str, Any] = field(default_factory=dict)

    rewards: List[float] = field(default_factory=list)

    user_data: Dict[str, Any] = field(default_factory=list)

    def to_dict(self):
        return {
            'messages': [message.to_dict() for message in self.messages],
            'tools': self.tools,
            'generation_config': self.generation_config,
            'experts': self.experts,
            'rewards': self.rewards,
            'user_data': self.user_data,
        }
