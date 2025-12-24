from dataclasses import dataclass, field
from typing import List, Dict, Any

from .message import Message


@dataclass
class Trajectory:

    messages: List[Message] = field(default_factory=list)

    generation_config: Dict[str, Any] = field(default_factory=dict)

    export_information: Dict[str, Any] = field(default_factory=dict)

    rewards: List[float] = field(default_factory=list)