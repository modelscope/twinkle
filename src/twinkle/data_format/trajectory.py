from dataclasses import dataclass, field
from typing import List, Dict, Any

from .message import Message, Tool


@dataclass
class Trajectory:

    messages: List[Message] = field(default_factory=list)

    tools: List[Tool] = field(default_factory=list)

    generation_config: Dict[str, Any] = field(default_factory=dict)

    experts: Any = None

    rewards: List[float] = field(default_factory=list)

    issues: List[str] = field(default_factory=list)
