# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import List, Dict, Any
from .message import Message, Tool

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class Trajectory(TypedDict, total=False):
    """The """
    messages: List[Message]
    tools: List[Tool]
    generation_config: Dict[str, Any]
    experts: Dict[str, Any]
    rewards: List[float]
    user_data: Dict[str, Any]
