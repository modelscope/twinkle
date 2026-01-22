# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import List, Any, Tuple
from .message import Message, Tool

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class Trajectory(TypedDict, total=False):
    """The input messages"""
    messages: List[Message]
    extend_message: List[Tuple[str, List[Message]]]
    tools: List[Tool]
    generation_config: List[Tuple[str, Any]]
    experts: List[Tuple[str, Any]]
    rewards: List[float]
    user_data: List[Tuple[str, Any]]
