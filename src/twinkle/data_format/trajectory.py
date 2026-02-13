# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import Any, List, Tuple

from .message import Message, Tool

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class Trajectory(TypedDict, total=False):
    messages: List[Message]
    extend_message: List[Tuple[str, List[Message]]]
    tools: List[Tool]
    advantages: float
    user_data: List[Tuple[str, Any]]
