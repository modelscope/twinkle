# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict

from twinkle.data_format import Tool as ToolInfo


class Tool(ABC):

    @abstractmethod
    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def tool_info(self) -> ToolInfo:
        raise NotImplementedError
