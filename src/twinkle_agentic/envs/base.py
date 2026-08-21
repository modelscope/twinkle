# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Tool as ToolInfo


@dataclass
class StepResult:
    """Result returned by :meth:`Env.step`."""
    observation: str = ''
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class Env(ABC):
    """Base class for RL execution environments.

    All environments implement this interface. Usage::

        env = SomeEnv(...)
        result = env.reset()
        result = env.step(tool_name, arguments)

    Tool-call markup is parsed upstream by
    :meth:`twinkle.template.base.Template.parse_tool_call`. This class only
    executes already-split ``(tool_name, arguments)`` pairs.
    """

    def reset(self, trajectory: Optional[Trajectory] = None) -> StepResult:
        return StepResult()

    @abstractmethod
    def step(self, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        raise NotImplementedError

    def step_batch(
        self,
        calls: Sequence[Tuple[str, Dict[str, Any]]],
    ) -> List[StepResult]:
        """Execute a batch of already-parsed ``(tool_name, arguments)`` pairs.

        Default is a serial loop over :meth:`step`. Subclasses that talk to a
        remote sandbox should override this so MultiTurn can keep tools off
        the generate critical path.
        """
        return [self.step(name, args or {}) for name, args in calls]

    def tools(self) -> List[ToolInfo]:
        return []

    def evaluate(self, trajectories: List[Trajectory], **kwargs) -> List[float]:
        return [0.0] * len(trajectories)

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
