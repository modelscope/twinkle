# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from twinkle.data_format import Trajectory
from twinkle.data_format.message import Tool as ToolInfo


@dataclass
class StepResult:
    """Result returned by :meth:`Gym.step`.

    Attributes:
        observation: Environment observation after the action, typically a
            string describing the tool execution result.
        reward: Scalar reward for this step (0.0 if unavailable until episode end).
        done: Whether the episode is terminated after this step.
        info: Arbitrary metadata for debugging / logging.
    """
    observation: str = ''
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class Env(ABC):
    """Base class for RL execution environments.

    Two usage modes:

    1. **Interactive mode** (multi-turn rollout) — the environment participates
       in the rollout by executing actions step-by-step::

           env.reset(trajectory)
           result = env.step(tool_name, arguments)
           # ... repeat until result.done

    2. **Batch evaluation mode** — evaluate a batch of completed trajectories
       and return rewards::

           rewards = env.evaluate(trajectories)

    To bridge with the existing :class:`ToolManager` / :class:`MultiTurnRollout`,
    wrap an ``Env`` instance with :class:`EnvTool` so it can be registered as a
    regular tool.
    """

    def reset(self, trajectory: Optional[Trajectory] = None) -> StepResult:
        """Reset environment for a new episode.

        Args:
            trajectory: Optional initial trajectory (user prompt / context).

        Returns:
            Initial observation after reset.
        """
        return StepResult()

    @abstractmethod
    def step(self, tool_name: str, arguments: Dict[str, Any]) -> StepResult:
        """Execute a single action in the environment.

        Args:
            tool_name: Name of the tool / action to invoke.
            arguments: Tool arguments as a JSON-serializable dict.

        Returns:
            StepResult with observation, reward, and done flag.
        """
        raise NotImplementedError

    def tools(self) -> List[ToolInfo]:
        """Return tool definitions available in this environment.

        Used by :class:`EnvTool` to expose the environment's capabilities to
        the LLM via the standard OpenAI-tools schema.
        """
        return []

    def evaluate(self, trajectories: List[Trajectory], **kwargs) -> List[float]:
        """Batch-evaluate completed trajectories and return rewards.

        Default implementation returns 0.0 for all trajectories. Override for
        custom reward computation (e.g. answer F1, code execution result).
        """
        return [0.0] * len(trajectories)

    def close(self) -> None:
        """Release environment resources."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
