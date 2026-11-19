# Copyright (c) ModelScope Contributors. All rights reserved.
"""Optional durable-resume protocol for execution environments."""

from __future__ import annotations
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from twinkle.data_format import Trajectory

__all__ = [
    "EnvResumeCapability",
    "ResumableEnv",
    "UnsafeCheckpoint",
    "UnsupportedEnvResume",
    "env_resume_capability",
]


class EnvResumeCapability(str, Enum):
    """How an environment can survive a different Twinkle process."""

    NONE = "none"
    LIVE_ATTACH = "live_attach"
    DURABLE_CHECKPOINT = "durable_checkpoint"


class UnsupportedEnvResume(NotImplementedError):
    """The environment has no safe way to resume this episode."""


class UnsafeCheckpoint(RuntimeError):
    """The episode is not at a point where its workspace can be captured."""


@runtime_checkable
class ResumableEnv(Protocol):
    """Structural contract implemented only by environments that opt into resume."""

    resume_capability: EnvResumeCapability

    def checkpoint_episode(self, episode_id: str, state_key: str) -> dict[str, Any]:
        """Capture one quiescent episode under its stable logical identity."""

    def restore_episode(
        self,
        episode_id: str,
        state_key: str,
        state: dict[str, Any],
        trajectory: Trajectory | None = None,
    ) -> None:
        """Bind a durable snapshot to a newly allocated logical session."""

    def discard_episode_state(self, state_key: str) -> None:
        """Remove durable state after the owning group has been committed."""


def env_resume_capability(env: Any) -> EnvResumeCapability:
    """Return an environment's declared capability after structural validation."""
    declared = getattr(env, "resume_capability", EnvResumeCapability.NONE)
    try:
        capability = EnvResumeCapability(declared)
    except ValueError as exc:
        raise UnsupportedEnvResume(f"{type(env).__name__} declares invalid resume capability {declared!r}") from exc
    if capability == EnvResumeCapability.NONE:
        return capability
    missing = [
        name
        for name in ("checkpoint_episode", "restore_episode", "discard_episode_state")
        if not callable(getattr(env, name, None))
    ]
    if missing:
        raise UnsupportedEnvResume(f"{type(env).__name__} declares {capability.value} but is missing {missing}")
    return capability
