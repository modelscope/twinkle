# Copyright (c) ModelScope Contributors. All rights reserved.
"""One call that turns a generation backend into a :class:`Rollout`.

Callers that only want turns appended to trajectories -- a challenger inventing
tasks, an evaluation script -- should not have to know that a local sampler and
an HTTP endpoint are driven by different classes with different required
arguments. They ask for a rollout, hand over whichever backend they happen to
have, and get something with the same contract:

    List[Trajectory] -> List[Trajectory]

What the two still differ in is what ends up *inside* the trajectory, and no
factory can paper over it: the sampler path keeps ``input_ids`` / ``labels`` /
``logprobs`` and is the only one whose output can be trained on, while the API
path returns messages only. Pick the backend accordingly.
"""
from typing import Any, Dict, Optional

from twinkle.data_format.sampling import SamplingParams
from .base import Rollout

__all__ = ['build_rollout']


def build_rollout(
    backend: Any,
    *,
    template: Any = None,
    tool_manager: Any = None,
    sampling_params: Optional[SamplingParams] = None,
    max_turns: int = 6,
    trace_dir: Optional[str] = None,
    **backend_kwargs: Any,
) -> Rollout:
    """Build the multi-turn rollout that matches ``backend``.

    Args:
        backend: an :class:`twinkle_agentic.protocol.base.API` (any
            OpenAI-compatible endpoint) or a sampler exposing ``sample()``.
        template: required for a sampler, rejected for an API. The sampler path
            continues a conversation by splicing token ids, which needs the
            local chat template; the API path re-sends messages as text.
        tool_manager: optional for both. Without one the model is told there
            are no tools.
        backend_kwargs: passed straight to the chosen class -- e.g. ``harness``
            and ``max_trajectory_tokens`` for a sampler, ``concurrency`` and
            ``extra_body`` for an API. An argument meant for the other backend
            surfaces as a TypeError naming it.
    """
    from twinkle_agentic.protocol.base import API

    common: Dict[str, Any] = {
        'tool_manager': tool_manager,
        'sampling_params': sampling_params,
        'max_turns': max_turns,
        'trace_dir': trace_dir,
    }

    if isinstance(backend, API):
        if template is not None:
            raise ValueError('template is only used by the sampler path; an API '
                             'backend re-sends messages as text and never encodes '
                             'them locally.')
        from .api_multi_turn import APIMultiTurnRollout
        return APIMultiTurnRollout(api=backend, **common, **backend_kwargs)

    if not hasattr(backend, 'sample'):
        raise TypeError(f'backend must be an API client or a sampler with a sample() '
                        f'method, got {type(backend).__name__}')
    if template is None:
        raise ValueError('a sampler backend needs a template: the rollout appends each '
                         'turn as token ids and cannot re-encode the history.')
    from .multi_turn import MultiTurnRollout
    return MultiTurnRollout(sampler=backend, template=template, **common, **backend_kwargs)
