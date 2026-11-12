# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-token process rewards from the driver's per-segment PRM scores."""
from __future__ import annotations

from typing import Any, Sequence

from twinkle import remote_class, remote_function

from ..rollout import turn_token_spans
from .streaming_driver import PRM_SEGMENT_TURNS


def weighted_score(values: Sequence[Any] | None, weights: Sequence[float] | None) -> float:
    """One score row folded by ``weights`` (default all-ones); ``None`` / nan entries count as 0."""
    row = [0.0 if value is None or value != value else float(value) for value in (values or ())]
    if not row:
        return 0.0
    if weights is None:
        weights = [1.0] * len(row)
    if len(weights) != len(row):
        raise ValueError(f'{len(weights)} weights for {len(row)} scores.')
    return sum(value * float(weight) for value, weight in zip(row, weights))


@remote_class(execute='first')
class ProcessReward:
    """Broadcast each episode's PRM segment scores over the response tokens of the turns they closed.

    ``samples[i].prm_rewards[scorer][segment]`` are the PRM scores and
    ``samples[i].extra[PRM_SEGMENT_TURNS]`` each segment's end turn, both stamped by the
    :class:`StreamingDriver`; a segment's ``weights``-folded score covers the response tokens of
    its assistant turns (``turn_token_spans`` of ``trajs[i]``). Returns one per-token row per episode.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(self, trajs: Sequence[dict], samples: Sequence[Any], *,
                 weights: Sequence[float] | None = None, **kwargs) -> list[list[float]]:
        if len(trajs) != len(samples):
            raise ValueError(f'{len(trajs)} trajectories for {len(samples)} samples.')
        rows = []
        for trajectory, sample in zip(trajs, samples):
            ends = list((sample.extra or {}).get(PRM_SEGMENT_TURNS) or [])
            columns = list(zip(*(sample.prm_rewards or [])))
            if len(columns) != len(ends):
                raise RuntimeError(f'episode {sample.episode_id!r} carries {len(columns)} PRM segment scores for '
                                   f'{len(ends)} recorded segments; the driver must record one end turn per segment.')
            spans = turn_token_spans(trajectory)
            per_token: list[float] = []
            start_turn = 0
            for column, end_turn in zip(columns, ends):
                value = weighted_score(column, weights)
                for start, end in spans[start_turn:end_turn]:
                    per_token.extend([value] * (end - start))
                start_turn = end_turn
            if start_turn != len(spans):
                raise RuntimeError(f'episode {sample.episode_id!r}: its PRM segments end at turn {start_turn}, but '
                                   f'the trajectory has {len(spans)} trainable assistant turns; every turn must '
                                   'fall in a segment (the last step always closes one).')
            rows.append(per_token)
        return rows
