# Copyright (c) ModelScope Contributors. All rights reserved.
"""Reward shaping: subtract a baseline and a KL penalty, per row or per response token."""
from typing import Any, List, Literal, Optional, Sequence, Union

from twinkle import remote_class, remote_function


@remote_class(execute='first')
class RewardShape:
    """``r' = r - baseline - beta * KL``, per row (``'outcome'``) or per response token (``'token'``).

    ``rewards`` holds one scalar per row and ``baseline`` is a scalar or one value per row. ``kl`` is
    a lane from :class:`twinkle.signals.KLEstimator`: one sequence value per row for ``'outcome'``,
    per-token rows for ``'token'``. ``'outcome'`` returns one shaped scalar per row; ``'token'``
    returns ``-beta * kl_t`` on every response token with ``r - baseline`` added on the row's last
    token -- the terminal-reward form a token-level estimator such as GAE consumes. An absent
    ``kl`` / ``baseline`` is an absent term.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(self,
                 rewards: Sequence[float],
                 kl: Optional[Sequence[Any]] = None,
                 baseline: Union[None, float, Sequence[float]] = None,
                 *,
                 beta: float = 0.0,
                 profile: Literal['outcome', 'token'] = 'outcome',
                 **kwargs) -> List[Any]:
        if profile not in ('outcome', 'token'):
            raise ValueError(f"reward_shape profile {profile!r} not in ('outcome', 'token').")
        shaped = [float(reward) for reward in rewards]
        if baseline is not None:
            baselines = ([float(baseline)] * len(shaped) if isinstance(baseline, (int, float))
                         else [float(value) for value in baseline])
            if len(baselines) != len(shaped):
                raise ValueError(f'{len(baselines)} baselines for {len(shaped)} rewards.')
            shaped = [reward - value for reward, value in zip(shaped, baselines)]
        if kl is not None and len(kl) != len(shaped):
            raise ValueError(f'{len(kl)} kl rows for {len(shaped)} rewards.')
        if profile == 'outcome':
            if kl is None:
                return shaped
            return [reward - beta * float(value) for reward, value in zip(shaped, kl)]
        if kl is None:
            raise ValueError("profile='token' needs the per-token kl lane: its rows give each row's response tokens.")
        rows = []
        for index, (reward, kl_row) in enumerate(zip(shaped, kl)):
            row = [-beta * float(value) for value in kl_row]
            if not row:
                raise ValueError(f'row {index} has no response token to carry its reward.')
            row[-1] += reward
            rows.append(row)
        return rows
