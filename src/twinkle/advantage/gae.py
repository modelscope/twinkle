# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, List, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function
from .base import Advantage, _absmean, _check_unit, _mean, _zero_row_fraction, weighted_score


@remote_class(execute='first')
class GAEAdvantage(Advantage):
    """Token-level generalized advantage estimation over ragged per-sample response tokens.

    The token-level member of the :class:`Advantage` family: trajectories and the critic /
    reference forward lanes in, ``advantages`` / ``returns`` / ``old_logps`` / ``old_values``
    written back onto the trajectories. The recursion itself (:meth:`token_advantages`) takes
    per-token rows, so a caller holding only references builds this class with
    ``remote_group`` and the rows are read where they live.
    """

    _token_level = True

    @remote_function(execute='first', collect='first', stash=(0,))
    def __call__(self, trajs, **kwargs):
        """Trajectories and forward lanes in, training-target columns out -- see :class:`Advantage`."""
        return self._call(trajs, **kwargs)

    def _estimate(self, trajs, rollout_logps, scores, *, ref_out, teacher_out, old_out, value_out, global_step=0):
        if teacher_out is not None or old_out is not None:
            raise ValueError("GAEAdvantage reads ref_out and value_out only; teacher_out / old_out are "
                             'the scalar-family lanes.')
        if ref_out is None:
            raise ValueError('GAEAdvantage requires ref_out: the reference forward lane for the token KL.')
        if value_out is None:
            raise ValueError("GAEAdvantage requires value_out: the pre-update critic forward (task='value').")
        if not any(value is not None and value == value
                   for score in scores for value in self._orm_row(score)):
            raise ValueError('GAEAdvantage got no valid ORM score in the batch (every score row is None): '
                             'the reward would degenerate into pure KL + value bootstrap. Wire a reward '
                             'scorer, or pick a scalar-family estimator for a teacher-only run.')
        # old_logps is fixed to the sampler's lane (no scalar-family pre-update policy forward);
        # old_values is the pre-update critic snapshot the value loss replays against.
        old_logps = rollout_logps
        ref_logps = self._lane('ref_out', ref_out, 'logps', trajs)
        old_values = self._lane('value_out', value_out, 'values', trajs)
        kl_rows = self._kl(old_logps, ref_logps, kind=self.kl_kind, reduce='none')
        rm_scores = [weighted_score(self._orm_row(score), self.reward_weights) for score in scores]
        # A row whose every ORM scorer failed folds to nan: it carries no reward signal, so it must
        # not bootstrap a value target either. Zero it before shaping and drop its advantage/return
        # afterwards -- a nan left in the reward would spread through the GAE recursion to every
        # earlier token of the row, and a fake 0 would train it as a genuine zero reward.
        unscored = [value != value for value in rm_scores]
        rm_scores = [0.0 if value != value else value for value in rm_scores]
        token_rewards = self._shape(rm_scores, kl_rows, beta=self.kl_coef, profile='token')
        advantages, returns = self.token_advantages(token_rewards, old_values)
        advantages = [[0.0] * len(row) if unscored[index] else row
                      for index, row in enumerate(advantages)]
        returns = [[0.0] * len(row) if unscored[index] else row
                   for index, row in enumerate(returns)]
        new_trajs = self._write_back(trajs, {'advantages': advantages, 'returns': returns,
                                             'old_logps': old_logps, 'old_values': old_values})
        return new_trajs, {'estimator': type(self).__name__, 'num_rows': len(trajs),
                           'unscored_rows': sum(unscored),
                           # Per-row form of the same count: the caller re-indexes the lane when a
                           # restore drops rows, and a scalar would keep counting the dropped ones.
                           'unscored_flags': list(unscored),
                           'mean_advantage': _mean(advantages), 'mean_return': _mean(returns),
                           'absmean_advantage': _absmean(advantages),
                           'zero_advantage_rows': _zero_row_fraction(advantages),
                           'reward_totals': [float(sum(row)) for row in token_rewards]}

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def token_advantages(
        self,
        rewards: Sequence[Any],
        values: Sequence[Sequence[float]],
        *,
        gamma: Optional[float] = None,
        gae_lambda: Optional[float] = None,
        normalize: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """``(advantages, returns)``, one row per sample, each as long as its ``values`` row.

        ``values[i]`` is sample i's per-token value estimates; ``rewards[i]`` is its per-token
        rewards (the same length) or one scalar terminal reward, placed on its last token.
        ``gamma`` / ``gae_lambda`` / ``normalize`` override the constructor's for this call;
        ``normalize`` whitens the advantages over every response token of the batch.
        """
        import torch

        gamma = self.gamma if gamma is None else _check_unit('gamma', float(gamma))
        gae_lambda = self.gae_lambda if gae_lambda is None else _check_unit('gae_lambda', float(gae_lambda))
        should_normalize = self.normalize if normalize is None else normalize
        if len(rewards) != len(values):
            raise ValueError(f'{len(rewards)} reward rows for {len(values)} value rows.')
        reward_rows: List[List[float]] = []
        value_rows: List[List[float]] = []
        for index, (reward, value) in enumerate(zip(rewards, values)):
            value = [float(v) for v in value]
            if isinstance(reward, (list, tuple)):
                reward = [float(r) for r in reward]
                if len(reward) != len(value):
                    raise ValueError(f'row {index}: {len(reward)} token rewards vs {len(value)} values; '
                                     'the value lane must cover exactly the response tokens.')
            else:
                if not value:
                    raise ValueError(f'row {index} has no response token to carry its terminal reward.')
                reward = [0.0] * (len(value) - 1) + [float(reward)]
            reward_rows.append(reward)
            value_rows.append(value)
        lengths = [len(row) for row in value_rows]
        max_len = max(lengths, default=0)
        rewards_t = torch.tensor([row + [0.0] * (max_len - len(row)) for row in reward_rows], dtype=torch.float32)
        values_t = torch.tensor([row + [0.0] * (max_len - len(row)) for row in value_rows], dtype=torch.float32)
        masks = torch.tensor([[True] * length + [False] * (max_len - length) for length in lengths], dtype=torch.bool)

        advantages = torch.zeros_like(rewards_t)
        for batch_idx, length in enumerate(lengths):
            last_gae = rewards_t.new_zeros(())
            for pos in range(length - 1, -1, -1):
                next_value = values_t[batch_idx, pos + 1] if pos + 1 < length else rewards_t.new_zeros(())
                delta = rewards_t[batch_idx, pos] + gamma * next_value - values_t[batch_idx, pos]
                last_gae = delta + gamma * gae_lambda * last_gae
                advantages[batch_idx, pos] = last_gae

        returns = advantages + values_t
        if should_normalize:
            valid = advantages[masks]
            if valid.numel() > 1:
                advantages = torch.where(masks, (advantages - valid.mean()) / (valid.std(unbiased=False) + 1e-8),
                                         advantages)
        return ([advantages[i, :length].tolist() for i, length in enumerate(lengths)],
                [returns[i, :length].tolist() for i, length in enumerate(lengths)])
