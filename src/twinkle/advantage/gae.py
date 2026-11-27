# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, List, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function
from twinkle.data_format import turn_token_spans
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

    @remote_function(execute='first', collect='first', lazy_collect=False, stash=(0,))
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
        # old_logps is fixed to the sampler's lane (no scalar-family pre-update policy forward);
        # old_values is the pre-update critic snapshot the value loss replays against.
        old_logps = rollout_logps
        ref_logps = self._lane('ref_out', ref_out, 'logps', trajs)
        old_values = self._lane('value_out', value_out, 'values', trajs)
        kl_rows = self._kl(old_logps, ref_logps, kind=self.kl_kind, reduce='none')
        # The env-family rewards (the rollout's own ``rewards.env`` / ``rewards.check``, read off
        # the trajectories -- the driver's score entry never carries them) fold into the same
        # per-row outcome scalar at env_reward_weight; without them the row is exactly the ORM
        # fold it always was.
        env_keys, env_rows = self._env_family_rows(trajs)
        fold_weights = self.reward_weights
        if env_keys:
            fold_weights = list(fold_weights or [1.0] * len(self._orm_row(scores[0]))) + \
                [self.env_reward_weight] * len(env_keys)
        rm_scores = [weighted_score(list(self._orm_row(score)) + list(env), fold_weights)
                     for score, env in zip(scores, env_rows)]
        # A row whose every reward lane failed (ORM and env-family both, or no lane at all) folds
        # to nan: it carries no outcome signal, so it must
        # not bootstrap a value target either. Zero it before shaping and drop its advantage/return
        # afterwards -- a nan left in the reward would spread through the GAE recursion to every
        # earlier token of the row, and a fake 0 would train it as a genuine zero reward.
        orm_unscored = [value != value for value in rm_scores]
        rm_scores = [0.0 if value != value else value for value in rm_scores]
        # PRM reward-level fusion (process reward as intermediate MDP reward): the outcome reward is
        # scaled by (1 - mix) onto the last token and each turn's process reward by mix onto that
        # turn's LAST token, then the continuous GAE recursion runs over the whole frame (no turn
        # reset). The critic sees the process reward through the returns target, not as an input.
        # mix=0 leaves the pure ORM-terminal form byte-identical to a non-PRM run.
        mix = self.prm_orm_mix if self.use_prm else 0.0
        prm_turns = ([self._prm_turn_scores(traj, score) for traj, score in zip(trajs, scores)]
                     if self.use_prm else [None] * len(trajs))
        # A row carries signal if its ORM scored, or (under PRM fusion) any turn's process reward
        # scored; a row with neither folds to zero on both targets (the unscored-row contract).
        unscored = [orm_unscored[i] and not (prm_turns[i] is not None and any(v == v for v in prm_turns[i]))
                    for i in range(len(trajs))]
        if all(unscored):
            raise ValueError('GAEAdvantage got no valid reward in the batch (every ORM and PRM score row is '
                             'None): the reward would degenerate into pure KL + value bootstrap. Wire a reward '
                             'scorer, or pick a scalar-family estimator for a teacher-only run.')
        token_rewards = self._shape([value * (1.0 - mix) for value in rm_scores], kl_rows,
                                    beta=self.kl_coef, profile='token')
        if self.use_prm:
            for index, (traj, per_turn) in enumerate(zip(trajs, prm_turns)):
                if unscored[index] or per_turn is None:
                    continue
                for (_start, end), value in zip(turn_token_spans(traj), per_turn):
                    if end > 0 and value == value:
                        token_rewards[index][end - 1] += value * mix
        advantages, returns = self.token_advantages(token_rewards, old_values)
        advantages = [[0.0] * len(row) if unscored[index] else row
                      for index, row in enumerate(advantages)]
        returns = [[0.0] * len(row) if unscored[index] else row
                   for index, row in enumerate(returns)]
        new_trajs = self._write_back(trajs, {'advantages': advantages, 'returns': returns,
                                             'old_logps': old_logps, 'old_values': old_values})
        # The env-family lanes report beside the ORM half so each source stays observable.
        env_summary = {}
        for key_index, key in enumerate(env_keys):
            values = [row[key_index] for row in env_rows]
            valid = [float(value) for value in values if value is not None and value == value]
            env_summary[f'mean_{key}_reward'] = sum(valid) / len(valid) if valid else 0.0
            env_summary[f'unscored_{key}_rows'] = sum(
                1 for value in values if value is None or value != value)
        return new_trajs, {'estimator': type(self).__name__, 'num_rows': len(trajs),
                           'unscored_rows': sum(unscored),
                           # Per-row form of the same count: the caller re-indexes the lane when a
                           # restore drops rows, and a scalar would keep counting the dropped ones.
                           'unscored_flags': list(unscored),
                           'mean_advantage': _mean(advantages), 'mean_return': _mean(returns),
                           'absmean_advantage': _absmean(advantages),
                           'zero_advantage_rows': _zero_row_fraction(advantages),
                           'reward_totals': [float(sum(row)) for row in token_rewards],
                           **env_summary}

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def token_advantages(
        self,
        rewards: Sequence[Any],
        values: Sequence[Sequence[float]],
        *,
        gamma: Optional[float] = None,
        gae_lambda: Optional[float] = None,
        gae_lambda_alpha: Optional[float] = None,
        normalize: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """``(advantages, returns)``, one row per sample, each as long as its ``values`` row.

        ``values[i]`` is sample i's per-token value estimates; ``rewards[i]`` is its per-token
        rewards (the same length) or one scalar terminal reward, placed on its last token.
        ``gamma`` / ``gae_lambda`` / ``gae_lambda_alpha`` / ``normalize`` override the constructor's
        for this call; ``gae_lambda_alpha`` (positive) replaces the fixed lambda per row with
        ``1 - 1 / (alpha * row_length)`` clamped to [0, 1] (SAO's length-adaptive lambda: long
        responses lean toward lambda=1 for low bias, short ones toward 0 for low variance);
        ``normalize`` whitens the advantages over every response token of the batch.
        """
        import torch

        gamma = self.gamma if gamma is None else _check_unit('gamma', float(gamma))
        gae_lambda = self.gae_lambda if gae_lambda is None else _check_unit('gae_lambda', float(gae_lambda))
        alpha = self.gae_lambda_alpha if gae_lambda_alpha is None else float(gae_lambda_alpha)
        if alpha is not None and alpha <= 0:
            raise ValueError(f'gae_lambda_alpha must be positive, got {alpha}.')
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
            lambda_value = gae_lambda
            if alpha is not None and length > 0:
                lambda_value = min(1.0, max(0.0, 1.0 - 1.0 / (alpha * length)))
            last_gae = rewards_t.new_zeros(())
            for pos in range(length - 1, -1, -1):
                next_value = values_t[batch_idx, pos + 1] if pos + 1 < length else rewards_t.new_zeros(())
                delta = rewards_t[batch_idx, pos] + gamma * next_value - values_t[batch_idx, pos]
                last_gae = delta + gamma * lambda_value * last_gae
                advantages[batch_idx, pos] = last_gae

        returns = advantages + values_t
        if should_normalize:
            valid = advantages[masks]
            if valid.numel() > 1:
                advantages = torch.where(masks, (advantages - valid.mean()) / (valid.std(unbiased=False) + 1e-8),
                                         advantages)
        return ([advantages[i, :length].tolist() for i, length in enumerate(lengths)],
                [returns[i, :length].tolist() for i, length in enumerate(lengths)])
