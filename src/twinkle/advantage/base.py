"""The advantage family: trajectories and forward lanes in, training-target columns out.

Recipes flatten the forwards themselves (``reference.forward_only(inputs=trajs,
output_frame='response', output_fields=('logps',))`` and friends) and call the estimator handle
with them: ``advantage(trajs, scores=..., ref_out=..., value_out=...)``. The driver hands the
score columns out BESIDE the rows (it never opens one), so the named reward maps arrive as the
row-aligned ``scores`` argument and meet the trajectories here, inside the worker. This base owns
the numeric half
shared by the family -- reward folding, group-relative estimation over the per-sequence math
each subclass provides, KL, PRM segment broadcast, teacher reweighting, the write-back -- and
returns ``(updated_trajs, summary)`` like every trajectory transform. Each CONCRETE subclass
carries ``@remote_class`` itself: the worker's class identity is the decorated class, so an
undecorated base keeps the math constructible and runnable wherever it is called.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from twinkle.data_format import ModelOutput, Trajectory, turn_token_spans, validate_trajectory
from twinkle.metric import RolloutCorrection
from twinkle.reward import RewardShape
from twinkle.signals import KLEstimator, SignalCombine

from .teacher_signal import TeacherReweight


def weighted_score(values: Optional[Sequence[Any]], weights: Optional[Sequence[float]]) -> float:
    """One score row folded by ``weights`` (default all-ones); ``None`` / nan entries count as 0."""
    row = [0.0 if value is None or value != value else float(value) for value in (values or ())]
    if not row:
        return 0.0
    if weights is None:
        weights = [1.0] * len(row)
    if len(weights) != len(row):
        raise ValueError(f'{len(weights)} weights for {len(row)} scores.')
    return sum(value * float(weight) for value, weight in zip(row, weights))


def _rlsd_lambda(lam: float, warmup: int, decay: int, global_step: int) -> float:
    """RLSD's mixing weight at ``global_step``: linear warmup, then linear decay to 0, else constant."""
    if warmup > 0 and global_step < warmup:
        return lam * global_step / warmup
    if decay > 0 and global_step >= warmup:
        return lam * max(1.0 - (global_step - warmup) / decay, 0.0)
    return lam


def _mean(rows: Sequence[Sequence[float]]) -> float:
    flat = [float(value) for row in rows for value in row]
    return sum(flat) / len(flat) if flat else 0.0


class Advantage:
    """The model-free advantage estimation step: one call per rollout batch, before training.

    GRPO-family estimators read the outcome rewards (the ``scores`` argument's
    ``rewards['orm']`` / ``rewards['baseline']``, optionally ``prm_segments``) plus the reference /
    teacher / pre-update-policy forward lanes and
    produce response-framed ``advantages`` (the per-sequence scalar is broadcast onto the response
    tokens by this component); ``estimator='gae'`` (PPO) reads them plus the critic's ``value_out``
    and produces token-level ``advantages`` / ``returns``. The training-target columns land on
    copies of the trajectories, returned with a light numeric summary -- the caller replaces its
    old references; nothing mutates an already-published ObjectRef. ``advantage_estimator`` is a
    registered estimator name, an ``Advantage`` subclass of your own, or a ready instance -- the
    estimator subclasses are the algorithm extension point, and they run inside this call's
    worker. Every constructor knob is scalar formula configuration, and a knob this
    configuration does not read is rejected at construction rather than silently ignored.
    """

    # The token-level (critic) half of the family; scalar estimators leave it False.
    _token_level = False

    def __init__(self, *,
                 num_generations: int = 1,
                 reward_weights: Optional[Sequence[float]] = None,
                 scale_rewards: Optional[str] = None,
                 beta: float = 0.0,
                 kl_in_reward: bool = False,
                 kl_kind: str = 'k3',
                 use_prm: bool = False,
                 prm_orm_mix: float = 0.0,
                 prm_weights: Optional[Sequence[float]] = None,
                 teacher_reweight: Optional[str] = None,
                 teacher_kl_coef: float = 0.0,
                 sdar_loss_coef: float = 0.0,
                 rlsd_lambda: float = 0.0,
                 rlsd_lambda_warmup_steps: int = 0,
                 rlsd_lambda_decay_steps: int = 0,
                 rlsd_reweight_clip_range: float = 5.0,
                 rlsd_negative_only: bool = False,
                 rollout_importance_sampling_mode: bool = False,
                 log_rollout_offpolicy_metrics: bool = False,
                 f_norm: float = 1.0,
                 gamma: float = 1.0,
                 gae_lambda: float = 0.95,
                 normalize: bool = True,
                 kl_coef: float = 0.0) -> None:
        """GRPO-family knobs: ``num_generations`` / ``scale_rewards`` / ``beta`` / ``kl_in_reward``
        / ``kl_kind`` (the k3 per-row mean folds into the reward), ``reward_weights`` (the ORM
        fold), ``use_prm`` + ``prm_orm_mix`` + ``prm_weights`` (PRM fusion), ``teacher_reweight``
        + its ``rlsd_*`` / ``teacher_kl_coef`` knobs (per-token reweighting), ``sdar_loss_coef``
        (teacher-lane requirement for the loss), ``rollout_importance_sampling_mode`` /
        ``log_rollout_offpolicy_metrics`` (the pre-update policy lane and its diagnostics),
        ``f_norm`` (gpg). GAE knobs: ``gamma`` / ``gae_lambda`` / ``normalize`` / ``kl_coef``
        (token-reward KL) / ``kl_kind`` (k1, per-token). PPO passes ``kl_kind='k1'``; GRPO keeps
        the k3 default.
        """
        if kl_kind not in ('k1', 'k2', 'k3'):
            raise ValueError(f'kl_kind {kl_kind!r} not in ("k1", "k2", "k3").')
        if teacher_reweight not in (None, 'rlsd', 'expand'):
            raise ValueError(f'teacher_reweight {teacher_reweight!r} not in (None, "rlsd", "expand").')
        if use_prm and teacher_reweight is not None:
            raise ValueError('PRM fusion and teacher reweighting are mutually exclusive per-token '
                             'sources; pick one.')
        # Every knob must have a reader for this configuration: a set-but-unread knob is a wiring
        # mistake that would otherwise silently degrade the run.
        if self._token_level:
            unused = [('num_generations', num_generations != 1), ('scale_rewards', scale_rewards is not None),
                      ('beta', beta != 0.0), ('kl_in_reward', kl_in_reward), ('f_norm', f_norm != 1.0),
                      ('use_prm', use_prm), ('prm_orm_mix', prm_orm_mix != 0.0),
                      ('prm_weights', prm_weights is not None), ('teacher_reweight', teacher_reweight is not None),
                      ('teacher_kl_coef', teacher_kl_coef != 0.0), ('sdar_loss_coef', sdar_loss_coef != 0.0),
                      ('rlsd_lambda', rlsd_lambda != 0.0),
                      ('rlsd_lambda_warmup_steps', rlsd_lambda_warmup_steps != 0),
                      ('rlsd_lambda_decay_steps', rlsd_lambda_decay_steps != 0),
                      ('rlsd_reweight_clip_range', rlsd_reweight_clip_range != 5.0),
                      ('rlsd_negative_only', rlsd_negative_only),
                      ('rollout_importance_sampling_mode', rollout_importance_sampling_mode),
                      ('log_rollout_offpolicy_metrics', log_rollout_offpolicy_metrics)]
            if kl_coef == 0.0:
                # kl_kind reaches the token-reward fold through kl_coef, and 'k1' is the value PPO
                # passes explicitly, so it is what "not set" means for this family.
                unused.append(('kl_kind', kl_kind != 'k1'))
        else:
            unused = [('gamma', gamma != 1.0), ('gae_lambda', gae_lambda != 0.95),
                      ('normalize', not normalize), ('kl_coef', kl_coef != 0.0)]
            if not kl_in_reward:
                # kl_kind is read only by the kl_in_reward fold in __call__, and 'k3' is this
                # family's own constructor default, so it is what "not set" means here.
                unused.append(('kl_kind', kl_kind != 'k3'))
            if teacher_reweight != 'expand':
                unused.append(('teacher_kl_coef', teacher_kl_coef != 0.0))
            if teacher_reweight != 'rlsd':
                unused.extend([('rlsd_lambda', rlsd_lambda != 0.0),
                               ('rlsd_lambda_warmup_steps', rlsd_lambda_warmup_steps != 0),
                               ('rlsd_lambda_decay_steps', rlsd_lambda_decay_steps != 0),
                               ('rlsd_reweight_clip_range', rlsd_reweight_clip_range != 5.0),
                               ('rlsd_negative_only', rlsd_negative_only)])
            if not use_prm:
                unused.extend([('prm_orm_mix', prm_orm_mix != 0.0), ('prm_weights', prm_weights is not None)])
            if not getattr(self, '_reads_f_norm', False):
                unused.append(('f_norm', f_norm != 1.0))
        dead = sorted(name for name, is_set in unused if is_set)
        if dead:
            raise ValueError(f'{type(self).__name__} does not read {", ".join(dead)}; drop the knobs '
                             'or pick the estimator that consumes them.')
        self.num_generations = num_generations
        self.reward_weights = reward_weights
        self.scale_rewards = scale_rewards
        self.beta = beta
        self.kl_in_reward = kl_in_reward
        self.kl_kind = kl_kind
        self.use_prm = use_prm
        self.prm_orm_mix = prm_orm_mix
        self.prm_weights = prm_weights
        self.teacher_reweight = teacher_reweight
        self.sdar_loss_coef = sdar_loss_coef
        self.rollout_importance_sampling_mode = rollout_importance_sampling_mode
        self.log_rollout_offpolicy_metrics = log_rollout_offpolicy_metrics
        self.f_norm = f_norm
        self.rlsd_lambda = rlsd_lambda
        self.rlsd_lambda_warmup_steps = rlsd_lambda_warmup_steps
        self.rlsd_lambda_decay_steps = rlsd_lambda_decay_steps
        self.rlsd_reweight_clip_range = rlsd_reweight_clip_range
        self.rlsd_negative_only = rlsd_negative_only
        self.teacher_kl_coef = teacher_kl_coef
        self.kl_coef = kl_coef
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize = normalize
        # Pure math, instantiated locally: the subclass is the remote boundary, these run in it.
        self._kl = KLEstimator()
        self._shape = RewardShape()
        self._combine = SignalCombine()
        self._teacher = TeacherReweight()
        self._correction = RolloutCorrection()

    def _call(self, trajs: List[Trajectory], *, scores: List[Dict[str, Any]],
              ref_out: Optional[ModelOutput] = None,
              teacher_out: Optional[ModelOutput] = None, old_out: Optional[ModelOutput] = None,
              value_out: Optional[ModelOutput] = None,
              global_step: int = 0) -> Tuple[List[Trajectory], Dict[str, Any]]:
        """Guards plus the family dispatch: estimate once for the whole batch.

        ``scores`` is the delivery facts' row-aligned score column: one row-shaped entry per
        trajectory (``rewards.orm`` / ``rewards.baseline`` named maps, optionally ``prm_segments``),
        as the driver hands them out beside the rows. ``ref_out`` / ``teacher_out`` / ``old_out`` /
        ``value_out`` are the response-framed
        ``forward_only`` outputs (or their lazy handles; the dispatch resolves them here in the
        worker) the estimator reads -- GRPO-family lanes are ``ref_out`` / ``teacher_out`` /
        ``old_out`` (the pre-update policy forward), the gae lane pair is ``ref_out`` +
        ``value_out`` (the pre-update critic). The result writes the training-target columns
        onto copies of the trajectories; group estimation happens over the whole batch, never
        per training micro-batch, and a batch already carrying targets is refused rather than
        re-initialized.
        """
        if not trajs:
            raise ValueError('advantage received an empty batch.')
        if len(scores) != len(trajs):
            raise ValueError(f'scores carries {len(scores)} rows for {len(trajs)} trajectories; the driver '
                             'hands out exactly one score entry per trajectory.')
        for traj in trajs:
            if 'advantages' in traj or 'returns' in traj:
                raise ValueError(f'trajectory {traj.get("episode_id")!r} already carries advantages/returns; '
                                 'one rollout batch is estimated once, before its first optimizer step.')
            if not traj.get('response_token_ids'):
                raise ValueError(f'trajectory {traj.get("episode_id")!r} has no response tokens to train on.')
        rollout_logps = []
        for traj in trajs:
            lane = traj.get('rollout_logps')
            if lane is None:
                raise ValueError(f'trajectory {traj.get("episode_id")!r} has no rollout_logps; the '
                                 'importance-ratio math needs the sampler log-probs lane.')
            rollout_logps.append(lane)
        return self._estimate(trajs, rollout_logps, scores, ref_out=ref_out, teacher_out=teacher_out,
                              old_out=old_out, value_out=value_out, global_step=global_step)

    def _estimate(self, trajs: List[Trajectory], rollout_logps: List[List[float]],
                  scores: List[Dict[str, Any]], *,
                  ref_out: Optional[ModelOutput], teacher_out: Optional[ModelOutput],
                  old_out: Optional[ModelOutput], value_out: Optional[ModelOutput],
                  global_step: int) -> Tuple[List[Trajectory], Dict[str, Any]]:
        # The scalar family has no critic; the token-level subclass overrides this whole step.
        if value_out is not None:
            raise ValueError(f'{type(self).__name__} has no critic; value_out is the token-level lane.')
        return self._estimate_group(trajs, rollout_logps, scores, ref_out, teacher_out, old_out, global_step)

    def _per_sequence(self, rewards: Any, num_generations: int = 1, scale: str = 'group',
                      **kwargs) -> Any:
        """The per-sequence scalar math -- one advantage per sample, the subclass's own formula.

        The family's shared flow calls it with the folded reward matrix and the
        estimator-specific inputs ``compute_advantages`` wires (``reward_weights``,
        ``kl_in_reward`` / ``beta`` / ``kl_values``, gpg's ``f_norm``, opo's ``response_lengths``,
        remax's ``reward_baselines``).
        """
        raise NotImplementedError(f'{type(self).__name__} does not implement its advantage math.')

    # ------------------------------------------------------------------ GRPO family

    def _estimate_group(self, trajs: List[Trajectory], rollout_logps: List[List[float]],
                        scores: List[Dict[str, Any]],
                        ref_out: Optional[ModelOutput], teacher_out: Optional[ModelOutput],
                        old_out: Optional[ModelOutput], global_step: int
                        ) -> Tuple[List[Trajectory], Dict[str, Any]]:
        from .estimate import compute_advantages

        # The pre-update policy lane replaces the sampler's logps when importance sampling or the
        # off-policy diagnostics compare the training side against it.
        needs_policy = self.rollout_importance_sampling_mode or self.log_rollout_offpolicy_metrics
        if needs_policy:
            old_logps = self._lane('old_out', old_out, 'logps', trajs)
        else:
            old_logps = rollout_logps
        ref_logps = self._lane('ref_out', ref_out, 'logps', trajs) if ref_out is not None else None
        if ref_logps is None and self.kl_in_reward:
            raise ValueError('kl_in_reward=True requires ref_out: a reference forward lane.')
        teacher_logps = self._lane('teacher_out', teacher_out, 'logps', trajs) if teacher_out is not None else None
        if teacher_logps is None and (self.teacher_reweight is not None or self.sdar_loss_coef > 0):
            raise ValueError('teacher reweighting / sdar_loss_coef>0 requires teacher_out: a teacher forward '
                             'lane (--teacher_model, a teacher_prompt view, or the adapter-disabled self-teacher).')
        kl_values = None
        if self.kl_in_reward:
            kl_values = self._kl(old_logps, ref_logps, kind=self.kl_kind)
        # None (an unparseable judge verdict) rides as nan, exactly as the scorer recorded it; a
        # no-scorer row (no ORM wired) stays [nan] -- a zero-column tensor cannot carry a group mean.
        # The nan rows then land per estimator: the grpo / rloo / passk lanes exclude them from the
        # group statistics and zero their advantage (see reduce_rewards / nanstd); the remaining
        # scalar estimators and gae fold them to 0, and gae refuses a batch with no valid score at
        # all.
        orm_rows = [self._orm_row(score) for score in scores]
        matrix_rows = [[float('nan') if value is None or value != value else float(value) for value in row]
                       or [float('nan')] for row in orm_rows]
        widths = {len(row) for row in matrix_rows}
        if len(widths) != 1:
            raise ValueError(f'the trajectories carry different ORM scorer counts {sorted(widths)}; every row '
                             'of a batch is scored by the same scorer set (a missing score is None, not a '
                             'shorter row).')
        reward_matrix = torch.tensor(matrix_rows, dtype=torch.float32)
        self._check_groups(trajs)
        baseline_rows = [list(((score.get('rewards') or {}).get('baseline') or {}).values()) for score in scores]
        baselines = ([weighted_score(row, self.reward_weights) for row in baseline_rows]
                     if any(baseline_rows) else None)
        advantages = compute_advantages(
            reward_matrix,
            self.num_generations,
            reward_weights=self.reward_weights,
            advantage_estimator=self,
            scale_rewards=self.scale_rewards or 'group',
            kl_in_reward=self.kl_in_reward,
            beta=float(self.beta or 0.0),
            kl_values=kl_values,
            response_lengths=[len(traj['response_token_ids']) for traj in trajs],
            f_norm=float(self.f_norm),
            reward_baselines=baselines)
        if self.use_prm:
            # advantage = mix * process + (1 - mix) * orm_advantage on every response token: mix=0
            # is standard GRPO, mix=1 pure PRM. The scalar ORM lane broadcasts over the token rows.
            process = [self._process_row(traj, score) for traj, score in zip(trajs, scores)]
            mix = float(self.prm_orm_mix)
            advantages = self._combine([process, advantages], weights=[mix, 1.0 - mix])
        elif self.teacher_reweight is not None:
            if self.teacher_reweight == 'rlsd':
                advantages = self._teacher(
                    advantages, old_logps, teacher_logps, mode='rlsd',
                    lam=_rlsd_lambda(self.rlsd_lambda, self.rlsd_lambda_warmup_steps,
                                     self.rlsd_lambda_decay_steps, global_step),
                    clip_range=self.rlsd_reweight_clip_range, negative_only=self.rlsd_negative_only)
            else:
                # sdar keeps the scalar advantage (the loss reads teacher_logps itself); expand
                # broadcasts it and adds the per-token teacher log-ratio.
                advantages = self._teacher(advantages, old_logps, teacher_logps, mode='expand',
                                           teacher_kl_coef=self.teacher_kl_coef)
        else:
            # The per-sequence scalar is broadcast onto each trajectory's response frame by its
            # producer, so the column is always response-framed downstream.
            advantages = [[float(value)] * len(traj['response_token_ids'])
                          for traj, value in zip(trajs, advantages)]
        columns = {'advantages': advantages, 'old_logps': old_logps}
        if ref_logps is not None:
            columns['ref_logps'] = ref_logps
        if teacher_logps is not None:
            columns['teacher_logps'] = teacher_logps
        new_trajs = self._write_back(trajs, columns)
        summary: Dict[str, Any] = {'estimator': type(self).__name__, 'num_rows': len(trajs),
                                   'mean_advantage': _mean(advantages)}
        if self.log_rollout_offpolicy_metrics:
            summary['rollout_correction'] = self._correction(old_logps, rollout_logps)
        return new_trajs, summary

    # ---------------------------------------------------------------------- helpers

    def _write_back(self, trajs: List[Trajectory], columns: Dict[str, List[Any]]) -> List[Trajectory]:
        """Write whole-batch columns onto copies of the trajectories; the caller replaces references."""
        new_trajs = []
        for index, traj in enumerate(trajs):
            merged = dict(traj)
            for field, rows in columns.items():
                merged[field] = rows[index]
            validate_trajectory(merged)
            new_trajs.append(merged)
        return new_trajs

    def _lane(self, param: str, out: ModelOutput, field: str, trajs: List[Trajectory]) -> List[Any]:
        """The ``field`` rows of one response-framed ``forward_only`` output, checked against ``trajs``."""
        if out is None:
            raise ValueError(f'{param} is required but was not provided.')
        rows = out.get(field)
        if rows is None:
            raise ValueError(f'{param} carries no {field!r} channel; request the lane with '
                             f'output_fields={(field,)!r} and output_frame="response".')
        frames = (out.get('field_frames') or {}).get(field)
        if frames is not None and frames != 'response':
            raise ValueError(f"the {field!r} lane of {param} is {frames!r}-framed; this component consumes "
                             'response-framed forwards.')
        if len(rows) != len(trajs):
            raise ValueError(f"the {field!r} lane of {param} carries {len(rows)} rows for {len(trajs)} "
                             'trajectories.')
        return rows

    @staticmethod
    def _orm_row(score: Dict[str, Any]) -> List[Optional[float]]:
        """The ORM scores of one score row, in the scorer order the driver stamped them."""
        rewards = score.get('rewards') or {}
        return list((rewards.get('orm') or {}).values())

    def _process_row(self, traj: Trajectory, score: Dict[str, Any]) -> List[float]:
        """Broadcast one trajectory's PRM segment scores over the turns they closed.

        ``prm_segments`` is ``{'end_turns': [...], 'scores': {scorer: [per-segment float | None]}}``:
        ``end_turns[k]`` is the cumulative count of assistant turns closed by segment ``k``, and
        segment ``k``'s ``prm_weights``-folded score covers the response tokens of its turns
        (``turn_token_spans`` of the trajectory). Same math as the driver's
        ``ProcessReward``, reading the score row instead of the sample record.
        """
        segments = score.get('prm_segments')
        if not segments:
            raise ValueError(f'trajectory {traj.get("episode_id")!r} carries no prm_segments; use_prm requires '
                             'the driver to record them in the score rows.')
        ends = list(segments.get('end_turns') or ())
        columns = list(zip(*(list(column) for column in (segments.get('scores') or {}).values())))
        if len(columns) != len(ends):
            raise RuntimeError(f'episode {traj.get("episode_id")!r} carries {len(columns)} PRM segment scores '
                               f'for {len(ends)} recorded segments; the driver must record one end turn per '
                               'segment.')
        spans = turn_token_spans(traj)
        per_token: List[float] = []
        start_turn = 0
        for column, end_turn in zip(columns, ends):
            value = weighted_score(column, self.prm_weights)
            for start, end in spans[start_turn:end_turn]:
                per_token.extend([value] * (end - start))
            start_turn = end_turn
        if start_turn != len(spans):
            raise RuntimeError(f'episode {traj.get("episode_id")!r}: its PRM segments end at turn {start_turn}, '
                               f'but the trajectory has {len(spans)} trainable assistant turns; every turn must '
                               'fall in a segment (the last step always closes one).')
        return per_token

    def _check_groups(self, trajs: List[Trajectory]) -> None:
        """Every prompt group arrives as one consecutive block of exactly ``num_generations`` rows."""
        if self.num_generations <= 1:
            return
        blocks: List[Tuple[Any, List[Trajectory]]] = []
        for traj in trajs:
            key = traj.get('group_key')
            if key is None:
                raise ValueError(f'trajectory {traj.get("episode_id")!r} has no group_key; grouped estimation '
                                 'needs it.')
            if not blocks or blocks[-1][0] != key:
                blocks.append((key, [traj]))
            else:
                blocks[-1][1].append(traj)
        keys = [key for key, _ in blocks]
        if len(set(keys)) != len(keys):
            raise ValueError('group_key blocks are not contiguous: every prompt group must arrive as one '
                             'consecutive block of rows.')
        for key, rows in blocks:
            if len(rows) != self.num_generations:
                raise ValueError(f'group {key!r} carries {len(rows)} rows; num_generations='
                                 f'{self.num_generations} requires complete groups.')
