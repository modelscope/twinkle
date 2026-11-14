# Copyright (c) ModelScope Contributors. All rights reserved.
"""Teacher-signal per-token advantage shaping for on-policy distillation RL (OPD-RL / RLSD).

Free functions that fold a frozen teacher's per-token log-probs -- scored on the *same* tokens the
student sampled -- into the group-relative advantage. They run at batch-construction time (not inside
the loss), so a caller expands the per-sequence base advantage ``[B]`` to the per-token frame ``[B, T]``
and optionally adds or reweights it with the teacher-vs-student log-prob gap. Sunk from the legacy
``swift.rl_core.advantage`` path so the dev RL stack carries no cross-stack runtime dependency.
:class:`TeacherReweight` applies them to ragged per-sample rows.
"""
from typing import List, Literal, Optional, Sequence

import torch

from twinkle import remote_class, remote_function


def compute_teacher_logratio(
    teacher_per_token_logps: torch.Tensor,
    policy_per_token_logps: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token signed teacher log-ratio (OPD-RL) -- the k1 reverse-KL estimator.

    This is the correct OPD-RL policy-gradient signal (PG OPD): the negative single-sample
    reverse-KL estimate used as a reward, ``r_t = teacher_logp(y_t) - student_logp(y_t)``

    Args:
        teacher_per_token_logps: ``[B, T]`` teacher logp on sampled tokens.
        policy_per_token_logps: ``[B, T]`` student (old) logp on the same tokens.
        completion_mask: ``[B, T]`` response-token mask.

    Returns:
        ``[B, T]`` per-token signed log-ratio (masked outside the response).
    """
    d = teacher_per_token_logps - policy_per_token_logps
    return d.masked_fill(~completion_mask.bool(), 0.0)


def expand_advantage_to_per_token(
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    teacher_per_token_logps: Optional[torch.Tensor] = None,
    policy_per_token_logps: Optional[torch.Tensor] = None,
    teacher_kl_coef: float = 0.0,
) -> torch.Tensor:
    """Expand the per-sequence base advantage ``[B]`` to per-token ``[B, T]``.

    Broadcasting the per-sequence advantage to per-token happens *here* (at batch
    construction) rather than in the loss, so the OPD-RL teacher signal can be added
    per token: ``adv_t = base_adv + coef * (teacher_logp_t - student_logp_t)`` .
    Without a teacher this is a plain broadcast.

    Args:
        advantages: ``[B]`` per-sequence base advantage.
        completion_mask: ``[B, T]`` response-token mask (defines the token frame).
        teacher_per_token_logps: ``[B, T]`` teacher logp on sampled tokens (OPD-RL).
        policy_per_token_logps: ``[B, T]`` student (old) logp on the same tokens.
        teacher_kl_coef: Coefficient for the per-token teacher signal.

    Returns:
        ``[B, T]`` per-token advantage.
    """
    per_token_adv = advantages.unsqueeze(1).expand_as(completion_mask).clone()
    if teacher_per_token_logps is not None and teacher_kl_coef != 0.0:
        signed = compute_teacher_logratio(teacher_per_token_logps, policy_per_token_logps, completion_mask)
        per_token_adv = per_token_adv + teacher_kl_coef * signed
    return per_token_adv


def apply_rlsd_reweight(
    base_advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    teacher_per_token_logps: torch.Tensor,
    policy_per_token_logps: torch.Tensor,
    lam: float,
    clip_range: float,
    negative_only: bool = False,
) -> torch.Tensor:
    """RLSD (Self-Distilled RLVR) token-level advantage reweighting.

    Redistributes the per-sequence GRPO advantage *inside* each trajectory using the
    teacher-vs-student log-prob gap, without ever flipping the sign of the environment
    reward (the reweight is strictly positive). Mirrors ``_build_stgca_advantages`` in
    the reference implementation (RLSD/verl/workers/actor/dp_opsd_actor.py)::

        delta_t  = stop_grad(logP_T(y_t) - logP_S(y_t))
        w_t      = exp(sign(A) * delta_t)
        reweight = (1 - lam) + lam * clip(w_t, 1 - clip_range, 1 + clip_range)
        A_hat_t  = A * stop_grad(reweight)

    ``lam`` mixes between pure GRPO (``lam=0`` -> plain broadcast) and full RLSD reweighting
    (``lam=1``). The teacher is the "informed self": the same policy conditioned on the
    ground-truth answer, scoring the *same* sampled tokens (``teacher_per_token_logps``).
    ``policy_per_token_logps`` is the student side (the batch-time old logps).

    Args:
        base_advantages: ``[B]`` per-sequence GRPO advantage.
        completion_mask: ``[B, T]`` response-token mask.
        teacher_per_token_logps: ``[B, T]`` teacher logp on sampled tokens.
        policy_per_token_logps: ``[B, T]`` student (old) logp on the same tokens.
        lam: Effective mixing weight (already schedule-adjusted).
        clip_range: Evidence-weight clip epsilon ``eps_w``.
        negative_only: When True, only reweight sequences with ``A < 0`` (incorrect
            responses); ``A >= 0`` sequences keep pure GRPO advantages.

    Returns:
        ``[B, T]`` per-token reweighted advantage (masked outside the response).
    """
    mask = completion_mask
    base_bt = base_advantages.unsqueeze(1).expand_as(mask)
    delta = (teacher_per_token_logps.detach() - policy_per_token_logps.detach()) * mask
    sign_a = torch.sign(base_bt)
    weights = torch.exp(sign_a * delta) * mask
    clipped = torch.clamp(weights, min=1.0 - clip_range, max=1.0 + clip_range)
    reweight = (1.0 - lam) + lam * clipped
    if negative_only:
        seq_neg = (base_advantages < 0).float().unsqueeze(1)
        reweight = seq_neg * reweight + (1.0 - seq_neg)
    return base_bt * reweight.detach() * mask


@remote_class(execute='first')
class TeacherReweight:
    """Per-token advantages from per-sequence ones and a teacher-vs-policy logp lane.

    ``mode='rlsd'`` reweights each sequence's advantage inside its tokens (:func:`apply_rlsd_reweight`
    with ``lam`` / ``clip_range`` / ``negative_only``); ``mode='expand'`` broadcasts it and adds
    ``teacher_kl_coef`` times the per-token log-ratio (:func:`expand_advantage_to_per_token`).
    ``old_logps`` / ``teacher_logps`` are ragged per-sample rows over the same response tokens; the
    result is one per-token row per sample, each as long as its ``old_logps`` row.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(self,
                 advantages: Sequence[float],
                 old_logps: Sequence[Sequence[float]],
                 teacher_logps: Sequence[Sequence[float]],
                 *,
                 mode: Literal['rlsd', 'expand'] = 'expand',
                 lam: float = 0.0,
                 clip_range: float = 5.0,
                 negative_only: bool = False,
                 teacher_kl_coef: float = 0.0,
                 **kwargs) -> List[List[float]]:
        if mode not in ('rlsd', 'expand'):
            raise ValueError(f"teacher reweight mode {mode!r} not in ('rlsd', 'expand').")
        if not len(advantages) == len(old_logps) == len(teacher_logps):
            raise ValueError(f'{len(advantages)} advantages, {len(old_logps)} policy rows and '
                             f'{len(teacher_logps)} teacher rows must align.')
        lengths = [len(row) for row in old_logps]
        for index, (row, length) in enumerate(zip(teacher_logps, lengths)):
            if len(row) != length:
                raise RuntimeError(f'row {index}: {len(row)} teacher logps vs {length} policy logps; '
                                   'teacher and policy must score the same response tokens.')
        max_len = max(lengths, default=0)

        def _pad(rows: Sequence[Sequence[float]]) -> torch.Tensor:
            return torch.tensor([[float(v) for v in row] + [0.0] * (max_len - len(row)) for row in rows],
                                dtype=torch.float32).reshape(len(rows), max_len)

        mask = torch.tensor([[True] * length + [False] * (max_len - length) for length in lengths],
                            dtype=torch.bool).reshape(len(lengths), max_len)
        # The estimators take the per-sequence base advantage as [B] and broadcast it to [B, T].
        base = torch.tensor([float(value) for value in advantages], dtype=torch.float32)
        old, teacher = _pad(old_logps), _pad(teacher_logps)
        if mode == 'rlsd':
            out = apply_rlsd_reweight(base, mask, teacher, old, lam=lam, clip_range=clip_range,
                                      negative_only=negative_only)
        else:
            out = expand_advantage_to_per_token(base, mask, teacher, old, teacher_kl_coef=teacher_kl_coef)
        return [out[i, :length].tolist() for i, length in enumerate(lengths)]
