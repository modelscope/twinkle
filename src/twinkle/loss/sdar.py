# Copyright (c) ModelScope Contributors. All rights reserved.
"""SDAR (Self-Distilled Agentic RL) confidence-gated teacher distillation loss.

A free auxiliary function rather than a registered :class:`~twinkle.loss.base.Loss`: callers *add* it
to the on-policy RL policy loss (``loss = policy_loss + sdar_loss_coef * L_SDAR``) instead of using it
as the sole training objective, and it returns gating metrics alongside the scalar. Sunk from the
legacy ``swift.rl_core.advantage`` path so the dev RL stack carries no cross-stack runtime dependency.
"""
from typing import Dict, Tuple

import torch


def _sdar_agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str) -> torch.Tensor:
    """Masked loss aggregation mirroring verl ``agg_loss`` (verl/trainer/ppo/core_algos.py).

    Supported modes: ``token-mean`` (default), ``seq-mean-token-sum``, ``seq-mean-token-mean``.
    """
    if loss_agg_mode == 'token-mean':
        return (loss_mat * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    if loss_agg_mode == 'seq-mean-token-sum':
        seq_losses = (loss_mat * loss_mask).sum(dim=-1)
        return seq_losses.mean()
    if loss_agg_mode == 'seq-mean-token-mean':
        seq_losses = (loss_mat * loss_mask).sum(dim=-1) / loss_mask.sum(dim=-1).clamp(min=1.0)
        return seq_losses.mean()
    raise ValueError(f'Unknown loss_agg_mode: {loss_agg_mode}')


def compute_sdar_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gate_beta: float = 5.0,
    loss_agg_mode: str = 'token-mean',
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """SDAR (Self-Distilled Agentic RL) confidence-gated teacher distillation loss.

    Exact port of ``compute_sdar_loss`` in the reference (SDAR/verl/trainer/ppo/sdar_utils.py).
    Token-level gated distillation where the gate is derived from the teacher-vs-student
    log-prob gap, so tokens where the teacher is more confident receive a stronger
    distillation signal::

        delta_t = logP_T(y_t) - logP_S(y_t)
        g_t     = sigmoid(gate_beta * delta_t)                 # detached, no grad
        L_SDAR  = agg( g_t * (logP_T(y_t) - logP_S(y_t)) )     # student keeps grad

    The gate ``g_t`` and the teacher log-probs are detached, so gradients flow only through
    the student log-probs. This auxiliary loss is *added* to the GRPO policy loss
    (``loss = policy_loss + sdar_loss_coef * L_SDAR``); it does not modify the advantage.

    Args:
        student_log_probs: ``[B, T]`` current-policy logP_theta(y_t | x, y_<t) (retains grad).
        teacher_log_probs: ``[B, T]`` teacher logP(y_t | x, r, y_<t) on the same sampled tokens.
            The teacher sees skill-augmented / privileged input ``r``; frozen (no grad).
        response_mask: ``[B, T]`` mask for valid response tokens.
        gate_beta: sigmoid gate temperature; higher = sharper gating.
        loss_agg_mode: aggregation mode (default ``token-mean``, matching the reference).

    Returns:
        ``(loss, metrics)`` where ``loss`` is a scalar and ``metrics`` holds gating statistics.
    """
    teacher_log_probs = teacher_log_probs.detach()

    delta_t = teacher_log_probs - student_log_probs.detach()

    gate = torch.sigmoid(gate_beta * delta_t).detach()

    kl_per_token = teacher_log_probs - student_log_probs

    gated_kl = gate * kl_per_token

    loss = _sdar_agg_loss(gated_kl, response_mask, loss_agg_mode)

    with torch.no_grad():
        mask_sum = response_mask.sum().clamp(min=1)
        gate_mean = (gate * response_mask).sum() / mask_sum
        gate_active = ((gate > 0.5).float() * response_mask).sum() / mask_sum
        gap_mean = (delta_t * response_mask).sum() / mask_sum

    metrics = {
        'sdar/gate_mean': gate_mean.item(),
        'sdar/gate_active_ratio': gate_active.item(),
        'sdar/teacher_gap_mean': gap_mean.item(),
        'sdar/loss': loss.detach().item(),
    }

    return loss, metrics
