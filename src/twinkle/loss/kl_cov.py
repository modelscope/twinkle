# Copyright (c) ModelScope Contributors. All rights reserved.
"""KL-Cov loss — PPO with selective KL penalty on high-covariance tokens.

Matches verl's ``@register_policy_loss("kl_cov")``.
Adapted from https://github.com/PRIME-RL/Entropy-Mechanism-of-RL
"""
from typing import TYPE_CHECKING, Dict, Optional

from twinkle.data_format import LossOutput
from .grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class KLCovLoss(GRPOLoss):
    """PPO with selective KL penalty on high-covariance tokens (KL-Cov).

    Identifies tokens with the highest covariance between advantages and log-probs,
    then applies an additional KL penalty only to those tokens. This targets the
    tokens most likely to cause entropy collapse without penalizing the entire sequence.

    Formula for selected tokens:
        L_selected = -advantage * ratio + ppo_kl_coef * |log_prob - old_log_prob|
    For unselected tokens:
        L_normal = -advantage * ratio

    Args:
        kl_cov_ratio: Fraction of valid tokens to apply KL penalty (default 0.0002).
        ppo_kl_coef: Coefficient for the KL penalty term (default 1.0).
    """

    def __init__(self, kl_cov_ratio: float = 0.0002, ppo_kl_coef: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        if kl_cov_ratio <= 0:
            raise ValueError(f'kl_cov_ratio must be positive, got {kl_cov_ratio}')
        self.kl_cov_ratio = kl_cov_ratio
        self.ppo_kl_coef = ppo_kl_coef

    def _reduce_loss(
        self,
        logps: 'torch.Tensor',
        old_logps: 'torch.Tensor',
        ref_logps: Optional['torch.Tensor'],
        advantages: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        log_importance_weights: 'torch.Tensor',
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        import torch

        # verl's kl_cov uses the UNCLAMPED log ratio (core_algos.py:1889-1891); the base class clamps
        # log_importance_weights to ±5, so recompute from logps/old_logps to match the oracle.
        negative_approx_kl = logps - old_logps
        abs_kl = negative_approx_kl.abs()
        ratio = torch.exp(negative_approx_kl)

        # Standard PPO loss (unclipped — matches verl which uses ratio * advantage directly)
        pg_losses = -advantages * ratio
        pg_losses_kl = -advantages * ratio + self.ppo_kl_coef * abs_kl

        # Compute covariance on CPU for top-k selection (matches verl implementation)
        all_valid = loss_mask > 0
        all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
        all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
        all_valid_logp = logps[all_valid].detach().reshape(-1).cpu()

        if len(all_valid_adv) > 0:
            k_percent_nums = max(1, int(len(all_valid_adv) * self.kl_cov_ratio))
            cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
            large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

            if len(large_cov_idxs) > 0:
                # Map flat indices back to 2D positions
                flat_indices = all_valid_idx[large_cov_idxs]
                seq_len = advantages.shape[1]
                row_idx = flat_indices // seq_len
                col_idx = flat_indices % seq_len
                pg_losses[row_idx, col_idx] = pg_losses_kl[row_idx, col_idx]

        # Optional ref-model KL penalty (additive, separate from the covariance KL)
        if self.beta > 0.0 and ref_logps is not None:
            per_token_kl = (torch.exp(ref_logps - logps) - (ref_logps - logps) - 1)
            pg_losses = pg_losses + self.beta * per_token_kl

        loss = self._aggregate_loss(pg_losses, loss_mask, **kwargs)
        return LossOutput(loss=loss, num_tokens=self._loss_num_tokens(loss_mask))
