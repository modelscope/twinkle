# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Optional, Literal, List
import torch

from twinkle.loss.base import Loss
from twinkle.data_format import Trajectory


class GRPOLoss(Loss):
    """
    GRPO (Group Relative Policy Optimization) Loss.

    Args:
        epsilon: Clipping epsilon for PPO objective
        epsilon_high: Clipping epsilon for high importance sampling ratio
        beta: KL penalty coefficient (0.0 = no KL penalty)
        temperature: Temperature for logits scaling
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        epsilon_high: Optional[float] = None,
        beta: float = 0.0,
        temperature: float = 1.0,
        **kwargs,
    ):
        self.epsilon = epsilon
        self.epsilon_high = epsilon_high if epsilon_high is not None else epsilon
        self.beta = beta
        self.temperature = temperature

    @staticmethod
    def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Compute log probabilities for selected tokens."""
        if logits.dtype in [torch.float32, torch.float64]:
            selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
            logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
            per_token_logps = selected_logits - logsumexp_values
        else:
            per_token_logps = []
            for row_logits, row_labels in zip(logits, index):
                row_logps = torch.nn.functional.log_softmax(row_logits, dim=-1)
                row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
                per_token_logps.append(row_per_token_logps)
            per_token_logps = torch.stack(per_token_logps)
        return per_token_logps

    def _get_per_token_logps(
        self, 
        logits: torch.Tensor, 
        input_ids: torch.Tensor, 
        logits_to_keep: int,
    ) -> torch.Tensor:
        """Extract per-token log probabilities for completion tokens."""
        logits_shifted = logits[:, -(logits_to_keep + 1):-1, :] / self.temperature
        input_ids_shifted = input_ids[:, -logits_to_keep:]
        per_token_logps = self.selective_log_softmax(logits_shifted, input_ids_shifted)
        return per_token_logps

    def _extract_advantages(self, trajectories: List[Trajectory], device: torch.device) -> torch.Tensor:
        """Extract advantages from trajectories."""
        advantages_list = []
        for trajectory in trajectories:
            if isinstance(trajectory, dict):
                adv = trajectory.get('advantages', None)
            else:
                adv = getattr(trajectory, 'advantages', None)
            assert adv is not None, "advantages must be present in trajectory"
            advantages_list.append(float(adv))
        return torch.tensor(advantages_list, dtype=torch.float32, device=device)

    def _compute_log_importance_weights(
        self,
        log_ratio: torch.Tensor,
        per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log importance sampling weights.
        
        Override this method in subclasses for different IS strategies.
        Default: token-level importance sampling.
        """
        return log_ratio

    def _compute_per_token_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-token loss with PPO clipping.
        
        Override this method in subclasses for different loss formulations.
        """
        advantages_expanded = advantages.unsqueeze(1)
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon_high)
        loss1 = ratio * advantages_expanded
        loss2 = clipped_ratio * advantages_expanded
        return -torch.min(loss1, loss2)

    def _aggregate_loss(
        self,
        per_token_loss: torch.Tensor,
        completion_mask: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Aggregate per-token loss to scalar.
        
        Override this method in subclasses for different normalization.
        Default: mean over sequences, then mean over batch.
        """
        return (
            (per_token_loss * completion_mask).sum(-1) 
            / completion_mask.sum(-1).clamp(min=1.0)
        ).mean()

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        old_logits: torch.Tensor,
        trajectories: List[Trajectory],
        ref_logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute GRPO loss.

        Args:
            inputs: Dict containing preprocessed fields from GRPOLossProcessor
            outputs: Dict containing 'logits' from current policy
            old_logits: Logits from old/sampling policy
            trajectories: List of Trajectory objects with pre-computed advantages
            ref_logits: Reference model logits (optional, for KL penalty)

        Returns:
            loss: Scalar loss value
        """
        logits = outputs['logits']
        input_ids = inputs['input_ids']

        def _unwrap_logits(value):
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return value[0]
                return torch.cat(value, dim=0)
            return value

        old_logits = _unwrap_logits(old_logits)
        if not torch.is_tensor(old_logits):
            old_logits = torch.as_tensor(old_logits, device=logits.device)
        else:
            old_logits = old_logits.to(logits.device)

        if ref_logits is not None:
            ref_logits = _unwrap_logits(ref_logits)
            if not torch.is_tensor(ref_logits):
                ref_logits = torch.as_tensor(ref_logits, device=logits.device)
            else:
                ref_logits = ref_logits.to(logits.device)

        # Get preprocessed fields
        completion_mask = inputs['completion_mask']
        logits_to_keep = inputs['logits_to_keep']
        num_items_in_batch = inputs.get('num_items_in_batch')

        # Clamp logits_to_keep
        max_keep = min(logits_to_keep, logits.shape[1] - 1, input_ids.shape[1] - 1)
        if max_keep <= 0:
            return torch.zeros((), dtype=logits.dtype, device=logits.device)
        if max_keep != logits_to_keep:
            logits_to_keep = max_keep
            completion_mask = completion_mask[:, -logits_to_keep:]

        # Compute per-token log probabilities
        per_token_logps = self._get_per_token_logps(logits, input_ids, logits_to_keep)
        old_per_token_logps = self._get_per_token_logps(old_logits, input_ids, logits_to_keep)

        ref_per_token_logps = None
        if ref_logits is not None:
            ref_per_token_logps = self._get_per_token_logps(ref_logits, input_ids, logits_to_keep)

        # Extract advantages from trajectories
        advantages = self._extract_advantages(trajectories, logits.device)

        # Compute KL penalty if beta > 0
        per_token_kl = None
        if self.beta > 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) 
                - (ref_per_token_logps - per_token_logps) 
                - 1
            )

        # Compute importance sampling ratio
        log_ratio = per_token_logps - old_per_token_logps
        log_importance_weights = self._compute_log_importance_weights(
            log_ratio, per_token_logps, completion_mask
        )
        ratio = torch.exp(log_importance_weights)

        # Compute per-token loss
        per_token_loss = self._compute_per_token_loss(ratio, advantages, per_token_logps)

        # Add KL penalty
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        # Aggregate loss
        loss = self._aggregate_loss(per_token_loss, completion_mask, num_items_in_batch)

        return loss

    def compute_metrics(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        per_token_kl: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute training metrics."""
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        def masked_mean(x):
            if x.shape[-1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count

        metrics = {}

        if per_token_kl is not None:
            metrics['kl'] = masked_mean(per_token_kl).item()

        advantages_expanded = advantages.unsqueeze(1)
        is_low_clipped = (ratio < 1 - self.epsilon) & (advantages_expanded < 0)
        is_high_clipped = (ratio > 1 + self.epsilon_high) & (advantages_expanded > 0)
        metrics['clip_ratio_low'] = masked_mean(is_low_clipped.float()).item()
        metrics['clip_ratio_high'] = masked_mean(is_high_clipped.float()).item()
        metrics['clip_ratio'] = masked_mean((is_low_clipped | is_high_clipped).float()).item()

        return metrics


class GSPOLoss(GRPOLoss):
    """
    GRPO with sequence-level importance sampling.
    
    Instead of per-token IS weights, uses the average log ratio over the sequence.
    """

    def _compute_log_importance_weights(
        self,
        log_ratio: torch.Tensor,
        per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sequence-level importance sampling: use mean log ratio."""
        seq_level_log_weights = (
            (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        ).unsqueeze(-1)
        return seq_level_log_weights


class SAPOLoss(GRPOLoss):
    """
    SAPO (Soft-gated Advantage Policy Optimization) Loss.
    
    Uses soft gating instead of hard clipping.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        beta: float = 0.0,
        temperature: float = 1.0,
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        **kwargs,
    ):
        super().__init__(epsilon=epsilon, beta=beta, temperature=temperature, **kwargs)
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def _compute_per_token_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Soft-gated loss."""
        advantages_expanded = advantages.unsqueeze(1)
        gate_pos = torch.sigmoid(self.tau_pos * (ratio - 1)) * (4.0 / self.tau_pos)
        gate_neg = torch.sigmoid(self.tau_neg * (ratio - 1)) * (4.0 / self.tau_neg)
        is_positive = advantages_expanded > 0
        soft_gate = torch.where(is_positive, gate_pos, gate_neg)
        return -soft_gate * advantages_expanded

class CISPOLoss(GRPOLoss):
    """
    CISPO (Clipped Importance Sampling Policy Optimization) Loss.
    
    Clamps the IS weight and uses policy gradient.
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        beta: float = 0.0,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(epsilon=epsilon, beta=beta, temperature=temperature, **kwargs)

    def _compute_per_token_loss(
        self,
        ratio: torch.Tensor,
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Clamped ratio * advantage * log_prob."""
        advantages_expanded = advantages.unsqueeze(1)
        clamped_ratios = torch.clamp(ratio, max=1 + self.epsilon).detach()
        return -clamped_ratios * advantages_expanded * per_token_logps

    def _aggregate_loss(
        self,
        per_token_loss: torch.Tensor,
        completion_mask: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Sum over all tokens, divide by num_items_in_batch."""
        if num_items_in_batch is None:
            num_items_in_batch = completion_mask.sum()
        return (per_token_loss * completion_mask).sum() / num_items_in_batch


class BNPOLoss(GRPOLoss):
    """
    BNPO (Batch-Normalized Policy Optimization) Loss.
    
    Normalizes by total completion tokens across batch.
    """

    def _aggregate_loss(
        self,
        per_token_loss: torch.Tensor,
        completion_mask: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        return (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)


class DRGRPOLoss(GRPOLoss):
    def __init__(
        self,
        epsilon: float = 0.2,
        beta: float = 0.0,
        temperature: float = 1.0,
        max_completion_length: int = 1024,
        **kwargs,
    ):
        super().__init__(epsilon=epsilon, beta=beta, temperature=temperature, **kwargs)
        self.max_completion_length = max_completion_length

    def _aggregate_loss(
        self,
        per_token_loss: torch.Tensor,
        completion_mask: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Normalize by batch_size * max_completion_length."""
        batch_size = completion_mask.shape[0]
        return (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
