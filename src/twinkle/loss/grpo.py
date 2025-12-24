from typing import Dict, Optional, Tuple, Literal
import torch

from twinkle.loss.base import Loss


class GRPOLoss(Loss):
    """
    GRPO (Group Relative Policy Optimization) Loss.

    Reference: https://arxiv.org/abs/2402.03300

    This is a simplified implementation that assumes:
    - `inputs` contains all necessary tensors (per_token_logps, old_per_token_logps, 
      ref_per_token_logps, completion_mask, etc.)
    - `rewards` is provided to compute advantages

    Args:
        loss_type: Type of loss normalization ('grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo', 'sapo')
        epsilon: Clipping epsilon (lower bound for asymmetric, both bounds for symmetric)
        epsilon_high: Upper clipping bound (defaults to epsilon if not set)
        beta: KL penalty coefficient (0.0 means no KL penalty)
        temperature: Temperature for logits scaling
        scale_rewards: How to scale advantages ('group', 'batch', 'none')
        advantage_estimator: Advantage estimation method ('grpo', 'rloo', 'reinforce_plus_plus')
        importance_sampling_level: Level of importance sampling ('token', 'sequence', 'sequence_token')
        num_generations: Number of generations per prompt (for advantage computation)
        max_completion_length: Maximum completion length (for dr_grpo normalization)
    """

    def __init__(
        self,
        loss_type: Literal['grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo', 'sapo'] = 'grpo',
        epsilon: float = 0.2,
        epsilon_high: Optional[float] = None,
        beta: float = 0.0,
        temperature: float = 1.0,
        scale_rewards: Literal['group', 'batch', 'none'] = 'group',
        advantage_estimator: Literal['grpo', 'rloo', 'reinforce_plus_plus'] = 'grpo',
        importance_sampling_level: Literal['token', 'sequence', 'sequence_token'] = 'token',
        num_generations: int = 8,
        max_completion_length: int = 1024,
        # SAPO specific
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
    ):
        self.loss_type = loss_type
        self.epsilon_low = epsilon
        self.epsilon_high = epsilon_high if epsilon_high is not None else epsilon
        self.beta = beta
        self.temperature = temperature
        self.scale_rewards = scale_rewards
        self.advantage_estimator = advantage_estimator
        self.importance_sampling_level = importance_sampling_level
        self.num_generations = num_generations
        self.max_completion_length = max_completion_length
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def compute_advantages(
        self,
        rewards: torch.Tensor,
        num_generations: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute advantages from rewards using group-relative normalization.

        Args:
            rewards: Reward values, shape [batch_size] or [batch_size, num_reward_funcs]
            num_generations: Number of generations per prompt (overrides self.num_generations)

        Returns:
            advantages: Computed advantages, shape [batch_size]
        """
        num_generations = num_generations or self.num_generations
        
        # If rewards has multiple columns, sum them (assuming equal weights)
        if rewards.dim() > 1:
            rewards = rewards.sum(dim=-1)
        
        # Reshape rewards to [num_prompts, num_generations]
        grouped_rewards = rewards.view(-1, num_generations)
        
        # Compute group mean
        group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        
        # Compute advantages based on estimator type
        if self.advantage_estimator == 'rloo':
            # RLOO: Leave-One-Out baseline
            # A_i = r_i * K/(K-1) - mean_all * K/(K-1)
            if num_generations > 1:
                K = num_generations
                advantages = grouped_rewards * K / (K - 1) - group_mean * K / (K - 1)
            else:
                advantages = grouped_rewards - group_mean
        else:  # 'grpo' or 'reinforce_plus_plus'
            advantages = grouped_rewards - group_mean
        
        # Normalize advantages
        advantages = self._normalize_advantages(advantages, grouped_rewards, num_generations)
        
        # Flatten back to [batch_size]
        return advantages.view(-1)

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        grouped_rewards: torch.Tensor,
        num_generations: int,
    ) -> torch.Tensor:
        """Normalize advantages based on scale_rewards setting."""
        if self.scale_rewards == 'none':
            return advantages
        
        if self.advantage_estimator == 'reinforce_plus_plus':
            # REINFORCE++: Use std of advantages
            if self.scale_rewards == 'batch':
                std = advantages.std() if advantages.numel() > 1 else torch.zeros_like(advantages.flatten()[0])
            else:  # 'group'
                std = advantages.std(dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(advantages)
        else:  # 'grpo' or 'rloo'
            # Use std of original rewards
            if self.scale_rewards == 'batch':
                std = grouped_rewards.std() if grouped_rewards.numel() > 1 else torch.zeros_like(grouped_rewards.flatten()[0])
            else:  # 'group'
                std = grouped_rewards.std(dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(grouped_rewards)
        
        return advantages / (std + 1e-8)

    def __call__(
        self,
        per_token_logps: torch.Tensor,
        old_per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
        rewards: torch.Tensor,
        ref_per_token_logps: Optional[torch.Tensor] = None,
        num_items_in_batch: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute GRPO loss.

        Args:
            per_token_logps: Log probabilities from current policy, shape [batch_size, seq_len]
            old_per_token_logps: Log probabilities from old policy, shape [batch_size, seq_len]
            completion_mask: Mask for completion tokens, shape [batch_size, seq_len]
            rewards: Reward values, shape [batch_size] or [batch_size, num_reward_funcs]
            ref_per_token_logps: Log probabilities from reference model (for KL penalty), 
                                 shape [batch_size, seq_len]
            num_items_in_batch: Total completion tokens across all processes (for DAPO/CISPO)

        Returns:
            loss: Scalar loss value
            metrics: Dictionary containing metrics for logging
        """
        # Compute advantages from rewards
        advantages = self.compute_advantages(rewards)
        
        # Compute KL divergence penalty if beta > 0
        per_token_kl = None
        if self.beta > 0.0 and ref_per_token_logps is not None:
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) 
                - (ref_per_token_logps - per_token_logps) 
                - 1
            )
        
        # Compute importance sampling log ratio
        log_ratio = per_token_logps - old_per_token_logps
        log_importance_weights = self._compute_log_importance_weights(
            log_ratio, per_token_logps, completion_mask
        )
        
        # Compute per-token loss based on loss type
        coef_1 = torch.exp(log_importance_weights)
        per_token_loss = self._compute_per_token_loss(coef_1, advantages, per_token_logps)
        
        # Add KL penalty
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # Aggregate loss
        loss = self._aggregate_loss(per_token_loss, completion_mask, num_items_in_batch)
        
        # Compute metrics
        metrics = self._compute_metrics(
            coef_1, advantages, completion_mask, per_token_kl
        )
        
        return loss, metrics

    def _compute_log_importance_weights(
        self,
        log_ratio: torch.Tensor,
        per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log importance weights based on importance_sampling_level."""
        if self.importance_sampling_level == 'token':
            return log_ratio
        
        # Sequence-level importance sampling
        seq_level_log_weights = (
            (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        ).unsqueeze(-1)
        
        if self.importance_sampling_level == 'sequence':
            return seq_level_log_weights
        else:  # 'sequence_token' (GSPO)
            seq_level_log_weight = seq_level_log_weights.detach()
            return per_token_logps - per_token_logps.detach() + seq_level_log_weight

    def _compute_per_token_loss(
        self,
        coef_1: torch.Tensor,
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token loss based on loss type."""
        advantages_expanded = advantages.unsqueeze(1)
        
        if self.loss_type == 'cispo':
            clamped_ratios = torch.clamp(coef_1, max=self.epsilon_high).detach()
            return -clamped_ratios * advantages_expanded * per_token_logps
        
        elif self.loss_type == 'sapo':
            gate_pos = torch.sigmoid(self.tau_pos * (coef_1 - 1))
            gate_neg = torch.sigmoid(self.tau_neg * (coef_1 - 1))
            is_positive = advantages_expanded > 0
            soft_gate = torch.where(is_positive, gate_pos, gate_neg)
            return -soft_gate * advantages_expanded
        
        else:  # 'grpo', 'bnpo', 'dr_grpo', 'dapo'
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages_expanded
            per_token_loss2 = coef_2 * advantages_expanded
            return -torch.min(per_token_loss1, per_token_loss2)

    def _aggregate_loss(
        self,
        per_token_loss: torch.Tensor,
        completion_mask: torch.Tensor,
        num_items_in_batch: Optional[int] = None,
    ) -> torch.Tensor:
        """Aggregate per-token loss to scalar based on loss type."""
        if self.loss_type in ['grpo', 'sapo']:
            # Per-sequence mean, then batch mean
            return (
                (per_token_loss * completion_mask).sum(-1) 
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        
        elif self.loss_type == 'bnpo':
            # Global token mean
            return (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        
        elif self.loss_type == 'dr_grpo':
            # Normalize by batch_size * max_completion_length
            batch_size = completion_mask.shape[0]
            return (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        
        elif self.loss_type in ['cispo', 'dapo']:
            # Normalize by total completion tokens
            if num_items_in_batch is None:
                num_items_in_batch = completion_mask.sum()
            return (per_token_loss * completion_mask).sum() / num_items_in_batch
        
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

    def _compute_metrics(
        self,
        coef_1: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        per_token_kl: Optional[torch.Tensor] = None,
    ) -> Dict:
        """Compute metrics for logging."""
        completion_token_count = completion_mask.sum().clamp(min=1.0)
        
        def masked_mean(x):
            if x.shape[-1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count
        
        metrics = {}
        
        # KL divergence
        if per_token_kl is not None:
            metrics['kl'] = masked_mean(per_token_kl).item()
        
        # Clip ratios
        advantages_expanded = advantages.unsqueeze(1)
        if self.loss_type == 'cispo':
            is_clipped = (coef_1 > self.epsilon_high) & (advantages_expanded > 0)
            metrics['clip_ratio'] = masked_mean(is_clipped.float()).item()
        elif self.loss_type != 'sapo':
            is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages_expanded < 0)
            is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (advantages_expanded > 0)
            metrics['clip_ratio_low'] = masked_mean(is_low_clipped.float()).item()
            metrics['clip_ratio_high'] = masked_mean(is_high_clipped.float()).item()
            metrics['clip_ratio'] = masked_mean((is_low_clipped | is_high_clipped).float()).item()
        
        return metrics
