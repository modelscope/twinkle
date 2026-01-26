# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Optional, Literal, List
import torch

from twinkle.loss.base import Loss
from twinkle.data_format import Trajectory


class GRPOLoss(Loss):
    """
    GRPO (Group Relative Policy Optimization) Loss.

    This implementation assumes inputs are preprocessed by GRPOLossProcessor, containing:
    - input_ids: Token ids
    - labels: Labels with -100 for prompt tokens
    - completion_mask: Mask for completion tokens
    - logits_to_keep: Number of completion tokens
    - num_items_in_batch: Total completion tokens (for DAPO/CISPO)

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
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        **kwargs,
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
            num_generations: Number of generations per prompt

        Returns:
            advantages: Computed advantages, shape [batch_size]
        """
        num_generations = num_generations or self.num_generations
        if rewards.dim() > 1:
            rewards = rewards.sum(dim=-1)
        # Guard against mis-specified num_generations causing invalid reshapes.
        if num_generations <= 0 or rewards.numel() % num_generations != 0:
            num_generations = 1
        if num_generations == 1:
            return rewards
        
        grouped_rewards = rewards.view(-1, num_generations)
        group_mean = grouped_rewards.mean(dim=1, keepdim=True)
        
        if self.advantage_estimator == 'rloo':
            if num_generations > 1:
                K = num_generations
                advantages = grouped_rewards * K / (K - 1) - group_mean * K / (K - 1)
            else:
                advantages = grouped_rewards - group_mean
        else:
            advantages = grouped_rewards - group_mean
        
        advantages = self._normalize_advantages(advantages, grouped_rewards, num_generations)
        return advantages.view(-1)

    def _normalize_advantages(
        self,
        advantages: torch.Tensor,
        grouped_rewards: torch.Tensor,
        num_generations: int,
    ) -> torch.Tensor:
        if self.scale_rewards == 'none':
            return advantages
        
        if self.advantage_estimator == 'reinforce_plus_plus':
            if self.scale_rewards == 'batch':
                std = advantages.std() if advantages.numel() > 1 else torch.zeros_like(advantages.flatten()[0])
            else:
                std = advantages.std(dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(advantages)
        else:
            if self.scale_rewards == 'batch':
                std = grouped_rewards.std() if grouped_rewards.numel() > 1 else torch.zeros_like(grouped_rewards.flatten()[0])
            else:
                std = grouped_rewards.std(dim=1, keepdim=True) if num_generations > 1 else torch.zeros_like(grouped_rewards)
        
        return advantages / (std + 1e-8)

    @staticmethod
    def selective_log_softmax(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
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
        logits_shifted = logits[:, -(logits_to_keep + 1):-1, :] / self.temperature
        input_ids_shifted = input_ids[:, -logits_to_keep:]
        per_token_logps = self.selective_log_softmax(logits_shifted, input_ids_shifted)
        return per_token_logps

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
            inputs: Dict containing (preprocessed by GRPOLossProcessor):
                - input_ids: Token ids, shape [batch_size, seq_len]
                - labels: Labels with -100 for prompt, shape [batch_size, seq_len]
                - completion_mask: Mask for completion tokens, shape [batch_size, logits_to_keep]
                - logits_to_keep: Number of completion tokens (int)
                - num_items_in_batch: Total completion tokens (int, for DAPO/CISPO)
            outputs: Dict containing:
                - logits: Current policy logits, shape [batch_size, seq_len, vocab_size]
            old_logits: Old policy logits, shape [batch_size, seq_len, vocab_size]
            trajectories: List of Trajectory objects containing rewards
            ref_logits: Reference model logits (optional, for KL penalty)

        Returns:
            loss: Scalar loss value
        """
        logits = outputs['logits']
        input_ids = inputs['input_ids']

        def _unwrap_logits(value):
            # Ray multi-worker gathers may wrap tensors in lists/tuples.
            if isinstance(value, (list, tuple)):
                if len(value) == 1:
                    return value[0]
                return torch.cat(value, dim=0)
            return value
        # old_logits and ref_logits may be wrapped; unwrap them.
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
        
        # Get preprocessed fields from inputs
        completion_mask = inputs['completion_mask']
        logits_to_keep = inputs['logits_to_keep']
        num_items_in_batch = inputs.get('num_items_in_batch')

        # Clamp to avoid negative lengths when truncation exceeds available tokens.
        # Observed errors: index out of range / size mismatch during loss computation when logits_to_keep
        # exceeded seq_len - 1, causing misaligned slices for next-token prediction.
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
        
        # Extract rewards from trajectories
        rewards_list = []
        for trajectory in trajectories:
            # Trajectories can be dicts after serialization; rewards can be None or list.
            if isinstance(trajectory, dict):
                reward_value = trajectory.get('rewards')
            else:
                reward_value = trajectory.rewards
            if reward_value is None:
                reward_value = 0.0
            reward_value = sum(reward_value) if isinstance(reward_value, list) else reward_value
            rewards_list.append(reward_value)
        rewards = torch.tensor(rewards_list, dtype=torch.float32, device=logits.device)
        
        # Compute advantages
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
        
        # Compute per-token loss
        coef_1 = torch.exp(log_importance_weights)
        per_token_loss = self._compute_per_token_loss(coef_1, advantages, per_token_logps)
        
        # Add KL penalty
        if per_token_kl is not None:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # Aggregate loss
        loss = self._aggregate_loss(per_token_loss, completion_mask, num_items_in_batch)
        
        return loss

    def _compute_log_importance_weights(
        self,
        log_ratio: torch.Tensor,
        per_token_logps: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.importance_sampling_level == 'token':
            return log_ratio
        
        seq_level_log_weights = (
            (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        ).unsqueeze(-1)
        
        if self.importance_sampling_level == 'sequence':
            return seq_level_log_weights
        else:
            seq_level_log_weight = seq_level_log_weights.detach()
            return per_token_logps - per_token_logps.detach() + seq_level_log_weight

    def _compute_per_token_loss(
        self,
        coef_1: torch.Tensor,
        advantages: torch.Tensor,
        per_token_logps: torch.Tensor,
    ) -> torch.Tensor:
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
        
        else:
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
        if self.loss_type in ['grpo', 'sapo']:
            return (
                (per_token_loss * completion_mask).sum(-1) 
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        
        elif self.loss_type == 'bnpo':
            return (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        
        elif self.loss_type == 'dr_grpo':
            batch_size = completion_mask.shape[0]
            return (per_token_loss * completion_mask).sum() / (batch_size * self.max_completion_length)
        
        elif self.loss_type in ['cispo', 'dapo']:
            if num_items_in_batch is None:
                num_items_in_batch = completion_mask.sum()
            return (per_token_loss * completion_mask).sum() / num_items_in_batch
        
        else:
            raise ValueError(f'Unknown loss type: {self.loss_type}')

    def compute_metrics(
        self,
        coef_1: torch.Tensor,
        advantages: torch.Tensor,
        completion_mask: torch.Tensor,
        per_token_kl: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        completion_token_count = completion_mask.sum().clamp(min=1.0)
        
        def masked_mean(x):
            if x.shape[-1] == 1:
                return x.mean()
            return (x * completion_mask).sum() / completion_token_count
        
        metrics = {}
        
        if per_token_kl is not None:
            metrics['kl'] = masked_mean(per_token_kl).item()
        
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
