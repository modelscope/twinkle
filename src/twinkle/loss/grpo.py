# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Dict, Optional, List, TYPE_CHECKING, Union

from twinkle.loss.base import Loss
from twinkle.utils.torch_utils import selective_log_softmax
from twinkle.data_format import Trajectory
import numpy as np

if TYPE_CHECKING:
    import torch

class GRPOLoss(Loss):
    """
    GRPO (Group Relative Policy Optimization) Loss.

    Args:
        epsilon: Clipping epsilon for PPO objective (lower bound)
        epsilon_high: Clipping epsilon for high importance sampling ratio (upper bound)
        beta: KL penalty coefficient (0.0 = no KL penalty)
        ignore_index: Index to ignore in labels (default: -100)
    """

    def __init__(
        self,
        epsilon: float = 0.2,
        epsilon_high: Optional[float] = None,
        beta: float = 0.0,
        ignore_index: int = -100,
        **kwargs,
    ):
        self.epsilon = epsilon
        self.epsilon_high = epsilon_high if epsilon_high is not None else epsilon
        self.beta = beta
        self.ignore_index = ignore_index

    def _extract_advantages_from_trajectories(
        self, 
        trajectories: List[Trajectory], 
        device: 'torch.device'
    ) -> 'torch.Tensor':
        """Extract advantages from trajectory objects."""
        import torch
        advantages_list = []
        for traj in trajectories:
            if isinstance(traj, dict):
                adv = traj.get('advantages', None)
            else:
                adv = getattr(traj, 'advantages', None)
            assert adv is not None, "trajectories must contain 'advantages'"
            advantages_list.append(float(adv))
        return torch.tensor(advantages_list, dtype=torch.float32, device=device)

    def _compute_loss_mask(self, labels: 'torch.Tensor') -> 'torch.Tensor':
        """
        Compute loss mask from labels.
        
        Args:
            labels: [batch, seq_len] target token ids, -100 for ignored positions
            
        Returns:
            mask: [batch, seq_len] float tensor, 1.0 for valid positions, 0.0 for ignored
        """
        return (labels != self.ignore_index).float()

    def _compute_log_importance_weights(
        self,
        per_token_logps: 'torch.Tensor',
        per_token_old_logps: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Compute log importance sampling weights.

        Override this method in subclasses for different IS strategies.
        Default: token-level importance sampling.
        
        Args:
            per_token_logps: [batch, seq_len] current policy log probabilities
            per_token_old_logps: [batch, seq_len] old policy log probabilities
            loss_mask: [batch, seq_len] mask for valid tokens
            
        Returns:
            log_weights: [batch, seq_len] log importance weights
        """
        import torch
        log_ratio = per_token_logps - per_token_old_logps
        # Clamp for numerical stability
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        return log_ratio

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Compute per-token loss with PPO clipping.

        Override this method in subclasses for different loss formulations.
        
        Args:
            ratio: [batch, seq_len] importance sampling ratio
            advantages: [batch, 1] or [batch, seq_len] advantage values (already expanded)
            per_token_logps: [batch, seq_len] current policy log probabilities
            
        Returns:
            per_token_loss: [batch, seq_len] loss for each token
        """
        import torch
        clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon_high)
        loss1 = ratio * advantages
        loss2 = clipped_ratio * advantages
        return -torch.min(loss1, loss2)

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        """
        Aggregate per-token loss to scalar.

        Override this method in subclasses for different normalization.
        Default: mean over sequences, then mean over batch.
        
        Args:
            per_token_loss: [batch, seq_len] per-token loss values
            loss_mask: [batch, seq_len] mask for valid tokens
            **kwargs: Additional arguments for subclass implementations
            
        Returns:
            loss: scalar loss value
        """
        # Per-sequence mean, then batch mean (aligned with Swift/TRL GRPO).
        # Each sequence contributes equally regardless of length.
        return (
            (per_token_loss * loss_mask).sum(-1)
            / loss_mask.sum(-1).clamp(min=1.0)
        ).mean()

    @staticmethod
    def _pad_and_align_to_batch(
        data: 'Union[torch.Tensor, List[float], np.ndarray]',
        mask: 'torch.Tensor',
        device: 'torch.device',
        dtype: 'torch.dtype',
        fill_value: float = 0.0,
    ) -> 'torch.Tensor':
        """Align data to mask: scalars broadcast, sequences scatter."""
        import torch

        batch_size, seq_len = mask.shape
        
        # Normalize to tensor
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data, dtype=dtype, device=device)
        else:
            data = data.to(device=device, dtype=dtype)
        
        if data.shape == (batch_size, seq_len):
            return data
        
        result = torch.full((batch_size, seq_len), fill_value, dtype=dtype, device=device)
        
        # Case 1: Scalars [batch] or [batch, 1]
        if data.dim() == 1 or data.shape[1] == 1:
            if data.dim() == 1:
                data = data.unsqueeze(1)
            if mask.any():
                batch_idx = mask.nonzero(as_tuple=True)[0]
                result[mask] = data[batch_idx, 0]
            return result
        
        # Case 2: Sequences [batch, *]
        for i in range(batch_size):
            pos = mask[i].nonzero(as_tuple=True)[0]
            n = min(len(pos), data[i].numel())
            if n > 0:
                result[i, pos[:n]] = data[i].flatten()[:n]
        
        return result

    @staticmethod
    def _unpack_packed_logps(
        logps: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        position_ids: 'Optional[torch.Tensor]',
        num_sequences: int,
    ) -> 'tuple':
        """Unpack packed (padding_free) tensors into per-sequence batch format.

        In padding_free / packing mode, the processor concatenates all
        sequences into a single row: ``[1, total_tokens]``.  This method
        splits them back into ``[num_sequences, max_seq_len]`` so that
        per-sequence operations (advantages broadcast, loss aggregation)
        work correctly.

        Sequence boundaries are detected from ``position_ids`` (which
        resets to 0 at each boundary).  If ``position_ids`` is unavailable,
        the method falls back to detecting contiguous non-masked (prompt)
        gaps in the packed ``loss_mask``.

        Args:
            logps: ``[1, total_tokens]`` packed log-probabilities.
            loss_mask: ``[1, total_tokens]`` packed loss mask.
            position_ids: ``[1, total_tokens]`` packed position ids, or None.
            num_sequences: Expected number of sequences in the pack.

        Returns:
            ``(logps, loss_mask)`` each of shape
            ``[num_sequences, max_seq_len]``, right-padded with 0.
        """
        import torch

        total_len = logps.shape[1]
        logps_flat = logps.squeeze(0)        # [total_tokens]
        mask_flat = loss_mask.squeeze(0)      # [total_tokens]

        # ── Find sequence boundaries ─────────────────────────────────────
        if position_ids is not None:
            pos_flat = position_ids.squeeze(0)  # [total_tokens]
            # position_ids resets to 0 at each new sequence
            boundary_indices = (pos_flat == 0).nonzero(as_tuple=True)[0]
        else:
            # Fallback: use loss_mask transitions.  Each sequence has a
            # prompt region (mask=0) followed by a response region (mask=1).
            # Detect 0→1 transitions preceded by a 0→0 gap (new prompt).
            # Simpler: find where mask goes from 1→0→...→0→1 (prompt gap).
            # We mark boundaries at the start of each prompt (first 0 after 1).
            shifted = torch.cat([torch.tensor([False], device=mask_flat.device), mask_flat[:-1]])
            # Start of a new sequence: transition from mask=1 (end of prev response)
            # to mask=0 (start of next prompt), or position 0 for the first sequence.
            prompt_starts = ((~mask_flat) & shifted).nonzero(as_tuple=True)[0]
            boundary_indices = torch.cat([
                torch.tensor([0], device=mask_flat.device),
                prompt_starts,
            ])

        # Deduplicate & sort
        boundary_indices = boundary_indices.unique(sorted=True)

        # Add end sentinel
        boundaries = torch.cat([
            boundary_indices,
            torch.tensor([total_len], device=boundary_indices.device),
        ])

        # ── Split and pad ────────────────────────────────────────────────
        seq_logps = []
        seq_masks = []
        n_seqs = min(boundaries.shape[0] - 1, num_sequences)
        for i in range(n_seqs):
            start = boundaries[i].item()
            end = boundaries[i + 1].item()
            seq_logps.append(logps_flat[start:end])
            seq_masks.append(mask_flat[start:end])

        max_len = max(s.shape[0] for s in seq_logps)
        padded_logps = torch.zeros(n_seqs, max_len, dtype=logps.dtype, device=logps.device)
        padded_masks = torch.zeros(n_seqs, max_len, dtype=loss_mask.dtype, device=loss_mask.device)
        for i in range(n_seqs):
            L = seq_logps[i].shape[0]
            padded_logps[i, :L] = seq_logps[i]
            padded_masks[i, :L] = seq_masks[i]

        return padded_logps, padded_masks

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        old_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        ref_logps: Optional['torch.Tensor'] = None,
        trajectories: Optional[List[Trajectory]] = None, # TODO: remove this argument
        advantages: Optional[Union['torch.Tensor', List[float], np.ndarray]] = None,
        **kwargs,
    ) -> 'torch.Tensor':
        """
        Compute GRPO loss.

        Args:
            inputs: Dict containing 'input_ids' and 'labels' [batch, seq_len].
                In packing mode, also expects 'position_ids' [1, total_tokens].
            outputs: Dict containing either:
                - 'logps'/'log_probs': [batch, seq_len] pre-computed log probs, OR
                - 'logits': [batch, seq_len, vocab] from which logps will be computed
            old_logps: [batch, seq_len] or List[List[float]] log probs from old/sampling policy.
                      Can have ragged per-sample lengths — will be padded and aligned
                      automatically.  If None, uses current logps (on-policy, ratio=1).
            ref_logps: Optional [batch, seq_len] reference model log probs for KL penalty.
                      Same padding/alignment rules as old_logps.
            advantages: advantage values
            **kwargs: Additional arguments

        Returns:
            loss: Scalar loss value
        """
        import torch
        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        logits = outputs.get('logits')
        if logits.shape[1] != labels.shape[1]:
            # some mllm return logits with image tokens, exclude here
            logits = logits[:, -labels.shape[1]:]

        # labels = torch.roll(labels, shifts=-1, dims=1)
        loss_mask = (labels != self.ignore_index).bool()
        masked_labels = labels.clone()
        masked_labels[~loss_mask] = 0
        logps = selective_log_softmax(logits, masked_labels)

        del logits

        device = logps.device

        # ── Detect and handle packing mode ──────────────────────────────
        # In padding_free / packing mode the processor concatenates all
        # sequences into a single row [1, total_tokens].  We detect this
        # by checking: batch_size == 1 but the actual number of sequences
        # (known from trajectories or advantages) is greater than 1.
        if trajectories is not None:
            num_sequences = len(trajectories)
        elif advantages is not None:
            num_sequences = len(advantages) if isinstance(advantages, (list, tuple)) else advantages.shape[0]
        else:
            num_sequences = logps.shape[0]
        is_packed = (logps.shape[0] == 1 and num_sequences > 1)
        if is_packed:
            position_ids = inputs.get('position_ids')
            logps, loss_mask = self._unpack_packed_logps(
                logps, loss_mask, position_ids, num_sequences,
            )

        # ── Prepare old_logps ────────────────────────────────────────────
        # old_logps may be ragged (List[List[float]]) containing only
        # response-token log-probs, whereas logps covers the full padded
        # sequence.  _pad_and_align_logps scatters them into the correct
        # positions using loss_mask.
        if old_logps is None:
            old_logps = logps.detach()
        else:
            old_logps = self._pad_and_align_to_batch(
                old_logps, loss_mask, device, logps.dtype,
            )

        # ── Prepare ref_logps (same treatment) ──────────────────────────
        if ref_logps is not None:
            ref_logps = self._pad_and_align_to_batch(
                ref_logps, loss_mask, device, logps.dtype,
            )

        
        assert advantages is not None, \
            "advantages must be provided (pass as kwarg to forward_backward)"

        advantages = self._pad_and_align_to_batch(
            advantages, loss_mask, device, logps.dtype,
        )

        # ── Compute loss ────────────────────────────────────────────────
        log_importance_weights = self._compute_log_importance_weights(logps, old_logps, loss_mask)
        ratio = torch.exp(log_importance_weights)
        
        per_token_loss = self._compute_per_token_loss(ratio, advantages, logps)
        
        if self.beta > 0.0 and ref_logps is not None:
            per_token_kl = (
                torch.exp(ref_logps - logps) 
                - (ref_logps - logps) 
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        loss = self._aggregate_loss(per_token_loss, loss_mask, **kwargs)
        
        return loss


    def compute_metrics(
        self,
        per_token_logps: 'torch.Tensor',
        per_token_old_logps: 'torch.Tensor',
        advantages: 'torch.Tensor',
        labels: 'torch.Tensor',
        ref_logps: Optional['torch.Tensor'] = None,
    ) -> Dict[str, float]:
        """Compute training metrics."""
        import torch
        # Ensure labels are shifted for loss_mask
        shift_labels = labels[:, 1:] if labels.shape[1] > per_token_logps.shape[1] else labels
        loss_mask = self._compute_loss_mask(shift_labels)
        
        # Align shapes
        seq_len = min(per_token_logps.shape[1], per_token_old_logps.shape[1], loss_mask.shape[1])
        per_token_logps = per_token_logps[:, -seq_len:]
        per_token_old_logps = per_token_old_logps[:, -seq_len:]
        loss_mask = loss_mask[:, -seq_len:]
        
        token_count = loss_mask.sum().clamp(min=1.0)
        
        def masked_mean(x):
            if x.shape[-1] == 1:
                return x.mean()
            return (x * loss_mask).sum() / token_count
        
        log_ratio = torch.clamp(per_token_logps - per_token_old_logps, min=-20.0, max=20.0)
        ratio = torch.exp(log_ratio)
        
        # Ensure advantages is 2D
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        
        metrics = {}
        
        # KL divergence
        metrics['kl'] = masked_mean(-log_ratio).item()
        
        # Clipping metrics
        is_low_clipped = (ratio < 1 - self.epsilon) & (advantages < 0)
        is_high_clipped = (ratio > 1 + self.epsilon_high) & (advantages > 0)
        metrics['clip_ratio_low'] = masked_mean(is_low_clipped.float()).item()
        metrics['clip_ratio_high'] = masked_mean(is_high_clipped.float()).item()
        metrics['clip_ratio'] = masked_mean((is_low_clipped | is_high_clipped).float()).item()
        
        # Ratio statistics
        metrics['ratio_mean'] = masked_mean(ratio).item()
        
        return metrics


class GSPOLoss(GRPOLoss):
    """
    GRPO with sequence-level importance sampling.

    Instead of per-token IS weights, uses the average log ratio over the sequence.
    """

    def _compute_log_importance_weights(
        self,
        per_token_logps: 'torch.Tensor',
        per_token_old_logps: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Sequence-level importance sampling: use mean log ratio."""
        import torch
        log_ratio = per_token_logps - per_token_old_logps
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        seq_level_log_weights = (
            (log_ratio * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)
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
        tau_pos: float = 1.0,
        tau_neg: float = 1.0,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(epsilon=epsilon, beta=beta, ignore_index=ignore_index, **kwargs)
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Soft-gated loss."""
        import torch
        gate_pos = torch.sigmoid(self.tau_pos * (ratio - 1)) * (4.0 / self.tau_pos)
        gate_neg = torch.sigmoid(self.tau_neg * (ratio - 1)) * (4.0 / self.tau_neg)
        is_positive = advantages > 0
        soft_gate = torch.where(is_positive, gate_pos, gate_neg)
        return -soft_gate * advantages


class CISPOLoss(GRPOLoss):
    """
    CISPO (Clipped Importance Sampling Policy Optimization) Loss.

    Clamps the IS weight and uses policy gradient.
    """

    def _compute_per_token_loss(
        self,
        ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        per_token_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Clamped ratio * advantage * log_prob."""
        import torch
        clamped_ratios = torch.clamp(ratio, max=1 + self.epsilon).detach()
        return -clamped_ratios * advantages * per_token_logps

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        """Sum over all tokens, divide by total token count."""
        # Use provided num_items_in_batch if available, otherwise use mask sum
        num_items = kwargs.get('num_items_in_batch', loss_mask.sum())
        return (per_token_loss * loss_mask).sum() / num_items


class BNPOLoss(GRPOLoss):
    """
    BNPO (Batch-Normalized Policy Optimization) Loss.

    Normalizes by total completion tokens across batch.
    """

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        """Sum over all tokens, divide by total token count."""
        return (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


class DRGRPOLoss(GRPOLoss):
    """
    DR-GRPO (Dynamic Ratio GRPO) Loss.
    
    Normalizes by batch_size * max_completion_length for consistent gradients.
    """
    
    def __init__(
        self,
        epsilon: float = 0.2,
        beta: float = 0.0,
        max_completion_length: int = 1024,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(epsilon=epsilon, beta=beta, ignore_index=ignore_index, **kwargs)
        self.max_completion_length = max_completion_length

    def _aggregate_loss(
        self,
        per_token_loss: 'torch.Tensor',
        loss_mask: 'torch.Tensor',
        **kwargs,
    ) -> 'torch.Tensor':
        """Normalize by batch_size * max_completion_length."""
        batch_size = loss_mask.shape[0]
        return (per_token_loss * loss_mask).sum() / (batch_size * self.max_completion_length)
