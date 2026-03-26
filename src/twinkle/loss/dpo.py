# Copyright (c) ModelScope Contributors. All rights reserved.
"""
DPO (Direct Preference Optimization) Loss Implementation.

Reference:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (https://arxiv.org/abs/2305.18290)
"""
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np

from twinkle.data_format import LossOutput
from twinkle.kernel import selective_log_softmax
from twinkle.loss.base import Loss

if TYPE_CHECKING:
    import torch


class DPOLoss(Loss):
    """Direct Preference Optimization (DPO) Loss.

    DPO directly optimizes the policy using preference data without explicit reward modeling.
    The loss function is derived from the Bradley-Terry preference model:

        L_DPO = -log(σ(β * (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x))))

    where:
        - y_w is the preferred (chosen) response
        - y_l is the dispreferred (rejected) response
        - β is the temperature parameter controlling deviation from reference
        - π is the current policy
        - π_ref is the reference policy (frozen)

    Args:
        beta: Temperature parameter controlling how much to deviate from ref policy (default: 0.1).
        label_smoothing: Label smoothing parameter for soft labels (default: 0.0).
        ignore_index: Index to ignore in labels (default: -100).
        loss_type: Type of DPO loss variant ('sigmoid', 'hinge', 'ipo', 'kto_pair') (default: 'sigmoid').
        reference_free: Whether to use reference-free DPO (default: False).
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        loss_type: str = 'sigmoid',
        reference_free: bool = False,
        **kwargs,
    ):
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.loss_type = loss_type
        self.reference_free = reference_free

    def _compute_logps_from_logits(
        self,
        logits: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute per-token log probabilities from logits.

        Args:
            logits: [batch, seq_len, vocab_size] model logits
            labels: [batch, seq_len] target token ids

        Returns:
            logps: [batch, seq_len] per-token log probabilities
        """
        loss_mask = (labels != self.ignore_index).bool()
        masked_labels = labels.clone()
        masked_labels[~loss_mask] = 0
        logps = selective_log_softmax(logits, masked_labels)
        return logps

    def _compute_sequence_logps(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute sequence-level log probabilities by summing valid token logps.

        Args:
            per_token_logps: [batch, seq_len] per-token log probabilities
            labels: [batch, seq_len] labels for computing mask

        Returns:
            seq_logps: [batch] sequence-level log probabilities
        """
        loss_mask = (labels != self.ignore_index).float()
        return (per_token_logps * loss_mask).sum(dim=-1)

    def _pad_and_align_logps(
        self,
        logps: Union['torch.Tensor', List[List[float]]],
        target_shape: tuple,
        loss_mask: 'torch.Tensor',
        device: 'torch.device',
        dtype: 'torch.dtype',
    ) -> 'torch.Tensor':
        """Pad and align log probabilities to target shape.

        Args:
            logps: Input log probabilities (tensor or ragged list)
            target_shape: Target (batch, seq_len) shape
            loss_mask: Boolean mask for valid positions
            device: Target device
            dtype: Target dtype

        Returns:
            Aligned tensor of shape target_shape
        """
        import torch

        if torch.is_tensor(logps):
            if logps.shape == target_shape:
                return logps.to(device=device, dtype=dtype)
            elif logps.dim() == 1:
                logps = logps.unsqueeze(0)
            if logps.shape == target_shape:
                return logps.to(device=device, dtype=dtype)

        # Handle ragged list input
        if isinstance(logps, (list, tuple)):
            batch_size, seq_len = target_shape
            padded = torch.zeros(target_shape, device=device, dtype=dtype)
            for i, row in enumerate(logps):
                if row is None:
                    continue
                row_t = torch.as_tensor(row, device=device, dtype=dtype)
                valid_positions = loss_mask[i].nonzero(as_tuple=True)[0]
                length = min(len(row_t), len(valid_positions))
                if length > 0:
                    padded[i, valid_positions[:length]] = row_t[:length]
            return padded

        return logps.to(device=device, dtype=dtype)

    def _compute_dpo_loss(
        self,
        policy_chosen_logps: 'torch.Tensor',
        policy_rejected_logps: 'torch.Tensor',
        reference_chosen_logps: 'torch.Tensor',
        reference_rejected_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute the DPO loss.

        Args:
            policy_chosen_logps: [batch/2] log probs of chosen under current policy
            policy_rejected_logps: [batch/2] log probs of rejected under current policy
            reference_chosen_logps: [batch/2] log probs of chosen under reference policy
            reference_rejected_logps: [batch/2] log probs of rejected under reference policy

        Returns:
            loss: Scalar DPO loss
        """
        import torch
        import torch.nn.functional as F

        # Compute log ratios
        if self.reference_free:
            # Reference-free: only use policy log probs
            chosen_logratios = policy_chosen_logps
            rejected_logratios = policy_rejected_logps
        else:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # Compute preference margin
        logits = self.beta * (chosen_logratios - rejected_logratios)

        if self.loss_type == 'sigmoid':
            # Standard DPO loss: -log(sigmoid(beta * margin))
            losses = -F.logsigmoid(logits)
        elif self.loss_type == 'hinge':
            # Hinge loss variant
            losses = torch.relu(1 - logits)
        elif self.loss_type == 'ipo':
            # IPO (Identity Preference Optimization) loss
            # Reference: "A General Theoretical Paradigm to Understand Learning from Human Feedback"
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == 'kto_pair':
            # KTO pair loss (simplified version)
            chosen_logratios_scaled = self.beta * chosen_logratios
            rejected_logratios_scaled = self.beta * rejected_logratios
            chosen_losses = 1 - F.sigmoid(chosen_logratios_scaled)
            rejected_losses = F.sigmoid(rejected_logratios_scaled)
            losses = chosen_losses + rejected_losses
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            # Soft labels: (1 - eps) * loss_chosen + eps * loss_rejected
            smooth_losses = -F.logsigmoid(-logits)  # Loss for flipped preference
            losses = (1 - self.label_smoothing) * losses + self.label_smoothing * smooth_losses

        return losses.mean()

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        ref_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        ref_chosen_logps: Optional['torch.Tensor'] = None,
        ref_rejected_logps: Optional['torch.Tensor'] = None,
        **kwargs,
    ) -> LossOutput:
        """Compute DPO loss.

        The inputs should contain concatenated chosen and rejected examples:
        - First half of batch: chosen responses
        - Second half of batch: rejected responses

        Args:
            inputs: Dict containing 'input_ids' and 'labels' [batch, seq_len].
                   Batch should be organized as [chosen_1, ..., chosen_n, rejected_1, ..., rejected_n]
            outputs: Dict containing either:
                - 'logps': [batch, seq_len] pre-computed log probs, OR
                - 'logits': [batch, seq_len, vocab] from which logps will be computed
            ref_logps: [batch, seq_len] or List[List[float]] reference model log probs.
                      Can also be provided as separate ref_chosen_logps and ref_rejected_logps.
            ref_chosen_logps: [batch/2] pre-computed reference log probs for chosen.
            ref_rejected_logps: [batch/2] pre-computed reference log probs for rejected.
            **kwargs: Additional arguments.

        Returns:
            LossOutput with DPO loss and metrics.
        """
        import torch

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        batch_size = labels.shape[0]
        assert batch_size % 2 == 0, "Batch size must be even (chosen + rejected pairs)"
        half_batch = batch_size // 2

        # Get log probabilities from outputs
        logps = outputs.get('logps')
        if logps is None:
            logits = outputs.get('logits')
            assert logits is not None, "outputs must contain 'logps' or 'logits'"
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1]:]
            logps = self._compute_logps_from_logits(logits, labels)

        device = logps.device
        dtype = logps.dtype

        # Split into chosen and rejected
        chosen_labels = labels[:half_batch]
        rejected_labels = labels[half_batch:]
        chosen_logps = logps[:half_batch]
        rejected_logps = logps[half_batch:]

        # Compute sequence-level log probs for policy
        policy_chosen_logps = self._compute_sequence_logps(chosen_logps, chosen_labels)
        policy_rejected_logps = self._compute_sequence_logps(rejected_logps, rejected_labels)

        # Handle reference log probs
        if ref_chosen_logps is not None and ref_rejected_logps is not None:
            # Pre-computed sequence-level reference log probs provided
            reference_chosen_logps = ref_chosen_logps.to(device=device, dtype=dtype)
            reference_rejected_logps = ref_rejected_logps.to(device=device, dtype=dtype)
        elif ref_logps is not None:
            # Per-token reference log probs provided, need to align and sum
            loss_mask = (labels != self.ignore_index).bool()
            ref_logps_aligned = self._pad_and_align_logps(
                ref_logps, labels.shape, loss_mask, device, dtype
            )
            ref_chosen = ref_logps_aligned[:half_batch]
            ref_rejected = ref_logps_aligned[half_batch:]
            reference_chosen_logps = self._compute_sequence_logps(ref_chosen, chosen_labels)
            reference_rejected_logps = self._compute_sequence_logps(ref_rejected, rejected_labels)
        elif self.reference_free:
            # Reference-free mode: no reference model needed
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            raise ValueError(
                "ref_logps or (ref_chosen_logps, ref_rejected_logps) must be provided "
                "unless reference_free=True"
            )

        # Compute DPO loss
        loss = self._compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        return LossOutput(loss=loss, num_tokens=0)


class SimPOLoss(Loss):
    """SimPO (Simple Preference Optimization) Loss.

    SimPO is a simpler variant of DPO that doesn't require a reference model.
    It uses length-normalized log probabilities.

    Reference:
        "SimPO: Simple Preference Optimization with a Reference-Free Reward"
        (https://arxiv.org/abs/2405.14734)

    Args:
        beta: Temperature parameter (default: 2.5).
        gamma: Target reward margin (default: 0.5).
        ignore_index: Index to ignore in labels (default: -100).
    """

    def __init__(
        self,
        beta: float = 2.5,
        gamma: float = 0.5,
        ignore_index: int = -100,
        **kwargs,
    ):
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index

    def _compute_logps_from_logits(
        self,
        logits: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute per-token log probabilities from logits."""
        loss_mask = (labels != self.ignore_index).bool()
        masked_labels = labels.clone()
        masked_labels[~loss_mask] = 0
        logps = selective_log_softmax(logits, masked_labels)
        return logps

    def _compute_length_normalized_logps(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute length-normalized sequence log probabilities.

        Args:
            per_token_logps: [batch, seq_len] per-token log probabilities
            labels: [batch, seq_len] labels for computing mask

        Returns:
            normalized_logps: [batch] length-normalized log probabilities
        """
        loss_mask = (labels != self.ignore_index).float()
        seq_lengths = loss_mask.sum(dim=-1).clamp(min=1)
        seq_logps = (per_token_logps * loss_mask).sum(dim=-1)
        return seq_logps / seq_lengths

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        """Compute SimPO loss.

        Args:
            inputs: Dict containing 'input_ids' and 'labels' [batch, seq_len].
                   Batch: [chosen_1, ..., chosen_n, rejected_1, ..., rejected_n]
            outputs: Dict containing 'logps' or 'logits'.

        Returns:
            LossOutput with SimPO loss.
        """
        import torch
        import torch.nn.functional as F

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        batch_size = labels.shape[0]
        assert batch_size % 2 == 0, "Batch size must be even (chosen + rejected pairs)"
        half_batch = batch_size // 2

        # Get log probabilities
        logps = outputs.get('logps')
        if logps is None:
            logits = outputs.get('logits')
            assert logits is not None, "outputs must contain 'logps' or 'logits'"
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1]:]
            logps = self._compute_logps_from_logits(logits, labels)

        # Split into chosen and rejected
        chosen_labels = labels[:half_batch]
        rejected_labels = labels[half_batch:]
        chosen_logps = logps[:half_batch]
        rejected_logps = logps[half_batch:]

        # Compute length-normalized log probs
        chosen_rewards = self._compute_length_normalized_logps(chosen_logps, chosen_labels)
        rejected_rewards = self._compute_length_normalized_logps(rejected_logps, rejected_labels)

        # SimPO loss: -log(sigmoid(beta * (r_w - r_l) - gamma))
        logits = self.beta * (chosen_rewards - rejected_rewards) - self.gamma
        loss = -F.logsigmoid(logits).mean()

        return LossOutput(loss=loss, num_tokens=0)


class CPOLoss(Loss):
    """CPO (Contrastive Preference Optimization) Loss.

    CPO adds a behavior cloning term to preference optimization.

    Reference:
        "Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation"
        (https://arxiv.org/abs/2401.08417)

    Args:
        beta: Temperature parameter for preference (default: 0.1).
        bc_coef: Behavior cloning coefficient (default: 1.0).
        ignore_index: Index to ignore in labels (default: -100).
    """

    def __init__(
        self,
        beta: float = 0.1,
        bc_coef: float = 1.0,
        ignore_index: int = -100,
        **kwargs,
    ):
        self.beta = beta
        self.bc_coef = bc_coef
        self.ignore_index = ignore_index

    def _compute_logps_from_logits(
        self,
        logits: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute per-token log probabilities from logits."""
        loss_mask = (labels != self.ignore_index).bool()
        masked_labels = labels.clone()
        masked_labels[~loss_mask] = 0
        logps = selective_log_softmax(logits, masked_labels)
        return logps

    def _compute_sequence_logps(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute sequence-level log probabilities."""
        loss_mask = (labels != self.ignore_index).float()
        return (per_token_logps * loss_mask).sum(dim=-1)

    def _compute_nll_loss(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute negative log likelihood loss for chosen responses."""
        loss_mask = (labels != self.ignore_index).float()
        nll = -(per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        return nll

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        """Compute CPO loss.

        Args:
            inputs: Dict containing 'labels' [batch, seq_len].
            outputs: Dict containing 'logps' or 'logits'.

        Returns:
            LossOutput with CPO loss.
        """
        import torch
        import torch.nn.functional as F

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        batch_size = labels.shape[0]
        assert batch_size % 2 == 0, "Batch size must be even"
        half_batch = batch_size // 2

        # Get log probabilities
        logps = outputs.get('logps')
        if logps is None:
            logits = outputs.get('logits')
            assert logits is not None, "outputs must contain 'logps' or 'logits'"
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1]:]
            logps = self._compute_logps_from_logits(logits, labels)

        # Split into chosen and rejected
        chosen_labels = labels[:half_batch]
        rejected_labels = labels[half_batch:]
        chosen_logps = logps[:half_batch]
        rejected_logps = logps[half_batch:]

        # Compute sequence-level log probs
        chosen_seq_logps = self._compute_sequence_logps(chosen_logps, chosen_labels)
        rejected_seq_logps = self._compute_sequence_logps(rejected_logps, rejected_labels)

        # Preference loss (reference-free DPO)
        logits = self.beta * (chosen_seq_logps - rejected_seq_logps)
        preference_loss = -F.logsigmoid(logits).mean()

        # Behavior cloning loss on chosen
        bc_loss = self._compute_nll_loss(chosen_logps, chosen_labels)

        # Combined loss
        loss = preference_loss + self.bc_coef * bc_loss

        return LossOutput(loss=loss, num_tokens=0)


class ORPOLoss(Loss):
    """ORPO (Odds Ratio Preference Optimization) Loss.

    ORPO combines SFT and preference alignment in a single objective using odds ratios.

    Reference:
        "ORPO: Monolithic Preference Optimization without Reference Model"
        (https://arxiv.org/abs/2403.07691)

    Args:
        lambda_orpo: Weight for the odds ratio term (default: 0.1).
        ignore_index: Index to ignore in labels (default: -100).
    """

    def __init__(
        self,
        lambda_orpo: float = 0.1,
        ignore_index: int = -100,
        **kwargs,
    ):
        self.lambda_orpo = lambda_orpo
        self.ignore_index = ignore_index

    def _compute_logps_from_logits(
        self,
        logits: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute per-token log probabilities from logits."""
        loss_mask = (labels != self.ignore_index).bool()
        masked_labels = labels.clone()
        masked_labels[~loss_mask] = 0
        logps = selective_log_softmax(logits, masked_labels)
        return logps

    def _compute_avg_logps(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute average log probabilities over valid tokens."""
        loss_mask = (labels != self.ignore_index).float()
        seq_lengths = loss_mask.sum(dim=-1).clamp(min=1)
        return (per_token_logps * loss_mask).sum(dim=-1) / seq_lengths

    def _compute_nll_loss(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute negative log likelihood loss."""
        loss_mask = (labels != self.ignore_index).float()
        return -(per_token_logps * loss_mask).sum() / loss_mask.sum().clamp(min=1)

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        """Compute ORPO loss.

        Args:
            inputs: Dict containing 'labels' [batch, seq_len].
            outputs: Dict containing 'logps' or 'logits'.

        Returns:
            LossOutput with ORPO loss.
        """
        import torch
        import torch.nn.functional as F

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        batch_size = labels.shape[0]
        assert batch_size % 2 == 0, "Batch size must be even"
        half_batch = batch_size // 2

        # Get log probabilities
        logps = outputs.get('logps')
        if logps is None:
            logits = outputs.get('logits')
            assert logits is not None, "outputs must contain 'logps' or 'logits'"
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1]:]
            logps = self._compute_logps_from_logits(logits, labels)

        # Split into chosen and rejected
        chosen_labels = labels[:half_batch]
        rejected_labels = labels[half_batch:]
        chosen_logps = logps[:half_batch]
        rejected_logps = logps[half_batch:]

        # SFT loss on chosen
        sft_loss = self._compute_nll_loss(chosen_logps, chosen_labels)

        # Compute average log probs for odds ratio
        chosen_avg_logps = self._compute_avg_logps(chosen_logps, chosen_labels)
        rejected_avg_logps = self._compute_avg_logps(rejected_logps, rejected_labels)

        # Odds ratio: log(odds_chosen / odds_rejected)
        # log(p/(1-p)) ≈ log(p) - log(1-p) ≈ log(p) + p for small p
        # Simplified: log_odds = avg_logp (since p is small)
        log_odds_chosen = chosen_avg_logps - torch.log1p(-torch.exp(chosen_avg_logps).clamp(max=1-1e-7))
        log_odds_rejected = rejected_avg_logps - torch.log1p(-torch.exp(rejected_avg_logps).clamp(max=1-1e-7))

        # ORPO odds ratio loss
        odds_ratio = log_odds_chosen - log_odds_rejected
        orpo_loss = -F.logsigmoid(odds_ratio).mean()

        # Combined loss
        loss = sft_loss + self.lambda_orpo * orpo_loss

        return LossOutput(loss=loss, num_tokens=0)
