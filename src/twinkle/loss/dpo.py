# Copyright (c) ModelScope Contributors. All rights reserved.
"""
DPO (Direct Preference Optimization) Loss Implementation.

Reference:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (https://arxiv.org/abs/2305.18290)
"""
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss
from twinkle.utils.torch_utils import selective_log_softmax

if TYPE_CHECKING:
    import torch


class PreferenceLossBase(Loss):
    """Base class for preference optimization losses with shared utilities."""

    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index

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
        return selective_log_softmax(logits, masked_labels)

    def _compute_sequence_logps(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute sequence-level log probabilities by summing valid token logps."""
        loss_mask = (labels != self.ignore_index).float()
        return (per_token_logps * loss_mask).sum(dim=-1)

    def _compute_avg_logps(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute length-normalized (average) log probabilities."""
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

    def _get_logps_from_outputs(
        self,
        outputs: Dict,
        labels: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Extract or compute log probabilities from model outputs."""
        logps = outputs.get('logps')
        if logps is None:
            logits = outputs.get('logits')
            assert logits is not None, "outputs must contain 'logps' or 'logits'"
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1]:]
            logps = self._compute_logps_from_logits(logits, labels)
        return logps

    def _split_chosen_rejected(
        self,
        tensor: 'torch.Tensor',
    ) -> tuple:
        """Split interleaved tensor into chosen and rejected.

        Input format: [pos_1, neg_1, pos_2, neg_2, ...] (interleaved for DP-safe slicing)
        Output: (chosen [pos_1, pos_2, ...], rejected [neg_1, neg_2, ...])
        """
        # Even indices = chosen (positive), odd indices = rejected (negative)
        return tensor[0::2], tensor[1::2]


class DPOLoss(PreferenceLossBase):
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
        sft_weight: Weight for SFT loss on chosen responses to prevent likelihood displacement (default: 0.0).
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        loss_type: str = 'sigmoid',
        reference_free: bool = False,
        sft_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index)
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.reference_free = reference_free
        self.sft_weight = sft_weight

    def _align_logps(
        self,
        logps: 'torch.Tensor',
        target_shape: tuple,
        device: 'torch.device',
        dtype: 'torch.dtype',
    ) -> 'torch.Tensor':
        """Align log probabilities to target shape.

        Args:
            logps: Input log probabilities tensor
            target_shape: Target (batch, seq_len) shape
            device: Target device
            dtype: Target dtype

        Returns:
            Aligned tensor of shape target_shape
        """
        import torch

        if not torch.is_tensor(logps):
            raise TypeError(f'Expected torch.Tensor, got {type(logps)}')

        if logps.dim() == 1:
            logps = logps.unsqueeze(0)

        if logps.shape == target_shape:
            return logps.to(device=device, dtype=dtype)

        # Handle tensor with different sequence length
        if logps.dim() == 2 and logps.shape[0] == target_shape[0]:
            batch_size, target_seq_len = target_shape
            src_seq_len = logps.shape[1]
            logps = logps.to(device=device, dtype=dtype)
            if src_seq_len > target_seq_len:
                # Truncate right (keep left part) - may happen in Ray result merging
                return logps[:, :target_seq_len]
            else:
                raise ValueError(f'ref_logps seq_len ({src_seq_len}) < target seq_len ({target_seq_len}). '
                                 f'This should not happen when both models process the same batch.')

        raise ValueError(f'Cannot align ref_logps shape {logps.shape} to target shape {target_shape}')

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
            losses = (logits - 1 / (2 * self.beta))**2
        elif self.loss_type == 'kto_pair':
            # KTO pair loss (simplified version)
            chosen_logratios_scaled = self.beta * chosen_logratios
            rejected_logratios_scaled = self.beta * rejected_logratios
            chosen_losses = 1 - F.sigmoid(chosen_logratios_scaled)
            rejected_losses = F.sigmoid(rejected_logratios_scaled)
            losses = chosen_losses + rejected_losses
        else:
            raise ValueError(f'Unknown loss_type: {self.loss_type}')

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
        ref_outputs: Optional[Dict] = None,
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
            ref_outputs: Dict from reference model forward, containing 'logps'.
            ref_logps: [batch, seq_len] or List[List[float]] reference model log probs.
                      Can also be provided as separate ref_chosen_logps and ref_rejected_logps.
            ref_chosen_logps: [batch/2] pre-computed reference log probs for chosen.
            ref_rejected_logps: [batch/2] pre-computed reference log probs for rejected.
            **kwargs: Additional arguments.

        Returns:
            LossOutput with DPO loss and metrics.
        """
        import torch

        # Extract ref_logps from ref_outputs if provided
        if ref_outputs is not None and ref_logps is None:
            ref_logps = ref_outputs.get('logps')
        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        batch_size = labels.shape[0]
        assert batch_size % 2 == 0, 'Batch size must be even (chosen + rejected pairs)'

        # Get log probabilities from outputs
        logps = self._get_logps_from_outputs(outputs, labels)
        device = logps.device
        dtype = logps.dtype

        # Split into chosen and rejected
        chosen_labels, rejected_labels = self._split_chosen_rejected(labels)
        chosen_logps, rejected_logps = self._split_chosen_rejected(logps)

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
            ref_logps_aligned = self._align_logps(ref_logps, labels.shape, device, dtype)
            ref_chosen, ref_rejected = self._split_chosen_rejected(ref_logps_aligned)
            reference_chosen_logps = self._compute_sequence_logps(ref_chosen, chosen_labels)
            reference_rejected_logps = self._compute_sequence_logps(ref_rejected, rejected_labels)
        elif self.reference_free:
            # Reference-free mode: no reference model needed
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            return LossOutput(loss=torch.tensor(0.0, device=chosen_logps.device), num_tokens=0)

        # Compute DPO loss
        dpo_loss = self._compute_dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        # Add SFT loss on chosen responses to prevent likelihood displacement
        if self.sft_weight > 0:
            sft_loss = self._compute_nll_loss(chosen_logps, chosen_labels)
            loss = dpo_loss + self.sft_weight * sft_loss
        else:
            loss = dpo_loss

        # Return 0 to skip gradient normalization by num_tokens
        # DPO loss is already per-sample mean, unlike SFT which sums per-token loss
        # When num_tokens=0, normalize_and_clip_grad_norm defaults to 1 (no division)
        return LossOutput(loss=loss, num_tokens=0)


class SimPOLoss(PreferenceLossBase):
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
        super().__init__(ignore_index=ignore_index)
        self.beta = beta
        self.gamma = gamma

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        """Compute SimPO loss."""
        import torch
        import torch.nn.functional as F

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        assert labels.shape[0] % 2 == 0, 'Batch size must be even (chosen + rejected pairs)'

        # Get log probabilities
        logps = self._get_logps_from_outputs(outputs, labels)

        # Split into chosen and rejected
        chosen_labels, rejected_labels = self._split_chosen_rejected(labels)
        chosen_logps, rejected_logps = self._split_chosen_rejected(logps)

        # Compute length-normalized log probs
        chosen_rewards = self._compute_avg_logps(chosen_logps, chosen_labels)
        rejected_rewards = self._compute_avg_logps(rejected_logps, rejected_labels)

        # SimPO loss: -log(sigmoid(beta * (r_w - r_l) - gamma))
        logits = self.beta * (chosen_rewards - rejected_rewards) - self.gamma
        loss = -F.logsigmoid(logits).mean()

        return LossOutput(loss=loss, num_tokens=0)


class CPOLoss(PreferenceLossBase):
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
        super().__init__(ignore_index=ignore_index)
        self.beta = beta
        self.bc_coef = bc_coef

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        """Compute CPO loss."""
        import torch
        import torch.nn.functional as F

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        assert labels.shape[0] % 2 == 0, 'Batch size must be even'

        # Get log probabilities
        logps = self._get_logps_from_outputs(outputs, labels)

        # Split into chosen and rejected
        chosen_labels, rejected_labels = self._split_chosen_rejected(labels)
        chosen_logps, rejected_logps = self._split_chosen_rejected(logps)

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


class ORPOLoss(PreferenceLossBase):
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
        super().__init__(ignore_index=ignore_index)
        self.lambda_orpo = lambda_orpo

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        **kwargs,
    ) -> LossOutput:
        """Compute ORPO loss."""
        import torch
        import torch.nn.functional as F

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        assert labels.shape[0] % 2 == 0, 'Batch size must be even'

        # Get log probabilities
        logps = self._get_logps_from_outputs(outputs, labels)

        # Split into chosen and rejected
        chosen_labels, rejected_labels = self._split_chosen_rejected(labels)
        chosen_logps, rejected_logps = self._split_chosen_rejected(logps)

        # SFT loss on chosen
        sft_loss = self._compute_nll_loss(chosen_logps, chosen_labels)

        # Compute average log probs for odds ratio
        chosen_avg_logps = self._compute_avg_logps(chosen_logps, chosen_labels)
        rejected_avg_logps = self._compute_avg_logps(rejected_logps, rejected_labels)

        # Odds ratio: log(odds_chosen / odds_rejected)
        # log_odds = log(p/(1-p)) = log(p) - log(1-p)
        # Use numerically stable computation
        prob_chosen = torch.exp(chosen_avg_logps).clamp(min=1e-7, max=1 - 1e-7)
        prob_rejected = torch.exp(rejected_avg_logps).clamp(min=1e-7, max=1 - 1e-7)
        log_odds_chosen = torch.log(prob_chosen) - torch.log(1 - prob_chosen)
        log_odds_rejected = torch.log(prob_rejected) - torch.log(1 - prob_rejected)

        # ORPO odds ratio loss
        odds_ratio = log_odds_chosen - log_odds_rejected
        orpo_loss = -F.logsigmoid(odds_ratio).mean()

        # Combined loss
        loss = sft_loss + self.lambda_orpo * orpo_loss

        return LossOutput(loss=loss, num_tokens=0)
