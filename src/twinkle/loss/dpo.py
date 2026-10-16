# Copyright (c) ModelScope Contributors. All rights reserved.
"""
DPO (Direct Preference Optimization) Loss Implementation.

Reference:
    "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
    (https://arxiv.org/abs/2305.18290)
"""
import math
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss
from twinkle.utils.rl_tensor_utils import align_per_token_values
from twinkle.utils.torch_utils import selective_log_softmax

if TYPE_CHECKING:
    import torch


def _get_exp_cap(value: 'torch.Tensor', decimal: int = 4) -> 'torch.Tensor':
    """dtype-aware exponent cap: ``floor(log(finfo(dtype).max) * 10**decimal) / 10**decimal``.

    Mirrors the legacy DPOTrainer helper so the alpha-divergence branch saturates at the largest
    exponent the tensor dtype can represent instead of a hard-coded constant.
    """
    import torch
    vdtype_max = torch.zeros([1]).to(value.dtype) + torch.finfo(value.dtype).max
    vdtype_log_max = torch.log(vdtype_max).to(value.device)
    return torch.floor(vdtype_log_max * 10**decimal) / 10**decimal if decimal > 0 else vdtype_log_max


def _cap_exp(value: 'torch.Tensor', cap: float = -1) -> 'torch.Tensor':
    """``exp(clamp(value, max=cap))``; ``cap < 0`` derives the dtype-aware cap from ``_get_exp_cap``."""
    import torch
    cap = _get_exp_cap(value) if cap < 0 else cap
    return torch.exp(torch.clamp(value, max=cap))


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
        label_smoothing: Label smoothing parameter for soft labels (default: 0.0). Only applied on the
            'sigmoid' variant; rejected unless 'sigmoid' is among ``loss_type``.
        ignore_index: Index to ignore in labels (default: -100).
        loss_type: One DPO variant or a list of variants combined as a weighted sum (MPO). Each item is
            one of 'sigmoid', 'hinge', 'ipo', 'kto_pair', 'discopop' (default: 'sigmoid').
        reference_free: Whether to use reference-free DPO (default: False).
        sft_weight: Weight for SFT loss on chosen responses to prevent likelihood displacement (default: 0.0).
        ld_alpha: Length-desensitization coefficient. When set, the policy sequence logps keep the shared
            prefix of each chosen/rejected pair at full weight and down-weight the length-dependent suffix
            by ``ld_alpha`` (applied to the policy only, never the reference) (default: None).
        discopop_tau: Temperature of the DiscoPOP log-ratio modulation gate (default: 0.05).
        f_divergence_type: f-divergence reshaping the preference margin before the per-variant transform;
            one of 'reverse_kl', 'forward_kl', 'js_divergence', 'alpha_divergence' (default: 'reverse_kl').
            'reverse_kl' and 'forward_kl' both reduce to the plain Bradley-Terry margin.
        f_alpha_divergence_coef: Coefficient of the 'alpha_divergence' branch (default: 0.5).
        loss_weights: Per-variant weights aligned with ``loss_type`` when it is a list (MPO). If given, its
            length must equal the number of variants; otherwise every variant weighs 1.0 (default: None).
    """

    _DPO_VARIANTS = ('sigmoid', 'hinge', 'ipo', 'kto_pair', 'discopop')
    _F_DIVERGENCE_TYPES = ('reverse_kl', 'forward_kl', 'js_divergence', 'alpha_divergence')

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        ignore_index: int = -100,
        loss_type: Union[str, List[str]] = 'sigmoid',
        reference_free: bool = False,
        sft_weight: float = 0.0,
        ld_alpha: Optional[float] = None,
        discopop_tau: float = 0.05,
        f_divergence_type: str = 'reverse_kl',
        f_alpha_divergence_coef: float = 0.5,
        loss_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index)
        loss_types = [loss_type] if isinstance(loss_type, str) else list(loss_type)
        if not loss_types:
            raise ValueError('loss_type must name at least one DPO variant.')
        for variant in loss_types:
            if variant not in self._DPO_VARIANTS:
                raise ValueError(f'Unknown loss_type: {variant}')
        if loss_weights is not None and len(loss_weights) != len(loss_types):
            raise ValueError(f'loss_weights must align with loss_type '
                             f'({len(loss_weights)} weights for {len(loss_types)} variants).')
        if label_smoothing > 0 and 'sigmoid' not in loss_types:
            raise ValueError('label_smoothing > 0 is only defined for loss_type="sigmoid", '
                             f'got loss_type={loss_types}. Set label_smoothing=0.0 or include sigmoid.')
        if f_divergence_type not in self._F_DIVERGENCE_TYPES:
            raise ValueError(f'Unknown f_divergence_type: {f_divergence_type}')
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_types = loss_types
        self.reference_free = reference_free
        self.sft_weight = sft_weight
        self.ld_alpha = ld_alpha
        self.discopop_tau = discopop_tau
        self.f_divergence_type = f_divergence_type
        self.f_alpha_divergence_coef = f_alpha_divergence_coef
        self.loss_weights = loss_weights

    def _compute_dpo_loss(
        self,
        policy_chosen_logps: 'torch.Tensor',
        policy_rejected_logps: 'torch.Tensor',
        reference_chosen_logps: 'torch.Tensor',
        reference_rejected_logps: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Compute the DPO loss, summing every configured variant weighted by ``loss_weights`` (MPO).

        Args:
            policy_chosen_logps: [batch/2] log probs of chosen under current policy
            policy_rejected_logps: [batch/2] log probs of rejected under current policy
            reference_chosen_logps: [batch/2] log probs of chosen under reference policy
            reference_rejected_logps: [batch/2] log probs of rejected under reference policy

        Returns:
            loss: Scalar DPO loss
        """
        import torch.nn.functional as F

        # Compute log ratios
        if self.reference_free:
            # Reference-free: only use policy log probs
            chosen_logratios = policy_chosen_logps
            rejected_logratios = policy_rejected_logps
        else:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

        # Plain Bradley-Terry margin h = (chosen - rejected). beta is applied per variant rather than folded
        # in here, because ipo operates on h itself (its target 1/(2*beta) lives in h-space), not on beta*h.
        plain_margin = chosen_logratios - rejected_logratios

        # Divergence-aware margin: the f-divergence family reshapes h before the per-variant transform.
        # reverse_kl (default) and forward_kl both reduce to the plain margin (legacy does not fork them);
        # js_divergence subtracts the softplus difference; alpha_divergence uses the capped-exponential form.
        if self.f_divergence_type == 'alpha_divergence':
            coef = self.f_alpha_divergence_coef
            divergence_margin = (_cap_exp(rejected_logratios * -coef) - _cap_exp(chosen_logratios * -coef)) / coef
        else:
            divergence_margin = plain_margin
            if self.f_divergence_type == 'js_divergence':
                divergence_margin = divergence_margin - (
                    F.softplus(chosen_logratios) - F.softplus(rejected_logratios))

        # Sum the per-variant, per-sample losses weighted by loss_weights (each 1.0 when unset), then mean.
        # The margin / divergence terms are computed once and shared by every variant.
        weights = self.loss_weights
        total = None
        for idx, loss_type in enumerate(self.loss_types):
            variant = self._variant_losses(loss_type, divergence_margin, plain_margin, chosen_logratios,
                                           rejected_logratios)
            weight = weights[idx] if weights is not None else 1.0
            total = variant * weight if total is None else total + variant * weight
        return total.mean()

    def _variant_losses(
        self,
        loss_type: str,
        divergence_margin: 'torch.Tensor',
        plain_margin: 'torch.Tensor',
        chosen_logratios: 'torch.Tensor',
        rejected_logratios: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """Per-sample losses for a single DPO variant (element-wise faithful to the legacy branch).

        sigmoid/hinge/ipo consume the divergence-aware margin; kto_pair uses the raw log-ratios; discopop
        recomputes from the plain margin because the legacy discopop branch ignores f_divergence.
        """
        import torch
        import torch.nn.functional as F

        if loss_type == 'sigmoid':
            # Standard DPO loss: -log(sigmoid(beta * h))
            logits = self.beta * divergence_margin
            losses = -F.logsigmoid(logits)
            # Apply label smoothing (only meaningful here: Bradley-Terry soft labels).
            if self.label_smoothing > 0:
                # Soft labels: (1 - eps) * loss_chosen + eps * loss_rejected
                smooth_losses = -F.logsigmoid(-logits)  # Loss for flipped preference
                losses = (1 - self.label_smoothing) * losses + self.label_smoothing * smooth_losses
            return losses
        if loss_type == 'hinge':
            # Hinge loss variant
            return torch.relu(1 - self.beta * divergence_margin)
        if loss_type == 'ipo':
            # IPO (Identity Preference Optimization) loss
            # Reference: "A General Theoretical Paradigm to Understand Learning from Human Feedback"
            # eq 17: (h - 1/(2*beta))**2 on the raw margin h (NOT beta-scaled).
            return (divergence_margin - 1 / (2 * self.beta))**2
        if loss_type == 'kto_pair':
            # KTO pair loss (simplified version)
            chosen_losses = 1 - F.sigmoid(self.beta * chosen_logratios)
            rejected_losses = F.sigmoid(self.beta * rejected_logratios)
            return chosen_losses + rejected_losses
        if loss_type == 'discopop':
            # DiscoPOP: legacy recomputes the plain margin here (f_divergence is NOT applied) and scales by beta.
            logits = self.beta * plain_margin
            log_ratio_modulation = torch.sigmoid(logits / self.discopop_tau)
            logistic_component = -F.logsigmoid(logits)
            exp_component = torch.exp(-logits)
            return logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation
        raise ValueError(f'Unknown loss_type: {loss_type}')

    def _compute_sequence_logps_ld(
        self,
        per_token_logps: 'torch.Tensor',
        labels: 'torch.Tensor',
        ld_alpha: float,
    ) -> 'torch.Tensor':
        """Length-desensitization sequence logps over the INTERLEAVED layout [c1, r1, c2, r2, ...].

        Pair k = (row 2k, row 2k+1). The shared prefix of length min(len_c, len_r) is kept at full weight
        and the length-dependent suffix is down-weighted by ``ld_alpha``. This is the legacy DPOTrainer
        standard branch adapted from the split [chosen..., rejected...] layout to the interleaved one:
        ``public.repeat_interleave(2)`` maps public[k] onto rows 2k and 2k+1.
        """
        import torch
        loss_mask = (labels != self.ignore_index)
        completion_lengths = loss_mask.sum(dim=1)
        public_lengths = torch.minimum(completion_lengths[0::2], completion_lengths[1::2])
        public_row = public_lengths.repeat_interleave(2)
        # cumsum over the float mask gives the 1-indexed position within the completion; shift to 0-indexed.
        position_ids = (loss_mask.float().cumsum(dim=1) - 1) * loss_mask.float()
        ld_mask = position_ids < public_row.unsqueeze(1)
        front_mask = (ld_mask & loss_mask).float()
        rear_mask = (~ld_mask & loss_mask).float()
        front_logps = (per_token_logps * front_mask).sum(dim=1)
        rear_logps = (per_token_logps * rear_mask).sum(dim=1)
        return front_logps + ld_alpha * rear_logps

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
                   Batch should be interleaved as [chosen_1, rejected_1, chosen_2, rejected_2, ...]
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

        # Compute sequence-level log probs for policy. With ld_alpha the length-desensitization front/rear
        # split must be paired across the interleaved layout, so it runs on the full [B, T] tensor before
        # splitting. Legacy applies ld_alpha to the policy only; the reference logps below keep the plain sum.
        if self.ld_alpha is not None:
            policy_logps = self._compute_sequence_logps_ld(logps, labels, self.ld_alpha)
            policy_chosen_logps, policy_rejected_logps = self._split_chosen_rejected(policy_logps)
        else:
            policy_chosen_logps = self._compute_sequence_logps(chosen_logps, chosen_labels)
            policy_rejected_logps = self._compute_sequence_logps(rejected_logps, rejected_labels)

        # Handle reference log probs
        if ref_chosen_logps is not None and ref_rejected_logps is not None:
            # Pre-computed sequence-level reference log probs provided
            reference_chosen_logps = torch.as_tensor(ref_chosen_logps, device=device, dtype=dtype)
            reference_rejected_logps = torch.as_tensor(ref_rejected_logps, device=device, dtype=dtype)
        elif ref_logps is not None:
            # Per-token reference log probs provided, need to align and sum
            ref_logps_aligned = align_per_token_values(
                ref_logps,
                tuple(labels.shape),
                device=device,
                dtype=dtype,
                name='ref_logps',
                valid_mask=labels != self.ignore_index,
            )
            ref_chosen, ref_rejected = self._split_chosen_rejected(ref_logps_aligned)
            reference_chosen_logps = self._compute_sequence_logps(ref_chosen, chosen_labels)
            reference_rejected_logps = self._compute_sequence_logps(ref_rejected, rejected_labels)
        elif self.reference_free:
            # Reference-free mode: no reference model needed
            reference_chosen_logps = torch.zeros_like(policy_chosen_logps)
            reference_rejected_logps = torch.zeros_like(policy_rejected_logps)
        else:
            zero = (policy_chosen_logps.sum() + policy_rejected_logps.sum()) * 0.0
            return LossOutput(loss=zero, num_tokens=0)

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


class KTOLoss(PreferenceLossBase):
    """KTO (Kahneman-Tversky Optimization) Loss -- the UNPAIRED preference objective.

    Where DPO needs a chosen/rejected pair per prompt, KTO needs only a single completion carrying a
    binary label (desirable / undesirable), so it trains directly on the asymmetric feedback people
    actually give. Each example is scored against a reference POINT ``z_KL`` rather than against a
    paired counterpart:

        desirable   (label=True):   L = desirable_weight   * (1 - sigmoid(beta * (logratio - z_KL)))
        undesirable (label=False):  L = undesirable_weight  * (1 - sigmoid(beta * (z_KL - logratio)))

    with ``logratio = log pi(y|x) - log pi_ref(y|x)`` per example and ``z_KL`` a DETACHED, clamped
    estimate of ``KL(pi || pi_ref)`` over a mismatched "KL batch" (a prompt paired with an unrelated
    completion). Anchoring on z_KL -- not on a paired response -- is what lets KTO absorb an imbalanced
    desirable/undesirable mix; the two weights counteract that imbalance.

    ``z_KL`` is supplied by the caller: the training loop owns the extra KL-batch forwards and reduces
    them to this detached scalar (mirroring TRL, whose ``kto_loss`` receives the KL logps and forms
    ``kl = mean(policy_KL - reference_KL).clamp(min=0).detach()``). ``z_kl=None`` -- ``calculate_KL``
    off -- anchors at 0, the degenerate no-KL form.

    Reference:
        "KTO: Model Alignment as Prospect Theoretic Optimization" (https://arxiv.org/abs/2402.01306)

    Args:
        beta: Temperature on the implicit reward (default: 0.1).
        desirable_weight: Weight on the desirable (label=True) term (default: 1.0).
        undesirable_weight: Weight on the undesirable (label=False) term (default: 1.0).
        ignore_index: Index to ignore in labels (default: -100).
    """

    def __init__(
        self,
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index)
        self.beta = beta
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        label: Optional[Union['torch.Tensor', List[bool]]] = None,
        z_kl: Optional[float] = None,
        ref_outputs: Optional[Dict] = None,
        ref_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        **kwargs,
    ) -> LossOutput:
        """Compute the KTO loss over an UNPAIRED completion batch.

        Args:
            inputs: Dict with 'labels' [batch, seq_len]; one completion per row. The batch is split by
                ``label``, NOT by position -- there is no chosen/rejected interleaving as in DPO.
            outputs: Dict with 'logps' [batch, seq_len] (or 'logits' to compute them from).
            label: Per-row binary label; True marks a desirable example, False an undesirable one.
            z_kl: The detached KL(pi||pi_ref) reference point estimated over the mismatched KL batch,
                or None to anchor at 0 (calculate_KL off).
            ref_outputs / ref_logps: Reference-model per-token logps for the completion batch.
        """
        import torch

        if ref_outputs is not None and ref_logps is None:
            ref_logps = ref_outputs.get('logps')
        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if label is None:
            raise ValueError("KTO needs a per-example binary 'label' (desirable/undesirable); the loop "
                             'must forward it through forward_backward.')
        if ref_logps is None:
            # KTO's implicit reward IS the reference log-ratio and z_KL is a KL against the reference, so
            # there is no reference-free form -- fail loudly rather than anchor on a silent zero.
            raise ValueError('KTO requires reference logps (log pi(y|x) - log pi_ref(y|x)); build a '
                             'reference model or use the LoRA adapter-disabled base as the reference.')

        logps = self._get_logps_from_outputs(outputs, labels)
        device, dtype = logps.device, logps.dtype
        policy_logps = self._compute_sequence_logps(logps, labels)
        ref_logps_aligned = align_per_token_values(
            ref_logps,
            tuple(labels.shape),
            device=device,
            dtype=dtype,
            name='ref_logps',
            valid_mask=labels != self.ignore_index,
        )
        reference_logps = self._compute_sequence_logps(ref_logps_aligned, labels)

        label_mask = torch.as_tensor(label, device=device).bool().reshape(-1)
        if label_mask.shape[0] != policy_logps.shape[0]:
            raise ValueError(f'KTO label count ({label_mask.shape[0]}) does not match the completion '
                             f'batch size ({policy_logps.shape[0]}).')

        z_kl_value = (torch.as_tensor(z_kl, device=device, dtype=dtype).reshape(())
                      if z_kl is not None else torch.zeros((), device=device, dtype=dtype))
        logratios = policy_logps - reference_logps
        chosen_losses = 1 - torch.sigmoid(self.beta * (logratios[label_mask] - z_kl_value))
        rejected_losses = 1 - torch.sigmoid(self.beta * (z_kl_value - logratios[~label_mask]))
        # Either side may be empty (an all-desirable or all-undesirable batch); cat + mean handles that,
        # each side scaled by its imbalance-correcting weight.
        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0)
        return LossOutput(loss=losses.mean(), num_tokens=0)


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
        # Compute log(1-p) = log(1 - exp(avg_logp)) numerically stably:
        #   - For x > -log(2):  log(-expm1(x))  (avoids log(0) when p → 1)
        #   - For x ≤ -log(2): log1p(-exp(x))  (avoids cancellation when p → 0)
        # ``avg_logp ∈ (-∞, 0]`` so the threshold partitions the safe regime.
        log_two = math.log(2.0)

        def _log1mexp(x: 'torch.Tensor') -> 'torch.Tensor':
            # Clamp at a tiny negative to keep both branches well-defined when p≈1.
            x_safe = torch.clamp(x, max=-1e-7)
            return torch.where(
                x_safe > -log_two,
                torch.log(-torch.expm1(x_safe)),
                torch.log1p(-torch.exp(x_safe)),
            )

        log_odds_chosen = chosen_avg_logps - _log1mexp(chosen_avg_logps)
        log_odds_rejected = rejected_avg_logps - _log1mexp(rejected_avg_logps)

        # ORPO odds ratio loss
        odds_ratio = log_odds_chosen - log_odds_rejected
        orpo_loss = -F.logsigmoid(odds_ratio).mean()

        # Combined loss
        loss = sft_loss + self.lambda_orpo * orpo_loss

        return LossOutput(loss=loss, num_tokens=0)
