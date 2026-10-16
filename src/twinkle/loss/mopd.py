# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

from twinkle.data_format import LossOutput
from twinkle.loss.opsd import OPSDLoss

if TYPE_CHECKING:
    import torch


class MOPDLoss(OPSDLoss):
    """Multi-Teacher On-Policy Distillation (MOPD) loss.

    Reference:
        "Open-MOPD: Multi-Teacher On-Policy Distillation" (arXiv:2606.30406).

    MOPD generalizes OPSD from a single privileged teacher to K domain teachers. Each teacher scores
    the SAME student on-policy rollout (the tokens the student generated), giving K response-only
    per-token log-prob channels ``teacher_logps[k]``. The K channels are fused into ONE target
    distribution before the k3 distillation surrogate, so the update stays a single dense token-level
    pull regardless of how many teachers there are.

    Fusion: weighted probability mixture
    -------------------------------------
    The teachers are combined into a single mixture distribution over the sampled token::

        p_mix = sum_k w_k * p_k          (w_k >= 0, sum_k w_k == 1)
        teacher_logp_mix = log p_mix = logsumexp_k(teacher_logp_k + log w_k)

    i.e. the log of the weighted AVERAGE of the teacher probabilities, computed stably in log space via
    ``logsumexp``. The student is then pulled toward ``teacher_logp_mix`` by the exact k3 surrogate
    OPSD uses (``_per_token_distill_loss``: ``r = teacher_logp_mix - student_logp``; ``exp(r) - r - 1``),
    aggregated BNPO-style.

    Why a probability mixture (not a weighted sum of per-teacher losses, nor a geometric mean):
      * A mixture yields ONE well-defined target distribution, matching the plan's "聚合成一个 token 级
        蒸馏目标" — the loss remains a single KL-style pull, so its scale does not grow with K.
      * The geometric mean (``sum_k w_k * teacher_logp_k``) would target the PRODUCT-of-experts
        distribution, which is sharper and can be dominated by whichever teacher is most confident on a
        token; the mixture is the standard multi-teacher distillation target and is more forgiving when
        teachers disagree.

    NOTE (no-oracle side, flagged for the test phase): unlike OPSD, whose single-teacher k3 form is
    pinned to the paper, MOPD's mixture fusion has no external reference implementation to diff
    against. Its two self-consistency anchors are (1) K=1, w=[1.0] degenerates EXACTLY to ``OPSDLoss``
    (``logsumexp`` of one unit-weighted channel is that channel, ``log 1 == 0``) and (2) a hand-worked
    two-teacher example checked against the closed-form ``log(w1*p1 + w2*p2)``.

    Contract
    --------
    * ``teacher_logps``: a non-empty ``List`` of K teacher channels, each a ``torch.Tensor`` or
      ``List[List[float]]`` in the RESPONSE-ONLY form (one log-prob per trainable/response token,
      matching the student loss mask). ``_pad_and_align_to_batch`` scatters each onto the response
      positions; the K channels must all describe the SAME tokens so they align to one shared mask.
    * ``teacher_weights``: optional length-K non-negative weights, normalized to sum 1. ``None`` means
      uniform ``1/K``. Length must equal K and the (pre-normalization) sum must be positive.
    * A single teacher passed as a bare ``torch.Tensor`` or ``List[List[float]]`` (OPSD's own input
      forms) is delegated to ``OPSDLoss.__call__`` unchanged, so MOPD is a strict superset of OPSD and
      the two are interchangeable at a call site.
    """

    def _is_multi_teacher(self, teacher_logps) -> bool:
        """True iff ``teacher_logps`` is the MOPD list-of-K-channels form (not an OPSD single teacher).

        The discriminator is the depth/type of the FIRST element, because both forms are lists:
          * ``torch.Tensor``                     -> single teacher (OPSD).
          * ``List[Tensor]``                     -> multi-teacher (each element is a teacher channel).
          * ``List[List[float]]``                -> single teacher, per-sample sequences (OPSD).
          * ``List[List[List[float]]]`` /
            ``List[List[Tensor]]``               -> multi-teacher, each in per-sample form.
        An empty/None value is not multi-teacher; it falls through to ``OPSDLoss`` which turns a missing
        teacher into the zero-loss-through-autograd no-op.
        """
        import torch

        if teacher_logps is None or isinstance(teacher_logps, torch.Tensor):
            return False
        if not isinstance(teacher_logps, (list, tuple)) or len(teacher_logps) == 0:
            return False
        first = teacher_logps[0]
        if isinstance(first, torch.Tensor):
            return True
        if isinstance(first, (list, tuple)) and len(first) > 0:
            # Single teacher per-sample form is List[List[float]] (numbers inside); multi-teacher is a
            # deeper list whose first element is itself a sequence/tensor, not a bare number.
            return not isinstance(first[0], (int, float))
        return False

    def _normalize_teacher_weights(self, teacher_weights, num_teachers, device, dtype) -> 'torch.Tensor':
        """Return length-``num_teachers`` weights that are non-negative and sum to 1.

        ``None`` -> uniform ``1/K``. Otherwise validates the weights align one-to-one with the teacher
        channels, rejects negatives and a non-positive total (fail loudly rather than silently emit
        ``log(0) = -inf`` or a zero-gradient mixture), then normalizes.
        """
        import torch

        if teacher_weights is None:
            return torch.full((num_teachers, ), 1.0 / num_teachers, device=device, dtype=dtype)

        weights = torch.as_tensor(teacher_weights, device=device, dtype=dtype).flatten()
        if weights.numel() != num_teachers:
            raise ValueError(f'teacher_weights has {weights.numel()} entries but there are '
                             f'{num_teachers} teacher channels; they must align one-to-one.')
        if torch.any(weights < 0):
            raise ValueError('teacher_weights must be non-negative.')
        total = weights.sum()
        if not torch.isfinite(total) or total <= 0:
            raise ValueError(f'teacher_weights must sum to a positive finite value, got {float(total)}.')
        return weights / total

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        teacher_logps: Optional[Union['torch.Tensor', Sequence]] = None,
        teacher_weights: Optional[Sequence[float]] = None,
        ref_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        **kwargs,
    ) -> LossOutput:
        # Single-teacher / missing-teacher inputs are OPSD's contract verbatim -- delegate so MOPD is a
        # strict superset and the two losses are drop-in interchangeable at a call site.
        if teacher_logps is None or not self._is_multi_teacher(teacher_logps):
            return super().__call__(
                inputs, outputs, teacher_logps=teacher_logps, ref_logps=ref_logps, **kwargs)

        import torch

        logps, loss_mask = self._student_logps_and_mask(inputs, outputs)
        device = logps.device

        num_teachers = len(teacher_logps)
        weights = self._normalize_teacher_weights(teacher_weights, num_teachers, device, logps.dtype)

        # Weighted probability mixture in log space: logsumexp_k(teacher_logp_k + log w_k). Zero-weight
        # teachers are skipped so they never contribute log(0) = -inf; the positive-total check above
        # guarantees at least one channel survives.
        weighted_channels: List['torch.Tensor'] = []
        for k in range(num_teachers):
            weight = weights[k]
            if float(weight) <= 0.0:
                continue
            aligned = self._pad_and_align_to_batch(teacher_logps[k], loss_mask, device, logps.dtype)
            weighted_channels.append(aligned.detach() + torch.log(weight))
        teacher_mix = torch.logsumexp(torch.stack(weighted_channels, dim=0), dim=0)

        per_token_loss = self._per_token_distill_loss(teacher_mix, logps)
        loss = self._aggregate_loss(per_token_loss, loss_mask, **kwargs)
        return LossOutput(loss=loss, num_tokens=0)
