# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from twinkle.data_format import LossOutput
from twinkle.loss.grpo import GRPOLoss

if TYPE_CHECKING:
    import torch


class OPSDLoss(GRPOLoss):
    """On-Policy Self-Distillation (OPSD) loss.

    Reference:
        "Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models"
        (Zhao et al., arXiv:2601.18734).

    A single model acts as BOTH teacher and student, differing only in context:
      * student policy conditions on the QUESTION ONLY (query-only prompt);
      * teacher policy conditions on PRIVILEGED information (question + rubric diagnosis).
    Training minimizes a per-token divergence between the two distributions over the
    STUDENT's own on-policy rollout (the tokens the student generated under the
    query-only prompt). Because both forwards score the SAME response tokens, only the
    prompt differs, so the per-token alignment is exact.

    Token-probability (sampled-token) form — v1, zero extra tensor channel
    ----------------------------------------------------------------------
    We only need the per-token log-prob of the SAMPLED tokens from each context
    (``teacher_logps`` from a teacher forward on the rubric-conditioned trajectory,
    ``logps`` from the student forward on the query-only trajectory). Reusing the exact
    k3 estimator already used by the GRPO KL penalty
    (``grpo.py``: ``exp(ref - logps) - (ref - logps) - 1``), the per-token loss is::

        r = teacher_logp - student_logp          # teacher detached
        per_token = exp(r) - r - 1                # k3 estimate, >= 0, pulls student -> teacher

    Its gradient w.r.t. the student log-prob is ``1 - exp(r)``: when the teacher assigns
    higher probability than the student (``r > 0``) the update RAISES the student log-prob
    toward the teacher, and lowers it when ``r < 0`` — a dense token-level distillation
    pull, no advantages / reward needed.

    Aggregation is BNPO-style token-mean (sum over all response tokens / total token count),
    matching the RL branch so OPSD and BNPO experiments share the same effective step scaling.

    Notes
    -----
    * ``teacher_logps`` is accepted via a dedicated kwarg; for pipelines that route the teacher
      log-probs through the existing reference channel it also falls back to ``ref_logps``.
      Provide it in the RESPONSE-ONLY form (one log-prob per trainable/response token, matching
      the student loss mask) — ``_pad_and_align_to_batch`` scatters it onto the response
      positions. The teacher and student prompts differ in length, so the full-sequence
      (right-padded) form must NOT be used here.
    * The divergence direction (this k3 form corresponds to KL(student || teacher)) should be
      re-confirmed against the official code release before treating it as final; it is exposed
      via ``reverse`` for a quick swap without touching call sites.
    """

    require_logps = True
    require_logits = False

    def __init__(
        self,
        beta: float = 0.0,
        ignore_index: int = -100,
        reverse: bool = True,
        **kwargs,
    ):
        # epsilon is unused (no PPO ratio here) but kept in the ctor so the shared
        # ``set_loss(epsilon=..., beta=...)`` call site does not need special-casing.
        super().__init__(epsilon=kwargs.pop('epsilon', 0.2), beta=beta,
                         ignore_index=ignore_index, **kwargs)
        self.reverse = reverse

    def _aggregate_loss(self, per_token_loss, loss_mask, **kwargs):
        """BNPO-style token-mean: sum over all response tokens / total token count."""
        return (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    def __call__(
        self,
        inputs: Dict,
        outputs: Dict,
        *,
        teacher_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        ref_logps: Optional[Union['torch.Tensor', List[List[float]]]] = None,
        **kwargs,
    ) -> LossOutput:
        import torch

        labels = inputs.get('labels')
        assert labels is not None, "inputs must contain 'labels'"
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        logps = outputs.get('logps')
        loss_mask = (labels != self.ignore_index).bool()
        if logps is None:
            from twinkle.utils.torch_utils import selective_log_softmax
            logits = outputs.get('logits')
            if logits.shape[1] != labels.shape[1]:
                logits = logits[:, -labels.shape[1]:]
            masked_labels = labels.clone()
            masked_labels[~loss_mask] = 0
            logps = selective_log_softmax(logits, masked_labels)

        device = logps.device

        # Teacher log-probs: prefer the dedicated kwarg, else reuse the reference channel.
        teacher = teacher_logps if teacher_logps is not None else ref_logps
        # Without a teacher this reduces to a no-op that still flows through autograd, so
        # ref-only / eval forwards (which harvest outputs['logps']) do not crash and DDP/FSDP
        # never see unused parameters. Mirrors GRPOLoss's advantages-None guard.
        if teacher is None:
            return LossOutput(loss=logps.sum() * 0.0, num_tokens=0)

        teacher = self._pad_and_align_to_batch(teacher, loss_mask, device, logps.dtype)
        teacher = teacher.detach()

        # r = teacher - student. k3 KL estimate: exp(r) - r - 1 (>= 0), pulls student -> teacher.
        r = teacher - logps if self.reverse else logps - teacher
        r = torch.clamp(r, min=-10.0, max=10.0)  # guard exp overflow on rare huge gaps
        per_token_loss = torch.exp(r) - r - 1

        loss = self._aggregate_loss(per_token_loss, loss_mask, **kwargs)
        return LossOutput(loss=loss, num_tokens=0)
