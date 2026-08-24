# Copyright (c) ModelScope Contributors. All rights reserved.
"""Liger Fused Linear GRPO loss.

Fuses the final ``lm_head`` matmul with the GRPO policy-gradient objective so
the full ``(B, T, V)`` logits tensor is never materialised — the same memory +
bandwidth win as the fused cross-entropy loss, but for RL fine-tuning of
large-vocab models.

Contract with the model forward
--------------------------------
Like :class:`~twinkle.loss.liger_fused_linear_cross_entropy.LigerFusedLinearCrossEntropyLoss`
this loss sets ``require_logits = True`` / ``require_logps = False`` so
``TransformersModel.forward`` keeps ``outputs['logits']`` and skips the
``selective_log_softmax`` pass. Under ``TransformersFusedCEPatch`` (applied via
``task='fused_lm_ce'``) the lm_head is replaced by identity, so
``outputs['logits']`` is actually the last hidden state ``[B, T, H]`` and the
patch stashes the lm_head module under ``outputs['lm_head']``. This loss then
reads both and calls Liger's fused GRPO kernel
(``liger_kernel.chunked_loss.LigerFusedLinearGRPOLoss``), which computes the
per-token log-probs internally from the hidden states and ``lm_head.weight``.

Defensive fallback (device-agnostic)
------------------------------------
This class subclasses :class:`~twinkle.loss.grpo.GRPOLoss`, so whenever the
fused path is unusable it degrades transparently to the pure-torch GRPO family
loss (same objective, just without the fusion). The fused path is skipped when:

  * ``outputs['lm_head']`` is absent — the forward did not run under
    ``TransformersFusedCEPatch`` (e.g. an eval / ref-logps forward). Here
    ``outputs['logits']`` already holds real logits, so the base ``GRPOLoss``
    consumes them directly.
  * ``advantages is None`` — an ill-defined GRPO step (ref-logps-only / eval);
    the base loss returns a zero loss that still flows through autograd.
  * ``liger_kernel`` is not installed, or the fused kernel raises for any reason
    (unsupported shape on the Ascend backend, a Triton compile failure on CUDA,
    a version mismatch, an OOM inside the fused kernel). The loss logs one
    warning, marks the fused path broken so subsequent calls skip straight to
    the fallback, materialises the logits via ``F.linear(hidden, lm_head.weight)``
    and runs the standard GRPO objective on them.

So selecting this loss is always safe on both NPU and CUDA: best case the fused
kernel saves the logits memory, worst case it degrades to the standard GRPO
path. There are no ``torch.cuda`` / ``Platform.is_npu`` probes — hardware
dispatch is delegated to Liger, and the guard is purely exception-based.

No token shift here
-------------------
Twinkle templates pre-shift labels via ``_roll_labels`` (``template/base.py``),
so ``labels[i]`` is already the target for the prediction made at position
``i`` and ``hidden[i]`` predicts it. Liger's fused GRPO forward consumes the
``(hidden, selected_token_ids, attention_mask)`` triple positionally aligned —
matching how the fused-CE loss feeds ``(hidden, labels)`` without re-shifting.
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np

from twinkle import get_logger
from twinkle.data_format import LossOutput
from .grpo import GRPOLoss

logger = get_logger()

# Lazily import Liger so the module loads even when liger_kernel is absent; the
# loss then falls back to the pure-torch GRPO objective instead of hard-failing.
_LigerFLGRPOModule = None


def _get_liger_module():
    global _LigerFLGRPOModule
    if _LigerFLGRPOModule is not None:
        return _LigerFLGRPOModule
    from liger_kernel.chunked_loss import LigerFusedLinearGRPOLoss as _Mod  # noqa: F401
    _LigerFLGRPOModule = _Mod
    return _LigerFLGRPOModule


class LigerFusedLinearGRPOLoss(GRPOLoss):
    """Fused lm_head + GRPO policy-gradient loss (Liger kernel).

    Args:
        beta: KL penalty coefficient (0.0 = no KL penalty, no ref model needed).
        epsilon: PPO clipping epsilon (lower bound); forwarded as ``epsilon_low``.
        epsilon_high: PPO clipping epsilon (upper bound); defaults to ``epsilon``.
        temperature: Sampling temperature applied to logits inside the kernel.
        loss_type: Aggregation / objective variant. One of Liger's
            ``{'grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo', 'sapo', 'luspo', 'vespo'}``.
            Defaults to ``'grpo'`` (per-sequence mean then batch mean), matching
            the base :class:`~twinkle.loss.grpo.GRPOLoss` fallback aggregation.
        max_completion_length: Required by ``loss_type='dr_grpo'``; ignored otherwise.
        importance_sampling_level: ``'token'`` or ``'sequence'`` (GSPO-style).
        sapo_temperature_pos / sapo_temperature_neg: Soft-gate temperatures for
            ``loss_type='sapo'``.
        compiled: Whether Liger torch-compiles the fused kernel (default False,
            matching swift's ``_prepare_liger_loss``).
        ignore_index: Label id treated as padding (excluded from the objective).
    """

    # Keep outputs['logits'] (hidden states under the fused-CE patch); skip logps.
    require_logits = True
    require_logps = False

    def __init__(self,
                 beta: float = 0.0,
                 epsilon: float = 0.2,
                 epsilon_high: Optional[float] = None,
                 temperature: float = 1.0,
                 loss_type: str = 'grpo',
                 max_completion_length: Optional[int] = None,
                 importance_sampling_level: str = 'token',
                 sapo_temperature_pos: float = 1.0,
                 sapo_temperature_neg: float = 1.05,
                 compiled: bool = False,
                 ignore_index: int = -100,
                 **kwargs):
        super().__init__(epsilon=epsilon, epsilon_high=epsilon_high, beta=beta, ignore_index=ignore_index, **kwargs)
        self.temperature = temperature
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.importance_sampling_level = importance_sampling_level
        # Defer Liger construction to the first fused call so a missing
        # liger_kernel degrades to the pure-torch fallback instead of erroring
        # at set_loss() time.
        self._liger_kwargs = dict(
            beta=beta,
            compiled=compiled,
            use_ref_model=(beta != 0.0),
            epsilon_low=self.epsilon,
            epsilon_high=self.epsilon_high,
            temperature=temperature,
            loss_type=loss_type,
            max_completion_length=max_completion_length,
            importance_sampling_level=importance_sampling_level,
            sapo_temperature_pos=sapo_temperature_pos,
            sapo_temperature_neg=sapo_temperature_neg,
        )
        self._liger = None
        self._fused_broken = False
        self._warned = False

    def _ensure_liger(self):
        if self._liger is None:
            self._liger = _get_liger_module()(**self._liger_kwargs)
        return self._liger

    @staticmethod
    def _full(tensor):
        """Gather a sharded DTensor to a full local tensor; pass others through."""
        if tensor is not None and hasattr(tensor, 'full_tensor'):
            return tensor.full_tensor()
        return tensor

    def _materialise_logits(self, outputs, lm_head):
        """Materialise real logits from the stashed lm_head weight (NOT its
        forward, which is identity under the patch) so the pure-torch fallback
        can run. Clears ``lm_head`` to prevent re-triggering the fused path."""
        import torch.nn.functional as F
        hidden = outputs['logits']
        weight = self._full(lm_head.weight)
        bias = self._full(getattr(lm_head, 'bias', None))
        logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), weight, bias=bias)
        out = dict(outputs)
        out['logits'] = logits.view(*hidden.shape[:-1], -1)
        out['lm_head'] = None
        return out

    def _fused_call(self, inputs, outputs, lm_head, *, old_logps, ref_logps, advantages, **kwargs):
        import torch
        labels = inputs['labels']
        if not torch.is_tensor(labels):
            labels = torch.as_tensor(labels)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)

        hidden = outputs['logits']  # [B, T, H] under TransformersFusedCEPatch
        if hidden.shape[1] != labels.shape[1]:
            # some mllm prepend image tokens to the hidden states; keep the tail
            hidden = hidden[:, -labels.shape[1]:]
        device = hidden.device
        labels = labels.to(device)

        loss_mask = (labels != self.ignore_index)
        # -100 is not a valid vocab id; map masked positions to 0 (excluded by mask).
        selected_token_ids = labels.clamp(min=0)
        attention_mask = loss_mask.to(torch.int32)

        # advantages -> per-sequence [B] (Liger expects one advantage per sequence).
        adv = self._pad_and_align_to_batch(advantages, loss_mask, device, hidden.dtype)
        adv_seq = (adv * loss_mask).sum(-1) / loss_mask.sum(-1).clamp(min=1.0)

        old_per_token_logps = None
        if old_logps is not None:
            old_per_token_logps = self._pad_and_align_to_batch(old_logps, loss_mask, device, hidden.dtype)
        ref_per_token_logps = None
        if ref_logps is not None:
            ref_per_token_logps = self._pad_and_align_to_batch(ref_logps, loss_mask, device, hidden.dtype)

        weight = self._full(lm_head.weight)
        bias = self._full(getattr(lm_head, 'bias', None))

        liger = self._ensure_liger()
        result = liger(
            hidden,
            weight,
            selected_token_ids,
            attention_mask,
            adv_seq,
            bias=bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
        )
        loss = result[0] if isinstance(result, (tuple, list)) else result
        return LossOutput(loss=loss, num_tokens=0)

    def __call__(
        self,
        inputs,
        outputs,
        *,
        old_logps: Optional[Union['np.ndarray', List]] = None,
        ref_logps=None,
        advantages=None,
        **kwargs,
    ):
        lm_head = outputs.get('lm_head')
        have_head = lm_head is not None and getattr(lm_head, 'weight', None) is not None

        if have_head and not self._fused_broken and advantages is not None:
            try:
                return self._fused_call(
                    inputs, outputs, lm_head, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages, **kwargs)
            except Exception as e:  # defensive, device-agnostic
                if not self._warned:
                    self._warned = True
                    logger.warning(
                        '[LigerFusedLinearGRPOLoss] fused kernel raised %r; falling back to the standard '
                        'GRPO objective (device-agnostic defensive fallback). Subsequent calls skip the fused '
                        'path. This is expected when liger_kernel is absent or the device backend does not '
                        'support the shape.', e)
                self._fused_broken = True

        # ── Fallback: pure-torch GRPO family loss ──
        # Under the fused-CE patch outputs['logits'] holds hidden states, so
        # materialise real logits before delegating to the base implementation.
        if have_head:
            outputs = self._materialise_logits(outputs, lm_head)
        return super().__call__(
            inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages, **kwargs)
