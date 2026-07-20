# Copyright (c) ModelScope Contributors. All rights reserved.
"""Liger Fused Linear Cross-Entropy loss.

Fuses the final ``lm_head`` matmul with cross-entropy so the full
``(B, T, V)`` logits tensor is never materialised — the memory + bandwidth
win for large-vocab models like Qwen3.5-35B-A3B (V ~= 151k).

Contract with the model forward
--------------------------------
This loss sets ``require_logits = True`` / ``require_logps = False`` so
``TransformersModel.forward`` keeps ``outputs['logits']`` and skips the
``selective_log_softmax`` pass. Under ``TransformersFusedCEPatch``
(applied via ``task='fused_lm_ce'``) the lm_head is replaced by identity, so
``outputs['logits']`` is actually the last hidden state ``[B, T, H]``, and the
patch stashes the lm_head module under ``outputs['lm_head']``. This loss then
reads both and calls Liger's fused kernel:

    LigerFusedLinearCrossEntropy(lin_weight=lm_head.weight,
                                 _input=hidden.view(-1, H),
                                 target=labels.view(-1))

If ``outputs['lm_head']`` is absent (e.g. an eval forward that did not pass
``task='fused_lm_ce'``), the loss transparently falls back to the standard CE
computed from materialised logits — same value, just without the fusion. This
makes the loss safe to register once and use across mixed train/eval task
contexts.

No token shift here
-------------------
Twinkle templates pre-shift labels via ``_roll_labels`` (``template/base.py``),
so ``labels[i]`` is already the target for the prediction made at position
``i``. Liger's own ``LigerForCausalLMLoss`` shifts because HF passes
unshifted labels — we must NOT re-shift, so we call Liger's
``LigerFusedLinearCrossEntropyLoss`` module (which does not shift) directly
rather than ``LigerForCausalLMLoss`` (which does).

Hardware
--------
Device selection is delegated to Liger's ``infer_device`` / ``select_impl``
dispatch: this file contains no ``torch.cuda`` / ``Platform.is_npu`` probes.
The same loss class runs on CUDA (Triton kernel) and Ascend NPU (the
``backends/_ascend`` fused-linear-CE backend), mirroring the bare-impl
philosophy of ``liger_builtin``.

FSDP2 collective ordering
------------------------
Under accelerate-FSDP2 the first model forward lazily runs ``fully_shard`` init
(collectives), and this loss issues an ad-hoc ``lm_head.weight.full_tensor()``
gather after the (identity) lm_head forward. If ranks enter the first fused-CE
step slightly desynchronised (asymmetric setup work), the two collective
streams enqueue in different orders across ranks and the first backward
deadlocks — e.g. rank A stuck in autograd backward while rank B has already
reached the optimizer's DTensor redistribution (Muon ``_zeropower_via_newtonschulz``).
This is a cookbook-level timing requirement, not a loss bug: the loss's
``full_tensor()`` gather is rank-symmetric. The fix is a one-time
``dist.barrier()`` + device ``synchronize()`` before the first fused-CE
training step — ``liger_bench.py`` already does a per-step ``_synchronize()``
and ``cookbook/transformers/fsdp2.py`` does a single pre-loop drain (both
verified sufficient). This loss stays device-agnostic: it does NOT issue the
drain itself (no ``torch.cuda`` / ``Platform.is_npu`` probes); the cookbook owns
it.
"""
from __future__ import annotations

import torch.nn.functional as F
from typing import Optional

from twinkle import get_logger
from twinkle.data_format import LossOutput, ModelOutput
from .base import Loss

logger = get_logger()

# Lazily import Liger so the module loads even when liger_kernel is absent;
# the loss then raises a clear error only if actually constructed/used.
_LigerFLCEModule = None


def _get_liger_module():
    global _LigerFLCEModule
    if _LigerFLCEModule is not None:
        return _LigerFLCEModule
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss as _Mod  # noqa: F401
    _LigerFLCEModule = _Mod
    return _LigerFLCEModule


class LigerFusedLinearCrossEntropyLoss(Loss):
    """Fused lm_head + cross-entropy loss (Liger kernel).

    Args:
        ignore_index: Label id treated as padding (excluded from loss);
            forwarded to Liger which masks it inside the fused kernel.
        reduction: 'sum' (default, matching Twinkle's CrossEntropyLoss model
            default) or 'mean'. With 'sum' the loss is the per-token CE summed
            over non-ignored tokens; ``calculate_loss`` accumulates this and the
            metric divides by ``num_tokens`` to recover the per-token value, and
            the gradient is divided by ``num_tokens`` to give a per-token-mean
            gradient — identical to ``CrossEntropyLoss(reduction='sum')``.
        label_smoothing: 0.0 (default) for hard labels; forwarded to Liger.
        softcap: Optional logit softcapping value (e.g. Gemma); forwarded to
            Liger. None disables it.

    Defensive fallback (device-agnostic)
    -------------------------------------
    The Liger fused kernel is wrapped in a try/except. If it raises for any
    reason — an unsupported shape on the Ascend backend, a Triton compile
    failure on CUDA, a version mismatch, an OOM inside the fused kernel — the
    loss logs one warning, marks the fused path broken (so subsequent calls skip
    straight to the fallback without re-throwing), materialises the logits via
    ``F.linear(hidden, lm_head.weight)`` (NOT ``lm_head.forward`` — that is
    identity under the patch), and computes standard cross-entropy on them. So
    ``--enable-liger`` is always safe on both NPU and CUDA: best case the fused
    kernel saves the logits memory, worst case it degrades transparently to the
    standard CE path (same numerics, just without the memory win). No
    ``torch.cuda`` / ``Platform.is_npu`` probes anywhere — the guard is purely
    exception-based.
    """

    # Keep outputs['logits'] (hidden states under the patch); skip logps.
    require_logits = True
    require_logps = False

    def __init__(self,
                 ignore_index: int = -100,
                 reduction: str = 'sum',
                 label_smoothing: float = 0.0,
                 softcap: float | None = None,
                 **kwargs):
        super().__init__()
        assert reduction in ('mean', 'sum'), f"reduction must be 'mean' or 'sum', got {reduction!r}"
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.softcap = softcap
        self._liger = _get_liger_module()(
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
            softcap=softcap,
        )
        self._fused_broken = False
        self._warned = False

    def _materialise_ce(self, inputs, outputs, lm_head, hidden, **kwargs):
        """Materialise logits from the stashed lm_head weight (NOT its forward,
        which is identity under the patch) and run standard cross-entropy."""
        from .cross_entropy import CrossEntropyLoss
        weight = lm_head.weight
        if hasattr(weight, 'full_tensor'):
            weight = weight.full_tensor()
        bias = getattr(lm_head, 'bias', None)
        if bias is not None and hasattr(bias, 'full_tensor'):
            bias = bias.full_tensor()
        logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), weight, bias=bias)
        out = dict(outputs)
        out['logits'] = logits.view(*hidden.shape[:-1], -1)
        return CrossEntropyLoss(ignore_index=self.ignore_index, reduction=self.reduction)(inputs, out, **kwargs)

    def __call__(self, inputs, outputs: ModelOutput, **kwargs):
        labels = inputs['labels']
        lm_head = outputs.get('lm_head')
        have_head = lm_head is not None and getattr(lm_head, 'weight', None) is not None

        if have_head and not self._fused_broken:
            # ── Fused path: hidden states + lm_head.weight ──
            hidden = outputs['logits']  # [B, T, H] under TransformersFusedCEPatch
            hidden_size = hidden.shape[-1]
            hidden_flat = hidden.reshape(-1, hidden_size)
            labels_flat = labels.reshape(-1).to(hidden_flat.device)
            # Under EP/FSDP2 the lm_head weight may be a sharded DTensor: the
            # identity-forward patch (TransformersFusedCEPatch) skips the normal
            # lm_head forward that would auto-unshard it, so materialise the
            # full weight here. ``full_tensor()`` is a collective gather; under
            # LoRA the lm_head is frozen so only the forward gather is needed
            # (grad flows to hidden, connecting back to the LoRA backbone). The
            # duck-typed ``hasattr`` check keeps this device/strategy-agnostic —
            # regular tensors (no EP/FSDP) pass through unchanged.
            weight = lm_head.weight
            if hasattr(weight, 'full_tensor'):
                weight = weight.full_tensor()
            try:
                loss = self._liger(weight, hidden_flat, labels_flat)
            except Exception as e:  # noqa: BLE001 — defensive, device-agnostic
                if not self._warned:
                    self._warned = True
                    logger.warning(
                        '[LigerFusedLinearCrossEntropyLoss] fused kernel raised %r; '
                        'falling back to materialised cross-entropy (device-agnostic defensive '
                        'fallback). Subsequent calls skip the fused path. This is expected when '
                        'the device backend does not support the shape (e.g. Ascend chunked path '
                        'on some shapes, CUDA Triton compile failure).', e)
                self._fused_broken = True
                return self._materialise_ce(inputs, outputs, lm_head, hidden, **kwargs)
            num_tokens = (labels != self.ignore_index).to(loss.device).sum().clamp(min=1)
            return LossOutput(loss=loss, num_tokens=num_tokens)

        if have_head:
            # fused path broken on a previous call -> materialise via the stashed weight
            return self._materialise_ce(inputs, outputs, lm_head, outputs['logits'], **kwargs)

        # ── Fallback: standard CE from materialised logits (eval w/o patch) ──
        # No `lm_head` in outputs means the forward did not run under
        # TransformersFusedCEPatch (e.g. an eval pass); outputs['logits'] then
        # already holds real logits. label_smoothing is a fused-kernel-only knob
        # here; the eval fallback uses hard labels (the common eval case).
        from .cross_entropy import CrossEntropyLoss
        return CrossEntropyLoss(ignore_index=self.ignore_index, reduction=self.reduction)(inputs, outputs, **kwargs)
