# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch a HF transformers causal LM to skip the lm_head matmul so the loss
layer can fuse it with cross-entropy (Liger Fused Linear-CE).

Two mutations applied to the model (both reverted by ``unpatch``):

1. ``lm_head.forward`` is replaced with identity, so the wrapped model returns
   the final hidden states under ``output.logits`` instead of the full
   ``(B, T, V)`` logits tensor. This skips the vocab projection GEMM and the
   materialisation of the logits tensor, which is the memory + bandwidth win
   that fused-linear-CE unlocks.
2. A forward hook on the lm-head-bearing submodule preserves the full HF
   ``ModelOutput`` (so MoE ``aux_loss`` / ``router_logits`` / ``past_key_values``
   survive) and additionally stashes a reference to the lm_head module under
   ``outputs['lm_head']`` so the loss can read ``.weight`` without holding the
   model.

This patch is deliberately hardware-agnostic: it contains no device probes
(``torch.cuda`` / ``Platform.is_npu`` / ``infer_device``). Device selection of
the fused-CE kernel is handled by Liger's own ``infer_device`` /
``select_impl`` dispatch (CUDA-Triton vs Ascend backend), exactly like the
``liger_builtin`` bundle emits bare impls. See ``liger.py`` docstring.

The patch is applied per-forward via ``apply_context`` (see
``_resolve_task_context`` with ``task='fused_lm_ce'``), so it composes with
Twinkle's SP / routing-replay / FSDP2 / processor-postprocess orchestration
without touching the forward body. Under FSDP2 the lm_head params are unsharded
by FSDP's pre-forward hook before the (identity) forward runs, so
``lm_head.weight`` read by the loss is the full weight.

FSDP2 collective ordering: under accelerate-FSDP2 the first fused-CE forward
lazily runs ``fully_shard`` init (collectives), and the loss's ad-hoc
``lm_head.weight.full_tensor()`` gather (issued after the identity forward) can
race with them if ranks enter the first fused-CE step slightly desynchronised
(asymmetric setup work) — the two collective streams enqueue in different orders
across ranks and the first backward deadlocks (rank A in autograd backward,
rank B already in the optimizer's DTensor redistribution). Cookbooks using
``task='fused_lm_ce'`` under accelerate-FSDP2 should issue a one-time
``dist.barrier()`` + device ``synchronize()`` before the first fused-CE training
step (see ``liger_fused_linear_cross_entropy.py`` and
``cookbook/transformers/fsdp2.py``; ``liger_bench.py`` already does a per-step
``_synchronize()``). This patch itself stays device-free (no ``torch.cuda`` /
``Platform.is_npu`` probes — enforced by ``test_patch_is_device_free``).

Composing with ``LigerFusedLinearCrossEntropyLoss`` (see ``twinkle.loss``):
the loss dispatches on the presence of ``outputs['lm_head']`` — present =>
fused path (hidden_states + lm_head.weight); absent => standard CE from
materialised logits. So eval forwards that omit ``task='fused_lm_ce'`` still
work with the same loss instance.
"""
from types import MethodType
from typing import TYPE_CHECKING, Optional

from twinkle.patch import Patch

if TYPE_CHECKING:
    import torch

# Reuse the lm-head discovery list from the embedding patch verbatim — same
# families (Qwen / Llama / Phi3 with 'lm_head', GPT-NeoX with 'embed_out', ...).
from twinkle.patch.transformers_emb import get_lm_head_model

_LM_HEADS = ['lm_head', 'embed_out', 'output_layer', 'output']


def _identity_forward(self, hidden_states):
    """Return hidden states unchanged; the vocab GEMM is deferred to the loss."""
    return hidden_states


class TransformersFusedCEPatch(Patch):
    """Skip the lm_head matmul so a fused-linear-CE loss can take over.

    Reversible via ``unpatch``. Device-agnostic: no ``cuda``/``npu`` branching.
    """

    def __call__(self, module, *args, **kwargs):
        from torch.nn import Module
        lm_head_model = get_lm_head_model(module, lm_heads=_LM_HEADS)

        head: Optional[Module] = None
        for name in _LM_HEADS:
            if hasattr(lm_head_model, name):
                head = getattr(lm_head_model, name)
                break
        assert head is not None, 'Cannot find the proper lm_head name'

        # Save originals BEFORE mutation so unpatch can restore them verbatim.
        self._head = head
        self._origin_forward = head.forward
        head.forward = MethodType(_identity_forward, head)

        # Closure captures `head` so the hook can stash the module reference
        # without polluting the lm_head_model with attributes.
        def _stash_lm_head_hook(module, args, kwargs, output):
            # Preserve the full HF ModelOutput (aux_loss / router_logits /
            # past_key_values / ...); only add the lm_head module reference.
            # output.logits is hidden_states here (identity forward).
            if hasattr(output, 'items'):
                out = dict(output.items())
            elif isinstance(output, dict):
                out = dict(output)
            elif isinstance(output, (tuple, list)):
                # HF non-return_dict path: (logits, ...) — keep logits only.
                out = {'logits': output[0] if len(output) else None}
            else:
                out = {'logits': output}
            out['lm_head'] = head
            return out

        self._hook_handle = lm_head_model.register_forward_hook(_stash_lm_head_hook, with_kwargs=True)
        return module

    def unpatch(self, module, *args, **kwargs):
        handle = getattr(self, '_hook_handle', None)
        if handle is not None:
            handle.remove()
            self._hook_handle = None

        head = getattr(self, '_head', None)
        origin = getattr(self, '_origin_forward', None)
        if head is not None and origin is not None:
            head.forward = origin
            self._origin_forward = None
            self._head = None
        return module
