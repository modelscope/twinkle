# Copyright (c) ModelScope Contributors. All rights reserved.
"""The model-side behaviour of each forward task: one class per task.

Layering: the cross-layer DATA contract (``requires_logps`` / ``token_level_labels`` /
``postprocess`` / ``readout``) stays in ``twinkle.processor.tasks`` -- the one table both
the processor and the model layers read; this module adds the model-layer BEHAVIOUR:
the per-forward patch context for each backend, and (for readout tasks) the component
(head) lifecycle. The split is a dependency-direction requirement, not duplication:
processor cannot import model, so the data row lives below and this class reaches down.

Each task states BOTH backends' context explicitly -- a task newly supported by one
backend adds that method override here; the base-class raise is the gate forcing that
explicit decision (admitting a task nowhere would run it down the causal path silently).
Tasks whose registry row carries readout kinds state no patch at all: their loader
installed a load-time custom forward or an attachable head, and the forward-layer
readout (a one-shot hook / the forward_step readout branch) takes over.
"""

import contextlib
from typing import Any, ClassVar, Optional

from twinkle.processor.tasks import TASKS, TaskSpec as TaskRow

__all__ = ['TaskSpec', 'get_task_spec_obj', 'TASK_SPECS']


class TaskSpec:
    """Model-side behaviour of one forward task (see the module docstring for the split
    against ``twinkle.processor.tasks``)."""

    #: registry name; ``TASK_SPECS`` keys must equal ``TASKS`` keys exactly.
    task_name: ClassVar[str]

    @classmethod
    def row(cls) -> TaskRow:
        """The task's data contract from the processor-level registry."""
        return TASKS[cls.task_name]

    # --- per-forward context, one method per backend -------------------------------------
    # Delayed imports inside the overrides: these patch modules carry conditional /
    # heavyweight dependencies (Liger, hardware-specific kernels) and must not be pulled
    # in by importing this module.

    def transformers_context(self, model, template=None):
        """The per-forward Patch context this task needs on the transformers backend."""
        raise NotImplementedError(
            f'{type(self).__name__}.transformers_context is not implemented: a task newly supported by '
            'this backend must state its context here explicitly (a patch, or nullcontext).')

    def megatron_context(self, model):
        """The per-forward Patch context this task needs on the megatron backend."""
        raise NotImplementedError(
            f'{type(self).__name__}.megatron_context is not implemented: a task newly supported by '
            'this backend must state its context here explicitly (a patch, or nullcontext).')

    # --- component (head) lifecycle; readout tasks override --------------------------------

    def head_of(self, model):
        """The task's attached component (head). Only tasks whose registry row declares a
        'task_head' readout kind override this; the base raise means the task carries none."""
        raise NotImplementedError(f'{type(self).__name__} carries no attachable component.')

    def attach(self, model, head, **kwargs) -> None:
        """Install this task's model-side component (head). Default: this task has none."""

    def reactivate(self, model) -> None:
        """Re-enable this task's component for adapter training (peft froze it). Default: no-op."""


def _nullcontext(model, template=None):
    return contextlib.nullcontext()


#: The no-patch context, shared as a staticmethod so subclasses can state it declaratively:
#: ``transformers_context = _no_patch`` / ``megatron_context = _no_patch``.
_no_patch = staticmethod(_nullcontext)


class CausalLM(TaskSpec):
    task_name = 'causal_lm'
    transformers_context = _no_patch
    megatron_context = _no_patch


class SeqCls(TaskSpec):
    task_name = 'seq_cls'

    def transformers_context(self, model, template=None):
        # Skip HF's internal pooling; surface per-token [B, T, num_labels] so the processor's
        # _pool_last_valid_token handles packed/non-packed uniformly.
        from twinkle.patch.transformers_value import TransformersValuePatch
        from twinkle.patch import apply_context
        return apply_context(model, TransformersValuePatch(squeeze_single=False))

    megatron_context = _no_patch  # mcore-bridge already builds the head; pooling in forward_step


class Reranker(SeqCls):
    task_name = 'reranker'


class Embedding(TaskSpec):
    task_name = 'embedding'

    def transformers_context(self, model, template=None):
        # lm_head -> identity + a feature-extraction hook, so downstream pooling can run inside
        # InputProcessor.postprocess_outputs(task='embedding', ...).
        from twinkle.patch import apply_context
        from twinkle.patch.transformers_emb import TransformersEmbeddingPatch
        return apply_context(model, TransformersEmbeddingPatch())

    def megatron_context(self, model):
        # The output_layer -> identity swap (with TP/SP gather), pooling to [n_seqs, hidden].
        from twinkle.patch import apply_context
        from twinkle.patch.megatron_emb import MegatronEmbeddingPatch
        return apply_context(model, MegatronEmbeddingPatch())


class GenerativeReranker(TaskSpec):
    task_name = 'generative_reranker'

    def transformers_context(self, model, template=None):
        from twinkle.patch import apply_context
        from twinkle.patch.transformers_reranker import TransformersGenerativeRerankerPatch
        pos_id, neg_id = _resolve_generative_reranker_token_ids(template)
        return apply_context(model, TransformersGenerativeRerankerPatch(pos_id, neg_id))

    megatron_context = _no_patch  # mcore-bridge already builds the head; pooling in forward_step


class FusedCE(TaskSpec):
    task_name = 'fused_lm_ce'

    def transformers_context(self, model, template=None):
        # Skip the lm_head GEMM so a fused-linear-CE loss can take over (Liger).
        # Device-agnostic: Liger self-dispatches the fused kernel across CUDA/NPU.
        from twinkle.patch import apply_context
        from twinkle.patch.transformers_fused_ce import TransformersFusedCEPatch
        return apply_context(model, TransformersFusedCEPatch())

    # megatron_context stays the base raise: this task is transformers-only.


class Value(TaskSpec):
    task_name = 'value'

    def transformers_context(self, model, template=None):
        # PPO critic: reuse the num_labels=1 seq_cls ``score`` head but expose its PER-TOKEN
        # output (skip the last-token pooling ``*ForSequenceClassification`` does), so forward
        # returns a value V(s_t) at every token in ``logits`` [B, T].
        from twinkle.patch import apply_context
        from twinkle.patch.transformers_value import TransformersValuePatch
        return apply_context(model, TransformersValuePatch())

    megatron_context = _no_patch  # mcore-bridge already builds the head; per-token via logits


class Decision(TaskSpec):
    """The decision (System-1 scoring) readout task: no patch on either backend -- the
    loader installed a load-time custom forward (``task_forward``: the transformers-only
    branch families) or an attachable head (``task_head``), and the forward-layer readout
    (the one-shot hook / the forward_step readout branch) drives it. This class owns that
    head's lifecycle: alignment, attachment (per backend), and re-activation after a peft
    wrap. The head OBJECT stays owned by the model family's dev loader (framework-agnostic
    nn.Module, often rebuilt worker-side from a shipped loader name); only its placement
    rules live here."""

    task_name = 'decision'
    transformers_context = _no_patch
    megatron_context = _no_patch

    def head_of(self, model):
        head = getattr(model, 'task_head', None)
        if head is None:
            raise ValueError(
                "task 'decision' declares readout kinds ['task_head', 'task_forward'] but the model "
                'carries neither: the transformers-only task_forward families (branch isolation / '
                'direct backbone) are rejected by their loader at build time, and a task_head must '
                'be attached by the decision loader at build time.')
        return head

    def head_for(self, model, adapter_name=None):
        """The head for ONE call: the per-adapter dictionary first (multi-head models), the
        single attached head as fallback. A model carrying per-adapter heads resolves them
        STRICTLY -- an adapter without its own head raises instead of silently borrowing one
        (two decision adapters would otherwise pollute each other's gradients through a
        shared head)."""
        heads = getattr(model, 'task_heads', None)
        if heads:
            if adapter_name not in heads:
                raise ValueError(
                    f'adapter {adapter_name!r} carries no decision task_head; bind one when the '
                    'adapter is added (task_head_loader, or task="decision" on the megatron '
                    'multi-tenant host) or train it as a non-decision adapter.')
            return heads[adapter_name]
        return self.head_of(model)

    @staticmethod
    def align_head(model, head):
        """Move the head onto the model's device and dtype policy: heads declaring
        ``keep_fp32`` stay fp32 (their readout upcasts anyway), the rest follow the output
        embeddings' dtype."""
        output_embeddings = model.get_output_embeddings() if hasattr(model, 'get_output_embeddings') else None
        ref_weight = getattr(output_embeddings, 'weight', None)
        if ref_weight is None:
            ref_weight = next((p for p in model.parameters()), None)
        if ref_weight is not None:
            if getattr(head, 'keep_fp32', False):
                head = head.to(device=ref_weight.device)
            else:
                head = head.to(device=ref_weight.device, dtype=ref_weight.dtype)
        return head

    def attach_transformers(self, model, head) -> None:
        """Attach the head for the forward-layer readout dispatch -- no forward patching.
        Dual names: ``task_head`` drives the readout; ``scoring_head`` is the legacy attribute
        the checkpoint / callback / metrics ecosystem reads."""
        head = self.align_head(model, head)
        model.task_head = head
        model.scoring_head = head
        if getattr(head, 'needs_output_embeddings', False):
            head.set_output_embeddings(model.get_output_embeddings())

    def attach_megatron(self, chunk, head) -> None:
        """Attach to the mcore model itself, not the MegatronModel wrapper: only parameters
        living on the model chunks enter the mcore optimizer and ride the DDP /
        mixed-precision wrap."""
        core = chunk
        while hasattr(core, 'module'):  # down to the mcore model, past DDP / Float16Module
            core = core.module
        core.task_head = head
        core.scoring_head = head

    def reactivate(self, model) -> None:
        """Re-enable the head after a peft wrap froze it: peft's mark-only-adapters sweep
        covers the module the head rides on, so an adapter run would otherwise train around
        a frozen head. fp16 storage upcasts to fp32 (bf16 rides -- the readout upcasts)."""
        import torch
        for name, param in model.named_parameters():
            if 'scoring_head' not in name:
                continue
            if not param.requires_grad:
                param.requires_grad_(True)
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)


def _resolve_generative_reranker_token_ids(template):
    """Resolve the positive/negative token ids for a generative reranker from the template tokenizer.

    Token strings default to ``yes`` / ``no`` and are overridable via the same env vars legacy swift
    uses (``GENERATIVE_RERANKER_POSITIVE_TOKEN`` / ``_NEGATIVE_TOKEN``), so a checkpoint trained here
    scores identically under legacy inference.
    """
    import os
    assert template is not None and getattr(template, 'processor', None) is not None, (
        'generative_reranker needs a template with a tokenizer; call set_template before forwarding.')
    tokenizer = template.processor
    pos_token = os.environ.get('GENERATIVE_RERANKER_POSITIVE_TOKEN', 'yes')
    neg_token = os.environ.get('GENERATIVE_RERANKER_NEGATIVE_TOKEN', 'no')
    return tokenizer.convert_tokens_to_ids(pos_token), tokenizer.convert_tokens_to_ids(neg_token)


TASK_SPECS = {cls.task_name: cls() for cls in (
    CausalLM, SeqCls, Reranker, Embedding, GenerativeReranker, FusedCE, Value, Decision)}


def get_task_spec_obj(task: Optional[str]) -> TaskSpec:
    """Look up a task's model-side behaviour object; ``None`` means the ``causal_lm`` default."""
    spec = TASK_SPECS.get(task or 'causal_lm')
    if spec is None:
        raise ValueError(f'Unknown task={task!r}; expected one of: {", ".join(sorted(TASK_SPECS))}.')
    if set(TASK_SPECS) != set(TASKS):
        # The behaviour registry and the data registry must cover the SAME task set: a task
        # with a data row but no behaviour class would fall off this lookup's raise and run
        # silently down whatever default the caller picked.
        raise RuntimeError(
            f'model.tasks.TASK_SPECS ({sorted(TASK_SPECS)}) and processor.tasks.TASKS ({sorted(TASKS)}) '
            'cover different task sets; add the missing TaskSpec subclass.')
    return spec
