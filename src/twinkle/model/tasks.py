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

Readout tasks come in two head flavours, split by WHO can build the head:
the decision families' heads are model-family artifacts (weights and calibration state
shipped in the model directory, rebuilt worker-side from a shipped loader NAME), while
the seq_cls family's head is a standard Linear any causal base can carry
(``twinkle.module.head.SeqClsHead``, built by ``build_head``). The head LIFECYCLE rules
(where a head lives, how it routes per adapter, how peft's freeze sweep is undone) are
task-independent and live once on the base class below.
"""

import contextlib
from typing import ClassVar, Optional

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

    # --- component (head) lifecycle; readout tasks carry one ------------------------------
    # Task-independent rules, inherited as-is by every readout task: only WHICH head a
    # task builds differs (``build_head`` / the decision families' loader replay).

    def head_of(self, model):
        """The model's single attached head for this task. Readout-task models resolve
        (their loader or adapter attached one); the raise below is the loud failure when
        a readout task runs on a model that carries neither readout kind."""
        head = getattr(model, 'task_head', None)
        if head is None:
            raise ValueError(
                f"task {self.task_name!r} declares readout kinds {list(self.row().readout)} but the "
                'model carries no task_head: attach one (the task loader at build time, '
                'task_head_loader, or a multi-LoRA head slot); the transformers-only '
                'task_forward families are rejected by their loader at build time.')
        return head

    def head_for(self, model, adapter_name=None):
        """The head for ONE call: the per-adapter dictionary first (multi-head models), the
        single attached head as fallback. A model carrying per-adapter heads resolves them
        STRICTLY -- the dictionary counts even when empty, and an adapter without its own
        head raises instead of silently borrowing one (two adapters of the same head task
        would otherwise pollute each other's gradients through a shared head)."""
        heads = getattr(model, 'task_heads', None)
        if heads is not None:
            if adapter_name not in heads:
                raise ValueError(
                    f'adapter {adapter_name!r} carries no {self.task_name} task_head; bind one when '
                    'the adapter is added (task= on add_adapter_to_model, or task_head_loader) or '
                    'train it as an adapter of another task.')
            return heads[adapter_name]
        return self.head_of(model)

    def carries_task_head(self, model) -> bool:
        """Whether THIS model carries the task's attachable head (per-adapter dictionary
        or single head). The dictionary counts even when empty: a multi-LoRA host built
        for a head task has no in-model fallback, so its readout must resolve through it
        and fail loudly instead of silently running a legacy path."""
        if isinstance(model, (list, tuple)):  # megatron chunks
            return any(self.carries_task_head(chunk) for chunk in model)
        return (getattr(model, 'task_heads', None) is not None
                or getattr(model, 'task_head', None) is not None)

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

    def place_megatron_head(self, head, chunk):
        """Move a freshly built head onto the device of the mcore chunk it rides, matching
        the backbone's dtype unless the head declares ``keep_fp32`` (a plain fp32 Linear over
        bf16 hidden states raises on the readout matmul; the keep_fp32 heads upcast inputs
        themselves)."""
        from twinkle import Platform
        core = chunk
        while hasattr(core, 'module'):  # down past DDP / Float16Module, same as attach
            core = core.module
        if getattr(head, 'keep_fp32', False):
            return head.to(Platform.get_local_device())
        backbone_dtype = next((p.dtype for p in core.parameters()), None)
        if backbone_dtype is None:
            return head.to(Platform.get_local_device())
        return head.to(device=Platform.get_local_device(), dtype=backbone_dtype)

    def attach_megatron(self, chunk, head) -> None:
        """Attach to the mcore model itself, not the MegatronModel wrapper: only parameters
        living on the model chunks enter the mcore optimizer and ride the DDP /
        mixed-precision wrap."""
        head = self.place_megatron_head(head, chunk)
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

    def build_head(self, config, num_labels=None):
        """Build this task's STANDARD head. Overridden by the seq_cls family; the decision
        families' heads are family artifacts rebuilt from a shipped loader NAME, so this
        base raises for them."""
        raise NotImplementedError(
            f'{type(self).__name__} has no standard head; the decision families build theirs from '
            'the family loader (task_head_loader).')


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
    """The classification-head tasks (seq_cls / reranker / value), dual-path per backend:
    a model carrying an ATTACHED head (single, per-adapter, or a multi-LoRA slot) reads it
    out through the forward-layer readout (no patch on transformers; an identity output
    layer on megatron so the readout branch sees the last hidden states); a model variant
    that BUILDS the head itself (HF ``*ForSequenceClassification`` / mcore-bridge task_type)
    keeps its legacy path -- per-token surfacing on transformers, bridge-head pooling in
    forward_step."""

    task_name = 'seq_cls'
    #: labels the head defaults to when the caller passes none (seq_cls demands an explicit
    #: count; reranker / value are single-score heads).
    default_num_labels: ClassVar[Optional[int]] = None
    #: mirrors ``TransformersValuePatch``: value squeezes per-token [B, T]; seq_cls keeps 3D.
    _squeeze_single: ClassVar[bool] = False

    def build_head(self, config, num_labels=None):
        from twinkle.module.head import SeqClsHead
        if num_labels is None:
            num_labels = self.default_num_labels
        if num_labels is None:
            raise ValueError(
                f'task {self.task_name!r} needs num_labels= to build its head (the config carries '
                'no label count).')
        hidden_size = getattr(config, 'hidden_size', None)
        if hidden_size is None:
            # Composite checkpoints (Qwen3.5 and other hybrid / VL families) keep the language
            # backbone's width on the nested text_config; the head rides that backbone.
            text_config = getattr(config, 'text_config', None)
            hidden_size = getattr(text_config, 'hidden_size', None) if text_config is not None else None
        if hidden_size is None:
            raise ValueError(f'cannot build a {self.task_name} head: the config has no hidden_size.')
        return SeqClsHead(hidden_size, num_labels, squeeze_single=self._squeeze_single)

    def transformers_context(self, model, template=None):
        if self.carries_task_head(model):
            # The readout hook drives the attached head; no patch.
            return contextlib.nullcontext()
        # Legacy path: the model's own *ForSequenceClassification score head -- skip HF's
        # internal pooling; surface per-token [B, T, num_labels] so the processor's
        # _pool_last_valid_token handles packed/non-packed uniformly.
        from twinkle.patch import apply_context
        from twinkle.patch.transformers_value import TransformersValuePatch
        return apply_context(model, TransformersValuePatch(squeeze_single=self._squeeze_single))

    def megatron_context(self, model):
        if self.carries_task_head(model):
            # Swap the output layer for an identity so the forward_step readout branch sees
            # the last hidden states [b, s, h] instead of vocab logits.
            from twinkle.patch import apply_context
            from twinkle.patch.megatron_emb import MegatronHiddenPassthroughPatch
            return apply_context(model, MegatronHiddenPassthroughPatch())
        # Legacy: mcore-bridge already built the head into the forward; pooling in forward_step.
        return contextlib.nullcontext()


class Reranker(SeqCls):
    task_name = 'reranker'
    default_num_labels = 1


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


class Value(SeqCls):
    """PPO critic: a num_labels=1 head whose PER-TOKEN output rides the logits channel
    ([b, s]; PPOValueLoss masks to response tokens)."""

    task_name = 'value'
    default_num_labels = 1
    _squeeze_single = True


class Decision(TaskSpec):
    """The decision (System-1 scoring) readout task: no patch on either backend -- the
    loader installed a load-time custom forward (``task_forward``: the transformers-only
    branch families) or an attachable head (``task_head``), and the forward-layer readout
    (the one-shot hook / the forward_step readout branch) drives it. Its heads are family
    artifacts rebuilt from the dev family loader's NAME (see ``twinkle/module/head.py``'s docstring
    for the split against the standard heads), so this class adds nothing over the base
    head lifecycle."""

    task_name = 'decision'
    transformers_context = _no_patch
    megatron_context = _no_patch


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
