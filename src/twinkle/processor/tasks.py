# Copyright (c) ModelScope Contributors. All rights reserved.
"""The forward-task registry: each task's processor-side semantics in one row.

A task differs from its siblings by data, not by type: whether the forward output carries
vocab log-softmax logps (``requires_logps``), whether labels and outputs are token/sequence
aligned (``token_level_labels``; False means the forward already reduced each sample, so
packing/unpacking and sequence-shaped merging do not apply), and the optional
``InputProcessor`` reduction method that must run after the forward (``postprocess``).

The former per-task string sets scattered across ``_NON_LOGPS_TASKS``,
``InputProcessor.token_level_labels``, ``unpack_packed_sequences`` and
``merge_micro_batch_outputs`` were hand-synced copies of these two booleans; they all read
this table now. This table is the task's DATA contract only (it must stay importable from
the processor layer); the model-layer BEHAVIOUR -- each backend's per-forward patch context
and the readout tasks' head lifecycle -- lives in ``twinkle.model.tasks`` (one TaskSpec
behaviour class per row; the two registries cover the same task set and check so at lookup).
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class TaskSpec:
    """The processor-side contract of one forward task."""

    #: False when the forward output is not a vocab distribution (classification/score heads,
    #: pooled embeddings, per-question decision scores), so logps gathering must be skipped.
    requires_logps: bool = True
    #: False when labels are question/record-level instead of token-level, and the forward
    #: output is already reduced per sample (no sequence axis to pack, unpack or merge).
    token_level_labels: bool = True
    #: ``InputProcessor`` method name for the task's post-forward reduction; None = none.
    postprocess: Optional[str] = None
    #: Readout hooks this task's models carry, so the forward never goes through the legacy
    #: per-forward patch dispatch: 'task_head' = an attachable head the backend feeds the
    #: hidden states / lm logits of one normal forward (via a one-shot hook, inside the
    #: strategy wrap so DDP's reducer sees the head parameters); 'task_forward' = a load-time
    #: custom execution that replaced the module forward. Which one a MODEL carries is a
    #: model (family) property -- the row lists the kinds the task allows. Empty = legacy path.
    readout: Tuple[str, ...] = ()


TASKS = {
    'causal_lm': TaskSpec(),
    'fused_lm_ce': TaskSpec(),
    'value': TaskSpec(requires_logps=False),
    'embedding': TaskSpec(requires_logps=False, token_level_labels=False, postprocess='_postprocess_embedding'),
    'seq_cls': TaskSpec(requires_logps=False, token_level_labels=False, postprocess='_postprocess_seq_cls'),
    'reranker': TaskSpec(requires_logps=False, token_level_labels=False, postprocess='_postprocess_seq_cls'),
    'generative_reranker': TaskSpec(
        requires_logps=False, token_level_labels=False, postprocess='_postprocess_generative_reranker'
    ),
    # The decision scoring loaders attach either a plain head ('task_head': Jev / Clef /
    # StartLux / generic / Laya) or a load-time custom forward ('task_forward': the
    # branch-isolation families OmniJev / Kev, plus Thx02 which reads the backbone directly).
    # Both are dispatched by the backends via this row's readout kinds, never by task name.
    'decision': TaskSpec(requires_logps=False, token_level_labels=False, readout=('task_head', 'task_forward')),
}


def get_task_spec(task: Optional[str]) -> TaskSpec:
    """Look up a task's spec; ``None`` means the ``causal_lm`` default forward callers use."""
    spec = TASKS.get(task or 'causal_lm')
    if spec is None:
        raise ValueError(f'Unknown task={task!r}; expected one of: {", ".join(sorted(TASKS))}.')
    return spec
