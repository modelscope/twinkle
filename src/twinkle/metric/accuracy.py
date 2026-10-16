# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Union

from ..data_format import InputFeature, ModelOutput
from .base import Metric


class Accuracy(Metric):
    """The accuracy metric.

    Args:
        device_mesh: The device mesh
        process_group: The process group to collect data from
        strategy: 'token' scores every trainable position independently; 'seq' scores a row once and
            counts it correct only when EVERY trainable token in it is correct (mirrors legacy
            swift/metrics/acc.py compute_acc). Both need materialized logits, so neither fires on a
            fused-linear-CE forward (outputs carry no 'logits') -- accumulate() returns early there.
    """

    def __init__(self, device_mesh, process_group, strategy: str = 'token', **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        if strategy not in ('token', 'seq'):
            raise ValueError(f"Accuracy strategy must be 'token' or 'seq', got {strategy!r}.")
        self.strategy = strategy
        self.total_correct = 0
        self.total_count = 0

    def accumulate(self, inputs: Union[InputFeature, List[InputFeature]], outputs: ModelOutput, **kwargs):
        assert not isinstance(inputs, list), 'Accuracy does not support list InputFeature yet.'
        labels = inputs.get('labels')
        logits = outputs.get('logits')
        if labels is None or logits is None:
            # Pairwise objectives (a reward model ranks chosen against rejected) and pooled heads
            # carry no per-token labels, so there is no token accuracy to accumulate. A fused-linear-CE
            # forward also lands here: it never materializes the full logits, so there is nothing to argmax.
            return
        output_token_ids = logits.argmax(dim=-1)
        mask = inputs.get('completion_mask')
        if mask is not None:
            mask = mask.bool()

        # Align labels/mask with truncated logits to avoid shape mismatches.
        if labels.shape != output_token_ids.shape:
            labels = labels[..., -output_token_ids.shape[-1]:]
            if mask is not None and mask.shape != output_token_ids.shape:
                mask = mask[..., -output_token_ids.shape[-1]:]

        # Same scope the loss uses: a position counts only when it is scored *and* it is the
        # policy's own completion, otherwise -100 positions inflate the denominator.
        trainable = labels != -100
        mask = trainable if mask is None else trainable & mask

        correct_mask = (output_token_ids == labels) & mask

        if self.strategy == 'seq':
            # Reduce over the sequence axis: a row is one sample, correct only if all its trainable
            # tokens match. Rows with no trainable token are dropped from both numerator and denominator.
            if correct_mask.dim() == 1:
                correct_mask = correct_mask.unsqueeze(0)
                mask = mask.unsqueeze(0)
            per_seq_total = mask.sum(dim=-1)
            per_seq_correct = correct_mask.sum(dim=-1)
            scored = per_seq_total > 0
            local_correct = int((per_seq_correct[scored] == per_seq_total[scored]).sum().item())
            local_total = int(scored.sum().item())
        else:
            local_correct = int(correct_mask.sum().item())
            local_total = int(mask.sum().item())

        self.total_correct += local_correct
        self.total_count += local_total

    def reset(self):
        self.total_correct = 0
        self.total_count = 0

    def calculate(self):
        local_results = [{'correct': self.total_correct, 'total': self.total_count}]

        all_results = self.gather_results(local_results)

        total_correct = sum(r['correct'] for r in all_results)
        total_count = sum(r['total'] for r in all_results)
        self.reset()
        if total_count > 0:
            accuracy = total_correct / total_count
            unit = 'sequences' if self.strategy == 'seq' else 'tokens'
            return {
                'accuracy': f'{accuracy:.2f}',
                f'correct_{unit}': total_correct,
                f'total_{unit}': total_count,
            }
        else:
            return {}
