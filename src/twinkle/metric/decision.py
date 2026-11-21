# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Metric


class DecisionAccuracy(Metric):
    """Count questions and jointly correct records, rather than language tokens."""

    def __init__(self, device_mesh=None, process_group=None, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.reset()

    def accumulate(self, inputs, outputs, **kwargs):
        logits, target = outputs.get('logits'), inputs.get('labels')
        if logits is None or target is None:
            return
        mask = outputs['option_mask']
        correct = logits.masked_fill(~mask, -float('inf')).argmax(-1) == target.argmax(-1)
        counts = inputs['decision_meta']['counts']
        if sum(counts) != correct.numel():
            raise ValueError('Question counts do not cover decision predictions')
        self.correct += int(correct.sum())
        self.questions += correct.numel()
        offset = 0
        for count in counts:
            self.record_correct += int(correct[offset:offset + count].all())
            self.records += 1
            offset += count

    def reset(self):
        self.correct = self.questions = self.record_correct = self.records = 0

    def calculate(self):
        rows = self.gather_results([{
            'correct': self.correct,
            'questions': self.questions,
            'record_correct': self.record_correct,
            'records': self.records
        }])
        total = {key: sum(row[key] for row in rows) for key in rows[0]}
        self.reset()
        if not total['questions']:
            return {}
        return {
            **total, 'acc': total['correct'] / total['questions'],
            'record_acc': total['record_correct'] / total['records']
        }
