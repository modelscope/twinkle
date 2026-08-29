# Copyright (c) ModelScope Contributors. All rights reserved.
"""Text-level generation metrics: score decoded completions against reference answers.

These are for evaluation runs, where what is available is the generated *text*. That makes them
different in kind from :class:`~twinkle.metric.Accuracy`, which scores logits against label ids at
the token level -- on the same run the two report different (and not comparable) numbers.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import Metric


class TextMetric(Metric):
    """Base for metrics over ``(prediction, reference)`` text pairs.

    Pairs accumulate locally and are gathered in :meth:`calculate`, so a subclass works unchanged on
    a single process (``device_mesh=None``) and across data-parallel ranks.

    Args:
        device_mesh: The device mesh, or None to score only what this process accumulated.
        process_group: The process group to collect pairs from.
    """

    def __init__(self, device_mesh=None, process_group=None, **kwargs):
        super().__init__(device_mesh, process_group, **kwargs)
        self.predictions: List[str] = []
        self.references: List[str] = []

    def reset(self):
        self.predictions = []
        self.references = []

    def accumulate(self,
                   inputs=None,  # ignore
                   outputs=None,  # ignore
                   *,
                   predictions: Optional[Sequence[str]] = None,
                   references: Optional[Sequence[str]] = None,
                   **kwargs):
        if predictions is None or references is None:
            return
        assert len(predictions) == len(references), (
            f'predictions and references must be parallel, got {len(predictions)} and {len(references)}.')
        self.predictions.extend(predictions)
        self.references.extend(references)

    def gathered_pairs(self) -> List[Tuple[str, str]]:
        """Every rank's pairs, and reset. Pairs travel as dicts: ``gather_object`` flattens sequences."""
        local_results = [{
            'prediction': prediction,
            'reference': reference
        } for prediction, reference in zip(self.predictions, self.references, strict=True)]
        all_results = self.gather_results(local_results)
        self.reset()
        return [(row['prediction'], row['reference']) for row in all_results]

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if len(values) > 0 else 0.0


class ExactMatch(TextMetric):
    """Fraction of predictions that equal their reference string exactly."""

    def calculate(self) -> Dict[str, Any]:
        pairs = self.gathered_pairs()
        if not pairs:
            return {}
        return {'acc': self._mean([float(prediction == reference) for prediction, reference in pairs])}


class RougeBleu(TextMetric):
    """ROUGE-1/2/L and BLEU-4 over jieba-tokenized text, as percentages.

    Needs ``jieba``, ``nltk`` and ``rouge``, imported on use so the rest of the metrics stay usable
    without them. Pairs where either side tokenizes to nothing are skipped rather than scored 0.
    """

    keys = ('rouge-1', 'rouge-2', 'rouge-l', 'bleu-4')

    def calculate(self) -> Dict[str, Any]:
        import jieba
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        from rouge.rouge import Rouge

        pairs = self.gathered_pairs()
        if not pairs:
            return {}
        scores: Dict[str, List[float]] = {key: [] for key in self.keys}
        for prediction, reference in pairs:
            hypothesis = [word.strip(' ') for word in jieba.cut(prediction) if word.strip(' ')]
            target = [word.strip(' ') for word in jieba.cut(reference) if word.strip(' ')]
            if not hypothesis or not target:
                continue
            for key, value in Rouge().get_scores(' '.join(hypothesis), ' '.join(target))[0].items():
                scores[key].append(value['f'])
            scores['bleu-4'].append(
                sentence_bleu([target], hypothesis, smoothing_function=SmoothingFunction().method3))
        return {key: round(self._mean(values) * 100, 6) for key, values in scores.items()}
