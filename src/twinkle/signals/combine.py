# Copyright (c) ModelScope Contributors. All rights reserved.
"""Blend aligned per-row lanes into one."""
import math
from typing import Any, List, Literal, Optional, Sequence, Tuple

from twinkle import remote_class, remote_function

from .reduce import as_rows

MODES = ('weighted_sum', 'probability_mixture')


def _blend(terms: List[Tuple[float, float]], mode: str) -> float:
    """``sum_k w_k v_k / sum_k w_k``, or ``logsumexp_k(log w_k + v_k)``, over ``(value, weight)`` terms."""
    if mode == 'probability_mixture':
        logs = [math.log(max(weight, 1e-12)) + value for value, weight in terms]
        top = max(logs)
        return top + math.log(sum(math.exp(value - top) for value in logs))
    total = sum(weight for _, weight in terms)
    return sum(weight * value for value, weight in terms) / total if total else 0.0


@remote_class(execute='first')
class SignalCombine:
    """Blend aligned lanes row by row: a weighted mean, or a probability mixture.

    ``signals`` is a list of lanes with one value per row; a row is a scalar or a token sequence.
    ``weighted_sum`` is ``sum_k w_k v_k / sum_k w_k`` (the weights are mixture coefficients, so
    scaling them all leaves the result unchanged); ``probability_mixture`` is
    ``logsumexp_k(log w_k + v_k)`` (logps lanes mix in probability space). A scalar row blended with
    token rows is broadcast over the tokens; token rows blend position by position over the lanes
    that have the position (a position no lane has reads 0.0). ``None`` drops out of its row; a row
    with no value left is ``None``. ``weights`` defaults to uniform.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(self,
                 signals: Sequence[Any],
                 *,
                 weights: Optional[Sequence[float]] = None,
                 mode: Literal['weighted_sum', 'probability_mixture'] = 'weighted_sum',
                 **kwargs) -> List[Any]:
        if mode not in MODES:
            raise ValueError(f'combine mode {mode!r} not in {MODES}.')
        lanes = [as_rows(lane) for lane in signals]
        if not lanes:
            raise ValueError('SignalCombine needs at least one lane.')
        weights = [1.0 / len(lanes)] * len(lanes) if weights is None else [float(weight) for weight in weights]
        if len(weights) != len(lanes):
            raise ValueError(f'{len(weights)} weights for {len(lanes)} lanes.')
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError('combine weights must be non-negative with a positive sum.')
        counts = {len(lane) for lane in lanes}
        if len(counts) != 1:
            raise ValueError(f'the lanes carry different row counts {sorted(counts)}; they must be aligned.')
        out: List[Any] = []
        for index in range(counts.pop()):
            terms = [(lane[index], weight) for lane, weight in zip(lanes, weights) if lane[index] is not None]
            if not terms:
                out.append(None)
                continue
            tokens = [value for value, _ in terms if isinstance(value, (list, tuple))]
            if not tokens:
                out.append(_blend([(float(value), weight) for value, weight in terms], mode))
                continue
            row = []
            for position in range(max(len(value) for value in tokens)):
                present = []
                for value, weight in terms:
                    if not isinstance(value, (list, tuple)):
                        present.append((float(value), weight))
                    elif position < len(value) and value[position] is not None:
                        present.append((float(value[position]), weight))
                row.append(_blend(present, mode) if present else 0.0)
            out.append(row)
        return out
