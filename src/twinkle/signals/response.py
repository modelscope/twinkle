# Copyright (c) ModelScope Contributors. All rights reserved.
"""Response-token score rows: one forward channel aligned to each trajectory's trainable tokens."""
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from twinkle import remote_class, remote_function


def response_positions(feature: Dict[str, Any]) -> List[int]:
    """Indices of the trainable (response) tokens in ``feature``, i.e. where ``labels != -100``.

    This is the exact frame the loss mask uses, so scores extracted at these positions (a teacher's
    log-probs, a critic's values) align one-to-one with the policy's per-token loss.
    """
    labels = feature.get('labels')
    if labels is None:
        return []
    if isinstance(labels, torch.Tensor):
        return torch.nonzero(labels.reshape(-1) != -100, as_tuple=True)[0].tolist()
    return np.flatnonzero(np.asarray(labels) != -100).tolist()


def _row_scores(row: torch.Tensor, positions: List[int], feature: Dict[str, Any]) -> List[float]:
    """One feature's response-token scores from its (possibly right-padded) full-sequence row.

    A response-only row already holds exactly ``len(positions)`` values and is returned as-is; a
    full-sequence row is indexed at the feature's own response positions. A row shorter than the
    feature cannot be aligned and raises rather than silently mis-scoring.
    """
    if row.numel() == len(positions):
        return row.tolist()
    labels = feature.get('labels')
    feature_length = len(labels) if labels is not None else 0
    if row.numel() < feature_length:
        raise RuntimeError(f'forward returned {row.numel()} values for a {feature_length}-token feature; '
                           'response-token alignment is impossible.')
    if (positions and positions[-1] - positions[0] + 1 == len(positions)
            and bool(np.all(np.diff(positions) == 1))):
        return row.narrow(0, positions[0], len(positions)).tolist()
    return row[positions].tolist()


def response_token_scores(outputs: Any, features: List[Dict[str, Any]], *, output_key: str = 'logps') -> List[List[float]]:
    """Align one whole-batch forward return to one response-token score row per feature.

    ``outputs`` is the forward's return (a pending handle is resolved here) -- a dict whose
    ``output_key`` channel (``'logps'`` for a causal scorer, ``'logits'`` / ``'values'`` for a value
    head) is a ``[B, T]`` tensor or a bare tensor. Full-sequence, right-padded rows are indexed at
    each feature's own response positions, so right-padding never shifts them.
    """
    outputs = outputs.result() if hasattr(outputs, 'result') else outputs
    values = outputs.get(output_key) if isinstance(outputs, dict) else outputs
    if values is None:
        raise RuntimeError(f'the forward returned no {output_key!r} channel.')
    values = torch.as_tensor(values).detach().float().cpu()
    if values.dim() == 1:
        values = values.unsqueeze(0)
    if values.shape[0] != len(features):
        raise RuntimeError(f'forward returned {values.shape[0]} rows for {len(features)} features; '
                           'per-row alignment is impossible.')
    return [_row_scores(values[i].reshape(-1), response_positions(feature), feature)
            for i, feature in enumerate(features)]


@remote_class(execute='first')
class ResponseScores:
    """One per-row score lane of a batch, read where the batch lives.

    - ``outputs`` and ``trajs``: the ``output_key`` channel of one whole-batch forward return,
      aligned to each trajectory's response tokens (:func:`response_token_scores`).
    - ``outputs`` alone: the channel itself, unaligned (a per-sample embedding / logits channel).
    - ``trajs`` alone: each trajectory's own ``output_key`` field, already one value per response
      token (the sampler's ``rollout_logprobs``).
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(self,
                 trajs: Optional[List[Dict[str, Any]]] = None,
                 outputs: Any = None,
                 *,
                 output_key: str = 'logps',
                 **kwargs) -> Any:
        if outputs is None:
            if trajs is None:
                raise ValueError('ResponseScores needs trajectories, a forward return, or both.')
            rows = []
            for index, traj in enumerate(trajs):
                row = traj.get(output_key)
                if row is None:
                    raise ValueError(f'trajectory {index} carries no {output_key!r} field.')
                rows.append([float(value) for value in row])
            return rows
        if trajs is None:
            channel = outputs.get(output_key) if isinstance(outputs, dict) else None
            if channel is None:
                raise ValueError(f'the forward returned no {output_key!r} channel.')
            return channel
        return response_token_scores(outputs, trajs, output_key=output_key)
