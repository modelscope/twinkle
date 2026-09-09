# Copyright (c) ModelScope Contributors. All rights reserved.
"""Persistent proposer trajectories for challenger training and diagnosis."""
import json
import os
import threading
import uuid
from typing import Any, Dict, List

import numpy as np

_TOKEN_FIELDS = ('input_ids', 'labels', 'completion_mask', 'attention_mask', 'position_ids')


def _as_numpy(value: Any, dtype: Any = None) -> np.ndarray:
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _logprob_column(logprobs: Any) -> List[float]:
    """Extract the chosen token's log probability from each sampling step."""
    out: List[float] = []
    for step in logprobs:
        if isinstance(step, (int, float)):
            out.append(float(step))
            continue
        if isinstance(step, (list, tuple)) and step:
            chosen = step[0]
            if isinstance(chosen, (list, tuple)) and len(chosen) >= 2:
                out.append(float(chosen[1]))
                continue
        raise TypeError(f'cannot read a chosen-token logprob from {step!r}')
    return out


def _json_default(value: Any) -> Any:
    if hasattr(value, 'tolist'):
        return value.tolist()
    return str(value)


class RolloutRecorder:
    """Write token arrays to NPZ and trajectory metadata to a JSONL index."""

    def __init__(self, save_dir: str):
        self.trajectory_dir = os.path.join(save_dir, 'trajs')
        self.index_path = os.path.join(self.trajectory_dir, 'index.jsonl')
        os.makedirs(self.trajectory_dir, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, trajectory: Dict[str, Any], **fields: Any) -> None:
        arrays: Dict[str, np.ndarray] = {}
        for key in _TOKEN_FIELDS:
            value = trajectory.get(key)
            if value is not None:
                arrays[key] = _as_numpy(value, np.int32)
        logprobs = trajectory.get('logprobs')
        if logprobs is not None:
            arrays['logprobs'] = np.asarray(_logprob_column(logprobs), dtype=np.float64)

        name = f'{uuid.uuid4().hex}.npz'
        labels = arrays.get('labels', np.asarray([], dtype=np.int32))
        completion_mask = arrays.get('completion_mask')
        if completion_mask is None:
            n_policy_tokens = int((labels != -100).sum())
        else:
            if completion_mask.size != labels.size:
                raise ValueError('completion_mask and labels must have the same number of tokens')
            n_policy_tokens = int(((labels != -100) & completion_mask.astype(bool)).sum())
        n_logprobs = len(arrays.get('logprobs', ()))
        if logprobs is not None and n_logprobs != n_policy_tokens:
            raise ValueError(f'logprobs contain {n_logprobs} policy tokens, expected '
                             f'{n_policy_tokens} from labels and completion_mask')
        metadata = {
            key: value
            for key, value in trajectory.items() if key not in _TOKEN_FIELDS and key not in ('logprobs', 'rewards')
        }
        record = dict(metadata)
        record.update(fields)
        record.update({
            'npz': name,
            'n_tokens': int(arrays.get('input_ids', np.asarray([])).size),
            'n_policy_tokens': n_policy_tokens,
            'has_logprobs': logprobs is not None,
            'n_logprobs': n_logprobs,
        })
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self._lock:
            np.savez_compressed(os.path.join(self.trajectory_dir, name), **arrays)
            with open(self.index_path, 'a', encoding='utf-8') as handle:
                handle.write(line + '\n')
