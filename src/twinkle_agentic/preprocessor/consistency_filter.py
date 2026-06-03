# Copyright (c) ModelScope Contributors. All rights reserved.
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import numpy as np

from twinkle.preprocessor import Preprocessor

from .llm_backend import LLMBackend, OpenAIBackend


def _get_assistant_text(messages: List[Dict[str, Any]]) -> Optional[str]:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get('role') == 'assistant':
            return (m.get('content') or '').strip()
    return None


def _get_prompt_messages(messages: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
    """Return messages up to (not including) the last assistant turn."""
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get('role') == 'assistant':
            return messages[:i]
    return None


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def _pairwise_cosine_mean(embeddings: np.ndarray) -> float:
    """Mean pairwise cosine similarity for N embeddings of shape (N, dim)."""
    n = len(embeddings)
    if n < 2:
        return 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.clip(norms, 1e-12, None)
    sim_matrix = normed @ normed.T
    return float(sim_matrix[np.triu_indices(n, k=1)].mean())


def _generate_rollouts(
    backend: LLMBackend,
    prompt_messages: List[Dict[str, Any]],
    n: int,
    temperature: float,
) -> List[str]:
    choices = backend.chat(prompt_messages, temperature=temperature, max_tokens=4096, n=n)
    return [c.get('content', '') for c in choices]


def _embed_texts(
    backend: LLMBackend,
    texts: List[str],
) -> np.ndarray:
    return backend.embeddings(texts)


def _process_row(
    backend: LLMBackend,
    embed_backend: LLMBackend,
    messages: List[Dict[str, Any]],
    n_rollouts: int,
    temperature: float,
) -> Optional[Dict[str, Any]]:
    """Returns {'C': float, 'D': float, 'best_rollout': str, 'best_d': float} or None."""
    prompt_msgs = _get_prompt_messages(messages)
    if not prompt_msgs:
        return None

    traj_text = _get_assistant_text(messages)
    if not traj_text:
        return None

    try:
        rollout_texts = _generate_rollouts(
            backend,
            prompt_msgs, n_rollouts, temperature,
        )
    except Exception:
        return None

    rollout_texts = [t for t in rollout_texts if t.strip()]
    if len(rollout_texts) < 2:
        return None

    try:
        embeddings = _embed_texts(
            embed_backend, [traj_text] + rollout_texts)
    except Exception:
        return None

    if len(embeddings) != 1 + len(rollout_texts):
        return None

    traj_emb = embeddings[0]
    rollout_embs = embeddings[1:]

    c = _pairwise_cosine_mean(rollout_embs)
    d = 1.0 - _cosine_sim(rollout_embs.mean(axis=0), traj_emb)

    # rollout closest to original traj
    norms = np.linalg.norm(rollout_embs, axis=1, keepdims=True)
    normed_r = rollout_embs / np.clip(norms, 1e-12, None)
    traj_norm = traj_emb / max(np.linalg.norm(traj_emb), 1e-12)
    sims = normed_r @ traj_norm
    best_idx = int(np.argmax(sims))

    return {
        'C': c,
        'D': d,
        'best_rollout': rollout_texts[best_idx],
        'best_d': 1.0 - float(sims[best_idx]),
    }


class ConsistencyFilter(Preprocessor):
    """2D consistency filter: rollout consistency (C) × deviation from original traj (D).

    Quadrants:
      A (C>=thresh, D<thresh): stable & faithful → keep
      B (C>=thresh, D>=thresh): stable but drifted → source-dependent
      C (C<thresh, D<thresh): unstable but on-target → high learning value
      D (C<thresh, D>=thresh): unstable & off-target → filter

    Modes (combinable):
      filter only:            drop quadrant D (and B when source=self)
      annotate=True:          keep all, attach _quadrant/_diff_score/_consistency/_deviation
      replace=True:           replace assistant traj with best rollout where safe
    """

    def __init__(
        self,
        backend: LLMBackend = None,
        embed_backend: LLMBackend = None,
        n_rollouts: int = 8,
        c_thresh: float = 0.7,
        d_thresh: float = 0.3,
        temperature: float = 0.7,
        max_workers: int = 4,
        source: str = 'auto',
        annotate: bool = False,
        replace: bool = False,
        min_density_ratio: float = 0.4,
        # Legacy params
        sampler_endpoint: str = '',
        embed_endpoint: str = '',
        sampler_model: str = 'default',
        embed_model: str = 'bge-m3',
    ):
        if backend is not None:
            self._backend = backend
        else:
            self._backend = OpenAIBackend(
                endpoint=sampler_endpoint, model=sampler_model, timeout=300.0)
        if embed_backend is not None:
            self._embed_backend = embed_backend
        else:
            self._embed_backend = OpenAIBackend(
                endpoint=embed_endpoint, model=embed_model, timeout=300.0)
        self._n_rollouts = n_rollouts
        self._c_thresh = c_thresh
        self._d_thresh = d_thresh
        self._temperature = temperature
        self._max_workers = max_workers
        self._source = source
        self._annotate = annotate
        self._replace = replace
        self._min_density_ratio = min_density_ratio

    def _assign_quadrant(self, c: float, d: float) -> str:
        if c >= self._c_thresh:
            return 'A' if d < self._d_thresh else 'B'
        return 'C' if d < self._d_thresh else 'D'

    def _should_drop(self, quadrant: str, row: Dict[str, Any]) -> bool:
        """Whether to remove the row entirely (only applies in non-annotate mode)."""
        if quadrant == 'D':
            return True
        if quadrant == 'B':
            if self._source == 'self':
                return True
            if self._source == 'auto' and row.get('_source') == 'self':
                return True
        return False

    def _try_replace(self, row: Dict[str, Any], metrics: Dict[str, Any], quadrant: str) -> None:
        """Attempt in-place replacement of assistant content with best rollout."""
        original = _get_assistant_text(row.get('messages') or []) or ''
        best = metrics['best_rollout']
        density = len(best) / max(len(original), 1)

        if quadrant == 'A':
            if density >= self._min_density_ratio:
                self._set_assistant_text(row, best)
                row['_replaced'] = True
            else:
                row['_replaced'] = False
                row['_needs_completion'] = True
        elif quadrant == 'C' and metrics['best_d'] < self._d_thresh * 0.8:
            if density >= self._min_density_ratio:
                self._set_assistant_text(row, best)
                row['_replaced'] = True
            else:
                row['_replaced'] = False
        elif quadrant == 'B':
            row['_replaced'] = False
            row['_needs_verification'] = True
        else:
            row['_replaced'] = False

    @staticmethod
    def _set_assistant_text(row: Dict[str, Any], text: str) -> None:
        for m in reversed(row.get('messages') or []):
            if isinstance(m, dict) and m.get('role') == 'assistant':
                m['content'] = text
                return

    def __call__(self, rows) -> List[Dict[str, Any]]:
        if not rows:
            return rows

        results: Dict[int, Optional[Dict[str, Any]]] = {}
        n_workers = min(self._max_workers, len(rows))

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            future_to_idx = {
                pool.submit(
                    _process_row,
                    self._backend, self._embed_backend,
                    row.get('messages') or [], self._n_rollouts, self._temperature,
                ): i
                for i, row in enumerate(rows)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = None

        out = []
        for i, row in enumerate(rows):
            metrics = results.get(i)

            if metrics is None:
                if self._annotate:
                    row['_quadrant'] = 'unknown'
                    row['_diff_score'] = -1.0
                out.append(row)
                continue

            c, d = metrics['C'], metrics['D']
            quadrant = self._assign_quadrant(c, d)

            # filter decision (skip in annotate mode — annotate keeps everything)
            if not self._annotate and self._should_drop(quadrant, row):
                continue

            if self._annotate:
                row['_quadrant'] = quadrant
                row['_diff_score'] = (1.0 - c) if d < self._d_thresh else 0.0
                row['_consistency'] = c
                row['_deviation'] = d

            if self._replace:
                self._try_replace(row, metrics, quadrant)

            out.append(row)

        return out
