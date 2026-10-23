# Copyright (c) ModelScope Contributors. All rights reserved.
"""Compute TF-IDF novelty among sibling proposals without a model call.

Fit on the full run corpus so shared scaffolding receives little IDF weight.
Distance is one minus the closest sibling cosine; spread is inverse cluster
size. The cluster threshold can be calibrated on observed sibling groups.
Out-of-vocabulary tokens are invisible, so fit on the run being scored.
scikit-learn and numpy are imported lazily. No reward integration is provided.
"""
from __future__ import annotations

import collections

__all__ = ['NoveltyScorer', 'constant_group_rate', 'spread']


def _union_find_components(n: int, edges) -> list[int]:
    """Size of each node's connected component, as a list indexed by node."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[root_b] = root_a
    sizes = collections.Counter(find(i) for i in range(n))
    return [sizes[find(i)] for i in range(n)]


class NoveltyScorer:
    """Fit on corpus, then score sibling groups using distance and spread.

    cluster_threshold overrides quantile calibration from calibration_groups.
    weight_distance and weight_spread control the blend; ngram_range,
    max_features and sublinear_tf configure TF-IDF.
    """

    def __init__(self,
                 corpus,
                 *,
                 ngram_range=(1, 2),
                 max_features: int = 200000,
                 cluster_threshold: float | None = None,
                 cluster_quantile: float = 0.75,
                 calibration_groups=None,
                 weight_distance: float = 0.5,
                 weight_spread: float = 0.5,
                 sublinear_tf: bool = True) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.cluster_quantile = float(cluster_quantile)
        self.cluster_threshold = float(cluster_threshold) if cluster_threshold is not None else None
        self.weight_distance = float(weight_distance)
        self.weight_spread = float(weight_spread)
        self.vectoriser = TfidfVectorizer(
            ngram_range=ngram_range,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
            strip_accents='unicode',
            lowercase=True,
            token_pattern=r'\b\w[\w\-]+\b|[^\s\w]')
        texts = [text for text in corpus if (text or '').strip()]
        if not texts:
            raise ValueError('NoveltyScorer needs at least one non-empty statement')
        self.vectoriser.fit(texts)
        self.n_fit = len(texts)
        self.n_features = len(self.vectoriser.vocabulary_)
        self.calibrated: dict[str, float] | None = None
        if self.cluster_threshold is None:
            pairs = self._pairwise(calibration_groups or [])
            if pairs:
                pairs.sort()
                index = min(len(pairs) - 1, max(0, int(round(self.cluster_quantile * (len(pairs) - 1)))))
                self.cluster_threshold = pairs[index]
                self.calibrated = {
                    'quantile': self.cluster_quantile,
                    'threshold': self.cluster_threshold,
                    'pairs': len(pairs)
                }
            else:
                self.cluster_threshold = 0.5

    def _pairwise(self, groups) -> list[float]:
        """Within-group sibling cosines: the distribution the threshold comes from."""
        out: list[float] = []
        for statements in groups:
            texts = [text for text in statements if (text or '').strip()]
            if len(texts) < 2:
                continue
            matrix = self.vectoriser.transform(texts)
            similarity = (matrix @ matrix.T).toarray()
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    out.append(float(similarity[i, j]))
        return out

    def score(self, statements, *, normalize: bool = True) -> list[float]:
        """One score per statement, in the same order. Empty input scores ``[]``.

        With ``normalize`` the raw mix is min-max mapped onto ``[0, 1]`` inside
        the group. That is not cosmetic: GRPO uses only a proposal's position
        relative to its siblings, so a term varying by 10% around a mean it
        shares with everyone is worth 10% of the ranking, and replacing a judge
        is pointless unless this axis decides its share of the ranking. The
        mapping is monotone, so the order -- the only thing the update sees -- is
        unchanged.
        """
        import numpy as np

        if not statements:
            return []
        if len(statements) == 1:
            # A group of one has nothing to differ from. 1.0 rather than 0.0, so
            # it does not read as "this proposal is a duplicate of something".
            return [1.0]
        matrix = self.vectoriser.transform([statement or '' for statement in statements])
        # Rows are L2-normalised by TfidfVectorizer (norm='l2' by default), so the
        # inner product is the cosine.
        similarity = (matrix @ matrix.T).toarray()
        np.fill_diagonal(similarity, -1.0)
        max_similarity = similarity.max(axis=1)
        edges = [(i, j) for i in range(len(statements)) for j in range(i + 1, len(statements))
                 if similarity[i, j] >= self.cluster_threshold]
        sizes = _union_find_components(len(statements), edges)
        raw = []
        for index in range(len(statements)):
            distance = max(0.0, 1.0 - float(max_similarity[index]))
            raw.append(self.weight_distance * distance + self.weight_spread * (1.0 / sizes[index]))
        if normalize and len(raw) > 1:
            low, high = min(raw), max(raw)
            if high > low:
                return [(value - low) / (high - low) for value in raw]
        return raw


def spread(values) -> float:
    """``max - min``: the thing that has to be non-zero for GRPO to see a term."""
    values = [float(value) for value in values if value is not None]
    return (max(values) - min(values)) if len(values) > 1 else 0.0


def constant_group_rate(groups) -> float:
    """Fraction of groups whose scores are all identical, i.e. contribute nothing.

    This is the number a novelty term is judged on, and the reason it is exported
    rather than kept in a notebook: a replacement that is not measured this way
    cannot be compared to what it replaced. On the judge it measured 0.66 and
    0.824 across two recorded runs.
    """
    total = constant = 0
    for values in groups:
        values = [float(value) for value in values if value is not None]
        if len(values) < 2:
            continue
        total += 1
        constant += spread(values) == 0.0
    return (constant / total) if total else 0.0
