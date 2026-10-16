# Copyright (c) ModelScope Contributors. All rights reserved.
"""Infrastructure-agnostic admission policy for dynamic GRPO sampling.

The policy turns the reward-signal observation used by GRPO into an explicit
sampling decision while preserving complete prompt groups.  It intentionally
does not call a sampler or a trainer: synchronous loops, asynchronous workers,
and offline rollout pipelines can share the same deterministic policy.
"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union

_SimilarityScorer = Callable[[str, str], float]

__all__ = [
    'GroupAdmissionConfig',
    'GroupAdmissionDecision',
    'GroupAdmissionPolicy',
    'ResamplePlan',
    'SamplingBudgetConfig',
    'SamplingBudgetController',
    'SamplingBudgetState',
    'group_admission_metrics',
]

_REWARD_REJECTION_REASONS = frozenset({
    'reward_std_below_threshold',
    'reward_range_below_threshold',
    'insufficient_nontrivial_advantages',
})
_NEAR_DUPLICATE_REASON = 'mean_pairwise_similarity_above_threshold'


@dataclass(frozen=True)
class GroupAdmissionConfig:
    """Thresholds for deciding whether one complete GRPO group is useful.

    ``min_reward_std``, ``min_reward_range``, and
    ``min_nontrivial_advantages`` generalize the reward-variance gate
    popularized by DAPO-style dynamic sampling.  The range threshold lets
    dense-reward applications reject differences below their meaningful
    reward resolution.  The optional similarity threshold adds a
    deterministic near-duplicate signal.  Defaults are deliberately disabled
    so constructing the policy is a no-op.
    """

    min_reward_std: float = 0.0
    min_reward_range: float = 0.0
    min_nontrivial_advantages: int = 0
    advantage_tolerance: float = 1e-8
    max_mean_pairwise_similarity: float | None = None
    similarity_ngram_size: int = 3

    def __post_init__(self) -> None:
        if not math.isfinite(self.min_reward_std) or self.min_reward_std < 0:
            raise ValueError('min_reward_std must be finite and non-negative')
        if not math.isfinite(self.min_reward_range) or self.min_reward_range < 0:
            raise ValueError('min_reward_range must be finite and non-negative')
        if self.min_nontrivial_advantages < 0:
            raise ValueError('min_nontrivial_advantages must be non-negative')
        if not math.isfinite(self.advantage_tolerance) or self.advantage_tolerance < 0:
            raise ValueError('advantage_tolerance must be finite and non-negative')
        threshold = self.max_mean_pairwise_similarity
        if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('max_mean_pairwise_similarity must be in [0, 1]')
        if self.similarity_ngram_size <= 0:
            raise ValueError('similarity_ngram_size must be positive')


@dataclass(frozen=True)
class GroupAdmissionDecision:
    """Decision and auditable observations for one indivisible prompt group."""

    reasons: tuple[str, ...]
    reward_std: float
    reward_range: float
    nontrivial_advantage_count: int
    mean_pairwise_similarity: float | None = None
    max_pairwise_similarity: float | None = None
    external_reasons: tuple[str, ...] = ()

    @property
    def admitted(self) -> bool:
        return not self.reasons

    @property
    def reward_rejected(self) -> bool:
        internal_reasons = self.reasons[len(self.external_reasons):]
        return any(reason in _REWARD_REJECTION_REASONS for reason in internal_reasons)

    @property
    def near_duplicate_rejected(self) -> bool:
        internal_reasons = self.reasons[len(self.external_reasons):]
        return _NEAR_DUPLICATE_REASON in internal_reasons

    @property
    def primary_rejection_reason(self) -> str | None:
        """Return one stable, mutually exclusive reason for rejected groups."""
        if self.admitted:
            return None
        if self.external_reasons:
            return 'external'
        if self.reward_rejected:
            return 'exact_dead' if self.reward_range == 0.0 else 'near_tie'
        if self.near_duplicate_rejected:
            return 'redundant'
        return None


class GroupAdmissionPolicy:
    """Admit complete groups with useful reward and representation signals."""

    def __init__(
        self,
        config: GroupAdmissionConfig | None = None,
        *,
        scorer: _SimilarityScorer | None = None,
    ):
        self.config = config or GroupAdmissionConfig()
        self._similarity_scorer = scorer

    def evaluate(
            self,
            rewards: Sequence[float],
            completions: Sequence[str] | None = None,
            *,
            external_reasons: Sequence[str] = (),
    ) -> GroupAdmissionDecision:
        reward_values = _validated_rewards(rewards)
        completion_values = list(completions) if completions is not None else None
        if completion_values is not None and len(completion_values) != len(reward_values):
            raise ValueError('rewards and completions must describe the same complete group')

        reward_mean = sum(reward_values) / len(reward_values)
        reward_std = statistics.pstdev(reward_values)
        reward_range = max(reward_values) - min(reward_values)
        nontrivial = sum(abs(value - reward_mean) > self.config.advantage_tolerance for value in reward_values)

        reward_reasons = []
        if reward_std < self.config.min_reward_std:
            reward_reasons.append('reward_std_below_threshold')
        if (reward_range < self.config.min_reward_range and not math.isclose(
                reward_range,
                self.config.min_reward_range,
                rel_tol=1e-12,
                abs_tol=1e-12,
        )):
            reward_reasons.append('reward_range_below_threshold')
        if nontrivial < self.config.min_nontrivial_advantages:
            reward_reasons.append('insufficient_nontrivial_advantages')

        mean_similarity, max_similarity, similarity_reasons = self._evaluate_similarity(completion_values)
        external = tuple(str(reason) for reason in external_reasons if str(reason))
        reasons = (*external, *reward_reasons, *similarity_reasons)
        return GroupAdmissionDecision(
            reasons=reasons,
            reward_std=reward_std,
            reward_range=reward_range,
            nontrivial_advantage_count=nontrivial,
            mean_pairwise_similarity=mean_similarity,
            max_pairwise_similarity=max_similarity,
            external_reasons=external,
        )

    def _evaluate_similarity(
        self,
        completions: Sequence[str] | None,
    ) -> tuple[float | None, float | None, tuple[str, ...]]:
        threshold = self.config.max_mean_pairwise_similarity
        if threshold is None:
            return None, None, ()
        if completions is None:
            raise ValueError('completions are required when the near-duplicate gate is enabled')
        values = list(completions)
        if len(values) < 2:
            raise ValueError('the near-duplicate gate requires at least two completions')

        scorer = self._similarity_scorer
        scores = []
        for left_index, left in enumerate(values):
            for right in values[left_index + 1:]:
                score = float(
                    scorer(left, right) if scorer is not None else _ngram_jaccard(
                        left,
                        right,
                        ngram_size=self.config.similarity_ngram_size,
                    ))
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError(f'similarity scorer must return a finite value in [0, 1], got {score}')
                scores.append(score)
        mean_similarity = sum(scores) / len(scores)
        reasons = (_NEAR_DUPLICATE_REASON, ) if mean_similarity > threshold else ()
        return mean_similarity, max(scores), reasons


@dataclass(frozen=True)
class SamplingBudgetConfig:
    """Hard caps and EMA parameters for allocating replacement groups."""

    max_extra_groups: int = 0
    max_extra_groups_per_round: int | None = None
    max_resample_rounds: int = 0
    max_total_samples: int | None = None
    max_total_tokens: int | None = None
    effective_rate_ema_alpha: float = 0.2
    min_effective_rate: float = 0.05

    def __post_init__(self) -> None:
        for name in ('max_extra_groups', 'max_resample_rounds'):
            if getattr(self, name) < 0:
                raise ValueError(f'{name} must be non-negative')
        for name in ('max_extra_groups_per_round', 'max_total_samples', 'max_total_tokens'):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f'{name} must be non-negative when provided')
        if not 0 < self.effective_rate_ema_alpha <= 1:
            raise ValueError('effective_rate_ema_alpha must be in (0, 1]')
        if not 0 < self.min_effective_rate <= 1:
            raise ValueError('min_effective_rate must be in (0, 1]')


@dataclass
class SamplingBudgetState:
    """Mutable, serializable state scoped to one sampling batch/partition."""

    sampling_rounds: int = 0
    resample_rounds: int = 0
    generated_groups: int = 0
    generated_samples: int = 0
    generated_tokens: int = 0
    admitted_groups: int = 0
    rejected_groups: int = 0
    effective_rate_ema: float | None = None


@dataclass(frozen=True)
class ResamplePlan:
    extra_groups: int
    remaining_target_groups: int
    effective_rate: float
    exhausted: bool
    limited_by: tuple[str, ...] = ()


class SamplingBudgetController:
    """Translate observed group effectiveness into bounded resampling work."""

    def __init__(self, config: SamplingBudgetConfig | None = None):
        self.config = config or SamplingBudgetConfig()

    def observe_round(
        self,
        state: SamplingBudgetState,
        decisions: Sequence[GroupAdmissionDecision],
        *,
        num_generations: int,
        generated_tokens: int = 0,
    ) -> None:
        if num_generations <= 0:
            raise ValueError('num_generations must be positive')
        if generated_tokens < 0:
            raise ValueError('generated_tokens must be non-negative')
        values = list(decisions)
        if not values:
            raise ValueError('at least one group decision is required')
        admitted = sum(decision.admitted for decision in values)
        rate = admitted / len(values)
        if state.effective_rate_ema is None:
            state.effective_rate_ema = rate
        else:
            alpha = self.config.effective_rate_ema_alpha
            state.effective_rate_ema = alpha * rate + (1 - alpha) * state.effective_rate_ema
        state.sampling_rounds += 1
        state.generated_groups += len(values)
        state.generated_samples += len(values) * num_generations
        state.generated_tokens += generated_tokens
        state.admitted_groups += admitted
        state.rejected_groups += len(values) - admitted

    def plan_resampling(
        self,
        state: SamplingBudgetState,
        *,
        target_groups: int,
        num_generations: int,
        estimated_tokens_per_group: float | None = None,
    ) -> ResamplePlan:
        if target_groups <= 0 or num_generations <= 0:
            raise ValueError('target_groups and num_generations must be positive')
        if estimated_tokens_per_group is not None and estimated_tokens_per_group <= 0:
            raise ValueError('estimated_tokens_per_group must be positive when provided')

        remaining = max(0, target_groups - state.admitted_groups)
        effective_rate = max(
            self.config.min_effective_rate,
            state.effective_rate_ema if state.effective_rate_ema is not None else 1.0,
        )
        if remaining == 0:
            return ResamplePlan(0, 0, effective_rate, False, ('target_met', ))
        if state.resample_rounds >= self.config.max_resample_rounds:
            return ResamplePlan(0, remaining, effective_rate, True, ('resample_round_budget', ))

        requested = max(remaining, math.ceil(remaining / effective_rate))
        limits: list[tuple[str, int]] = [
            ('extra_group_budget', target_groups + self.config.max_extra_groups - state.generated_groups),
        ]
        if self.config.max_extra_groups_per_round is not None:
            limits.append(('per_round_group_budget', self.config.max_extra_groups_per_round))
        if self.config.max_total_samples is not None:
            remaining_samples = self.config.max_total_samples - state.generated_samples
            limits.append(('sample_budget', remaining_samples // num_generations))
        if self.config.max_total_tokens is not None:
            token_estimate = estimated_tokens_per_group
            if token_estimate is None and state.generated_groups and state.generated_tokens:
                token_estimate = state.generated_tokens / state.generated_groups
            if token_estimate is not None:
                remaining_tokens = self.config.max_total_tokens - state.generated_tokens
                limits.append(('token_budget', math.floor(remaining_tokens / token_estimate)))

        normalized_limits = [(name, max(0, value)) for name, value in limits]
        extra_groups = min([requested, *(value for _, value in normalized_limits)])
        limited_by = tuple(name for name, value in normalized_limits if value <= requested and value == extra_groups)
        exhausted = extra_groups == 0
        if extra_groups:
            state.resample_rounds += 1
        return ResamplePlan(extra_groups, remaining, effective_rate, exhausted, limited_by)


def group_admission_metrics(decisions: Sequence[GroupAdmissionDecision]) -> dict[str, float | int]:
    """Aggregate decisions using names suitable for Twinkle metric records."""
    values = list(decisions)
    if not values:
        return {
            'candidate_group_count': 0,
            'admitted_group_count': 0,
            'rejected_group_count': 0,
            'effective_group_rate': 0.0,
            'reward_rejected_group_count': 0,
            'near_duplicate_rejected_group_count': 0,
            'externally_rejected_group_count': 0,
            'exact_dead_group_count': 0,
            'near_tie_group_count': 0,
            'redundant_group_count': 0,
        }
    admitted = sum(decision.admitted for decision in values)
    reward_rejected = sum(decision.reward_rejected for decision in values)
    near_duplicate_rejected = sum(decision.near_duplicate_rejected for decision in values)
    externally_rejected = sum(bool(decision.external_reasons) for decision in values)
    primary_reasons = [decision.primary_rejection_reason for decision in values]
    return {
        'candidate_group_count': len(values),
        'admitted_group_count': admitted,
        'rejected_group_count': len(values) - admitted,
        'effective_group_rate': admitted / len(values),
        'reward_rejected_group_count': reward_rejected,
        'near_duplicate_rejected_group_count': near_duplicate_rejected,
        'externally_rejected_group_count': externally_rejected,
        'exact_dead_group_count': primary_reasons.count('exact_dead'),
        'near_tie_group_count': primary_reasons.count('near_tie'),
        'redundant_group_count': primary_reasons.count('redundant'),
    }


def _validated_rewards(rewards: Sequence[float]) -> list[float]:
    values = [float(value) for value in rewards]
    if len(values) < 2:
        raise ValueError('a GRPO group requires at least two rewards')
    if any(not math.isfinite(value) for value in values):
        raise ValueError('group rewards must be finite')
    return values


def _ngram_jaccard(left: str, right: str, *, ngram_size: int) -> float:
    """Return lexical overlap for the default near-duplicate detector."""
    left_ngrams = _ngrams(left, ngram_size)
    right_ngrams = _ngrams(right, ngram_size)
    if not left_ngrams and not right_ngrams:
        return 1.0
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else 1.0


def _ngrams(text: str, ngram_size: int) -> set[tuple[str, ...]]:
    if not isinstance(text, str):
        raise TypeError(f'completion must be str, got {type(text)!r}')
    tokens = re.findall(r'\w+|[^\w\s]', text.lower(), flags=re.UNICODE)
    if not tokens:
        return set()
    if len(tokens) < ngram_size:
        return {tuple(tokens)}
    return {tuple(tokens[index:index + ngram_size]) for index in range(len(tokens) - ngram_size + 1)}
