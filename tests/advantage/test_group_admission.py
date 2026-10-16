# Copyright (c) ModelScope Contributors. All rights reserved.

import pytest

from twinkle.advantage import __all__ as advantage_exports
from twinkle.advantage.group_admission import (GroupAdmissionConfig, GroupAdmissionPolicy, SamplingBudgetConfig,
                                               SamplingBudgetController, SamplingBudgetState,
                                               group_admission_metrics)


def _decision(admitted: bool):
    policy = GroupAdmissionPolicy(
        GroupAdmissionConfig(
            min_reward_std=0.1,
            min_nontrivial_advantages=2,
        ))
    rewards = [0.0, 1.0] if admitted else [0.0, 0.0]
    return policy.evaluate(rewards)


def test_group_admission_does_not_expand_the_advantage_root_api() -> None:
    specialized_names = {
        'GroupAdmissionConfig',
        'GroupAdmissionPolicy',
        'SamplingBudgetConfig',
        'SamplingBudgetController',
    }
    assert specialized_names.isdisjoint(advantage_exports)


def test_default_policy_preserves_existing_behavior() -> None:
    decision = GroupAdmissionPolicy().evaluate(
        [0.0, 0.0, 0.0],
        ['same completion', 'same completion', 'same completion'],
    )
    assert decision.admitted
    assert decision.reasons == ()
    assert decision.primary_rejection_reason is None
    assert decision.reward_range == 0.0


def test_reward_gate_rejects_zero_signal_group() -> None:
    decision = GroupAdmissionPolicy(
        GroupAdmissionConfig(
            min_reward_std=1e-6,
            min_nontrivial_advantages=2,
        )).evaluate([1.0, 1.0, 1.0])
    assert not decision.admitted
    assert decision.reward_std == 0.0
    assert decision.reward_range == 0.0
    assert decision.nontrivial_advantage_count == 0
    assert decision.primary_rejection_reason == 'exact_dead'
    assert decision.reasons == (
        'reward_std_below_threshold',
        'insufficient_nontrivial_advantages',
    )


def test_reward_range_rejects_nonzero_but_resolution_insignificant_group() -> None:
    decision = GroupAdmissionPolicy(
        GroupAdmissionConfig(min_reward_range=0.02)).evaluate([0.5, 0.5, 0.51])
    assert not decision.admitted
    assert decision.reward_std > 0
    assert decision.reward_range == pytest.approx(0.01)
    assert decision.reasons == ('reward_range_below_threshold', )
    assert decision.primary_rejection_reason == 'near_tie'


def test_reward_range_admits_difference_at_meaningful_resolution() -> None:
    decision = GroupAdmissionPolicy(
        GroupAdmissionConfig(min_reward_range=0.02)).evaluate([0.1, 0.1, 0.12])
    assert decision.admitted
    assert decision.reward_range == pytest.approx(0.02)
    assert decision.primary_rejection_reason is None


def test_zero_range_threshold_preserves_std_only_dapo_style_gate() -> None:
    policy = GroupAdmissionPolicy(
        GroupAdmissionConfig(min_reward_std=1e-6, min_reward_range=0.0))
    assert not policy.evaluate([0.5, 0.5, 0.5]).admitted
    assert policy.evaluate([0.5, 0.5, 0.5001]).admitted


def test_reward_gate_admits_group_with_nontrivial_advantages() -> None:
    decision = GroupAdmissionPolicy(
        GroupAdmissionConfig(
            min_reward_std=0.4,
            min_nontrivial_advantages=2,
        )).evaluate([0.0, 1.0, 0.0, 1.0])
    assert decision.admitted
    assert decision.reward_std == pytest.approx(0.5)
    assert decision.nontrivial_advantage_count == 4


def test_diversity_gate_rejects_collective_near_duplicates() -> None:
    policy = GroupAdmissionPolicy(
        GroupAdmissionConfig(max_mean_pairwise_similarity=0.8, similarity_ngram_size=1))
    decision = policy.evaluate(
        [0.0, 1.0, 0.0],
        ['return price + tax', 'return price + tax', 'return price + tax'],
    )
    assert not decision.admitted
    assert decision.mean_pairwise_similarity == 1.0
    assert decision.reasons == ('mean_pairwise_similarity_above_threshold', )
    assert decision.primary_rejection_reason == 'redundant'


def test_diversity_gate_uses_group_mean_instead_of_one_duplicate_pair() -> None:
    policy = GroupAdmissionPolicy(
        GroupAdmissionConfig(max_mean_pairwise_similarity=0.5, similarity_ngram_size=1))
    decision = policy.evaluate(
        [0.0, 1.0, 0.0],
        ['alpha beta', 'alpha beta', 'gamma delta'],
    )
    assert decision.admitted
    assert decision.max_pairwise_similarity == 1.0
    assert decision.mean_pairwise_similarity == pytest.approx(1 / 3)


def test_custom_similarity_scorer_supports_domain_semantics() -> None:

    class ExecutionSignatureScorer:

        def __call__(self, left: str, right: str) -> float:
            return 1.0 if left.split(':', 1)[0] == right.split(':', 1)[0] else 0.0

    policy = GroupAdmissionPolicy(
        GroupAdmissionConfig(max_mean_pairwise_similarity=0.6),
        scorer=ExecutionSignatureScorer(),
    )
    decision = policy.evaluate(
        [0.0, 1.0, 0.0],
        ['success:syntax-a', 'success:syntax-b', 'success:syntax-c'],
    )
    assert not decision.admitted


def test_diversity_gate_requires_completions_only_when_enabled() -> None:
    assert GroupAdmissionPolicy().evaluate([0.0, 1.0]).admitted
    policy = GroupAdmissionPolicy(GroupAdmissionConfig(max_mean_pairwise_similarity=0.9))
    with pytest.raises(ValueError, match='completions are required'):
        policy.evaluate([0.0, 1.0])


def test_external_precondition_rejects_an_otherwise_useful_group() -> None:
    decision = GroupAdmissionPolicy().evaluate(
        [0.0, 1.0],
        external_reasons=['missing_execution'],
    )
    assert not decision.admitted
    assert decision.reasons == ('missing_execution', )
    assert decision.primary_rejection_reason == 'external'


def test_external_reason_name_cannot_be_misclassified_as_a_reward_rejection() -> None:
    decision = GroupAdmissionPolicy().evaluate(
        [0.0, 1.0],
        external_reasons=['reward_std_below_threshold'],
    )
    assert decision.primary_rejection_reason == 'external'
    assert not decision.reward_rejected
    assert group_admission_metrics([decision])['reward_rejected_group_count'] == 0


def test_default_near_duplicate_scorer_is_normalized_and_handles_empty_text() -> None:
    policy = GroupAdmissionPolicy(
        GroupAdmissionConfig(max_mean_pairwise_similarity=0.3, similarity_ngram_size=1))
    assert policy.evaluate([0.0, 1.0], ['', '']).mean_pairwise_similarity == 1.0
    assert policy.evaluate([0.0, 1.0], ['', 'value']).mean_pairwise_similarity == 0.0
    decision = policy.evaluate([0.0, 1.0], ['Alpha beta', 'alpha gamma'])
    assert decision.mean_pairwise_similarity == pytest.approx(1 / 3)


def test_budget_controller_uses_effective_rate_to_plan_replacements() -> None:
    controller = SamplingBudgetController(
        SamplingBudgetConfig(
            max_extra_groups=8,
            max_resample_rounds=2,
            effective_rate_ema_alpha=0.5,
        ))
    state = SamplingBudgetState()
    controller.observe_round(
        state,
        [_decision(True), _decision(False), _decision(True), _decision(False)],
        num_generations=3,
        generated_tokens=120,
    )
    plan = controller.plan_resampling(state, target_groups=4, num_generations=3)
    assert plan.extra_groups == 4
    assert plan.remaining_target_groups == 2
    assert plan.effective_rate == 0.5
    assert state.resample_rounds == 1


def test_budget_controller_updates_ema_between_rounds() -> None:
    controller = SamplingBudgetController(
        SamplingBudgetConfig(
            max_extra_groups=8,
            max_resample_rounds=2,
            effective_rate_ema_alpha=0.5,
        ))
    state = SamplingBudgetState()
    controller.observe_round(state, [_decision(True), _decision(False)], num_generations=2)
    controller.plan_resampling(state, target_groups=3, num_generations=2)
    controller.observe_round(state, [_decision(True), _decision(True)], num_generations=2)
    assert state.effective_rate_ema == pytest.approx(0.75)
    plan = controller.plan_resampling(state, target_groups=3, num_generations=2)
    assert plan.extra_groups == 0
    assert plan.limited_by == ('target_met', )


def test_budget_controller_enforces_sample_cap_at_group_boundary() -> None:
    controller = SamplingBudgetController(
        SamplingBudgetConfig(
            max_extra_groups=10,
            max_resample_rounds=3,
            max_total_samples=18,
        ))
    state = SamplingBudgetState()
    controller.observe_round(state, [_decision(False)] * 4, num_generations=3)
    plan = controller.plan_resampling(state, target_groups=4, num_generations=3)
    assert plan.extra_groups == 2
    assert plan.limited_by == ('sample_budget', )


def test_budget_controller_enforces_token_and_per_round_caps() -> None:
    controller = SamplingBudgetController(
        SamplingBudgetConfig(
            max_extra_groups=10,
            max_extra_groups_per_round=3,
            max_resample_rounds=3,
            max_total_tokens=200,
        ))
    state = SamplingBudgetState()
    controller.observe_round(
        state,
        [_decision(False)] * 4,
        num_generations=2,
        generated_tokens=120,
    )
    plan = controller.plan_resampling(
        state,
        target_groups=4,
        num_generations=2,
        estimated_tokens_per_group=40,
    )
    assert plan.extra_groups == 2
    assert plan.limited_by == ('token_budget', )


def test_default_budget_disables_resampling() -> None:
    controller = SamplingBudgetController()
    state = SamplingBudgetState()
    controller.observe_round(state, [_decision(False)], num_generations=4)
    plan = controller.plan_resampling(state, target_groups=1, num_generations=4)
    assert plan.extra_groups == 0
    assert plan.exhausted
    assert plan.limited_by == ('resample_round_budget', )


def test_admission_metrics_report_both_rejection_signals() -> None:
    reward_reject = _decision(False)
    near_tie_reject = GroupAdmissionPolicy(
        GroupAdmissionConfig(min_reward_range=0.02)).evaluate([0.5, 0.51])
    diversity_reject = GroupAdmissionPolicy(
        GroupAdmissionConfig(max_mean_pairwise_similarity=0.5, similarity_ngram_size=1)).evaluate(
            [0.0, 1.0], ['same', 'same'])
    metrics = group_admission_metrics(
        [_decision(True), reward_reject, near_tie_reject, diversity_reject])
    assert metrics == {
        'candidate_group_count': 4,
        'admitted_group_count': 1,
        'rejected_group_count': 3,
        'effective_group_rate': pytest.approx(1 / 4),
        'reward_rejected_group_count': 2,
        'near_duplicate_rejected_group_count': 1,
        'externally_rejected_group_count': 0,
        'exact_dead_group_count': 1,
        'near_tie_group_count': 1,
        'redundant_group_count': 1,
    }


def test_empty_admission_metrics_keep_new_counts_zero() -> None:
    metrics = group_admission_metrics([])
    assert metrics['candidate_group_count'] == 0
    assert metrics['exact_dead_group_count'] == 0
    assert metrics['near_tie_group_count'] == 0
    assert metrics['redundant_group_count'] == 0


def test_invalid_configs_and_group_inputs_fail_early() -> None:
    with pytest.raises(ValueError, match='min_reward_std'):
        GroupAdmissionConfig(min_reward_std=-1)
    with pytest.raises(ValueError, match='min_reward_range'):
        GroupAdmissionConfig(min_reward_range=-1)
    with pytest.raises(ValueError, match='max_mean_pairwise_similarity'):
        GroupAdmissionConfig(max_mean_pairwise_similarity=1.1)
    with pytest.raises(ValueError, match='at least two rewards'):
        GroupAdmissionPolicy().evaluate([1.0])
    with pytest.raises(ValueError, match='finite'):
        GroupAdmissionPolicy().evaluate([0.0, float('nan')])
    with pytest.raises(ValueError, match='same complete group'):
        GroupAdmissionPolicy().evaluate([0.0, 1.0], ['only one completion'])
