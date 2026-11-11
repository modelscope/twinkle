# Copyright (c) ModelScope Contributors. All rights reserved.
import copy

import pytest

from twinkle_agentic.rollout.multi_turn import RolloutSample
from twinkle_client.rollout import training_groups


def group(key='a'):
    return [RolloutSample(
        encoded={'input_ids': [1, 10], 'labels': [10, -100]}, response_token_ids=[10],
        rollout_logprobs=[-0.1], response_loss_mask=[1], orm_reward=[float(index), 0.5],
        group_key=key, extra={'trajectory_idx': index, 'adapter_uri': 'policy-0'},
    ) for index in range(2)]


def test_out_of_order_groups_preserve_member_order_and_sample_identity():
    a, b = group('a'), group('b')
    samples = [a[1], b[0], a[0], b[1]]
    before = copy.deepcopy(samples)
    result = training_groups(samples, num_generations=2)
    assert result == [[a[1], a[0]], b]
    assert result[0][0] is a[1]
    assert samples == before


def test_truncation_is_an_explicit_task_choice():
    samples = group()
    samples[0].truncated = True
    assert training_groups(samples, num_generations=2) == [samples]
    assert training_groups(samples, num_generations=2, drop_truncated=True) == []


@pytest.mark.parametrize('change', ['incomplete', 'version', 'uri'])
def test_untrainable_groups_do_not_discard_other_groups(change):
    a, b = group(), group('b')
    if change == 'incomplete':
        a.pop()
    elif change == 'version':
        a[0].policy_version = 1
    else:
        a[0].extra['adapter_uri'] = 'another-policy'
    assert training_groups(a + b, num_generations=2) == [b]


@pytest.mark.parametrize('change', ['duplicate', 'missing_index', 'excess', 'missing_key'])
def test_bad_group_identity_is_not_silently_trained(change):
    samples = group()
    if change == 'duplicate':
        samples[0].extra['trajectory_idx'] = 1
    elif change == 'missing_index':
        samples[0].extra.pop('trajectory_idx')
    elif change == 'excess':
        samples.append(copy.deepcopy(samples[0]))
    else:
        samples[0].group_key = None
    with pytest.raises(RuntimeError, match='group'):
        training_groups(samples, num_generations=2)


@pytest.mark.parametrize('field,value', [
    ('response_token_ids', []), ('rollout_logprobs', []), ('rollout_logprobs', [float('nan')]),
    ('rollout_logprobs', [float('inf')]), ('response_loss_mask', []), ('response_loss_mask', [2]),
    ('orm_reward', None), ('orm_reward', []), ('orm_reward', [1.0, float('nan')]),
])
def test_invalid_training_data_fails_before_any_group_is_returned(field, value):
    a, b = group(), group('b')
    setattr(b[0], field, value)
    with pytest.raises(RuntimeError, match='missing or invalid'):
        training_groups(a + b, num_generations=2)


@pytest.mark.parametrize('failure', ['error', 'abort', 'generation_error', 'infrastructure'])
def test_rollout_failure_survives_group_filtering_and_update_budgets(failure):
    a, b = group(), group('b')[:1]
    b[0].truncated = True
    b[0].rollout_infos = {'error': 'provider failed'} if failure == 'infrastructure' else {'stop_reason': failure}
    with pytest.raises(RuntimeError, match='rollout failed'):
        training_groups(a + b, num_generations=2, drop_truncated=True)


@pytest.mark.parametrize('count', [0, -1, True, 1.5])
def test_group_size_must_be_a_positive_integer(count):
    with pytest.raises(ValueError, match='positive integer'):
        training_groups([], num_generations=count)
