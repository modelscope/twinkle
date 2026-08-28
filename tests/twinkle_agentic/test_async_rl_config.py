from __future__ import annotations

import pytest

from twinkle_agentic.async_rl.pipeline import _reward_for_context
from twinkle_agentic.async_rl.utils import (
    TrainBatchConfig,
    build_native_fsdp_model_kwargs,
    configure_lora_lr_scheduler,
    resolve_context_learning_rate,
    resolve_context_lora_target_modules,
    resolve_context_loss_config,
    resolve_model_attention_implementation,
    resolve_sequence_parallel_size,
    sampler_data_parallel_size,
    validate_context_batch_config,
)


def test_sequence_parallel_size_must_divide_model_gpus():
    assert resolve_sequence_parallel_size(2, 1) == 1
    assert resolve_sequence_parallel_size(2, 2) == 2

    with pytest.raises(ValueError, match='must be divisible'):
        resolve_sequence_parallel_size(2, 3)


def test_padding_free_sequence_parallel_requires_flash_attention():
    assert resolve_model_attention_implementation(
        {'attn_implementation': 'flash_attention_2'},
        padding_free=True,
        sequence_parallel_size=2,
    ) == 'flash_attention_2'

    with pytest.raises(ValueError, match='model.attn_implementation'):
        resolve_model_attention_implementation({}, padding_free=True, sequence_parallel_size=2)


@pytest.mark.parametrize(
    ('sampler_gpus', 'sampler_tp', 'expected_dp'),
    [(8, 2, 4), (1, 1, 1)],
)
def test_sampler_data_parallel_size(sampler_gpus, sampler_tp, expected_dp):
    assert sampler_data_parallel_size(sampler_gpus, sampler_tp) == expected_dp


def test_sampler_parallelism_rejects_incomplete_tp_group():
    with pytest.raises(ValueError, match='must be divisible'):
        sampler_data_parallel_size(3, 2)


def test_lora_lr_scheduler_uses_shared_adapter_config():
    calls = []

    class Model:
        def set_lr_scheduler(self, scheduler_cls, **kwargs):
            calls.append((scheduler_cls, kwargs))

    configure_lora_lr_scheduler(
        Model(),
        'tenant_lora',
        {
            'lr_scheduler': {
                'cls': 'CosineAnnealingLR',
                'T_max': 2000,
                'eta_min': 0.0,
            },
        },
    )

    assert calls == [('CosineAnnealingLR', {
        'adapter_name': 'tenant_lora',
        'T_max': 2000,
        'eta_min': 0.0,
    })]


def test_context_learning_rate_overrides_global_default():
    assert resolve_context_learning_rate({'learning_rate': 5e-6}, {'learning_rate': 1e-6}) == pytest.approx(5e-6)
    assert resolve_context_learning_rate({}, {'learning_rate': 1e-6}) == pytest.approx(1e-6)


@pytest.mark.parametrize('value', [0, -1e-6, float('inf')])
def test_context_learning_rate_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='positive finite'):
        resolve_context_learning_rate({'learning_rate': value}, {'learning_rate': 1e-6})


def test_context_lora_target_modules_override_global_default():
    defaults = {'target_modules': 'all-linear'}

    assert resolve_context_lora_target_modules({}, defaults) == 'all-linear'
    assert resolve_context_lora_target_modules(
        {'lora': {'target_modules': ['q_proj', 'v_proj']}},
        defaults,
    ) == ['q_proj', 'v_proj']


@pytest.mark.parametrize('value', ['', [], [None], {'q_proj': True}])
def test_context_lora_target_modules_reject_invalid_values(value):
    with pytest.raises(ValueError, match='target_modules'):
        resolve_context_lora_target_modules(
            {'lora': {'target_modules': value}},
            {'target_modules': 'all-linear'},
        )


def test_context_loss_config_overrides_global_defaults():
    loss_cls, loss_kwargs = resolve_context_loss_config(
        {'loss': {'cls': 'GSPOLoss', 'epsilon_high': 0.3}},
        {'cls': 'GRPOLoss', 'epsilon': 0.2},
    )

    assert loss_cls == 'GSPOLoss'
    assert loss_kwargs == {'epsilon': 0.2, 'epsilon_high': 0.3}


def test_context_loss_config_uses_grpo_defaults():
    assert resolve_context_loss_config({}) == ('GRPOLoss', {'epsilon': 0.2})


def test_context_loss_config_rejects_empty_class_name():
    with pytest.raises(ValueError, match='loss.cls'):
        resolve_context_loss_config({'loss': {'cls': ''}})


def test_rl_model_kwargs_enforce_native_fsdp():
    assert build_native_fsdp_model_kwargs({}) == {
        'strategy': 'native_fsdp',
        'fsdp_config': {},
    }
    assert build_native_fsdp_model_kwargs({
        'strategy': 'native_fsdp',
        'fsdp_config': {'reshard_after_forward': False},
    }) == {
        'strategy': 'native_fsdp',
        'fsdp_config': {'reshard_after_forward': False},
    }
    with pytest.raises(ValueError, match='must be native_fsdp'):
        build_native_fsdp_model_kwargs({'strategy': 'accelerate'})


def test_reward_factory_loads_class_and_resolved_kwargs():
    reward = _reward_for_context(
        {
            'class_path': 'twinkle.reward.DAPOMathReward',
            'kwargs': {
                'max_response_length': 8192,
                'overlong_buffer_length': 4096,
                'overlong_penalty_factor': 1.0,
                'score_tail_chars': 300,
            },
        },
        context_key='tenant/run/adapter',
    )

    assert reward.max_response_length == 8192
    assert reward.overlong_buffer_length == 4096


def test_reward_factory_rejects_non_reward_class():
    with pytest.raises(TypeError, match='Reward subclass'):
        _reward_for_context(
            {'class_path': 'collections.Counter'},
            context_key='tenant/run/adapter',
        )


def test_context_batch_config_accepts_group_aligned_dp_batches():
    validate_context_batch_config(
        'tenant/run/adapter',
        rollout_groups=8,
        num_generations=4,
        train=TrainBatchConfig(mini_batch_size=8, micro_batch_size=2),
        sampler_dp=2,
        model_dp=2,
    )


def test_context_batch_config_allows_training_group_to_span_model_dp_ranks():
    validate_context_batch_config(
        'tenant/run/adapter',
        rollout_groups=8,
        num_generations=4,
        train=TrainBatchConfig(mini_batch_size=16, micro_batch_size=2),
        sampler_dp=1,
        model_dp=8,
    )


def test_context_batch_config_rejects_partition_tail_and_undersized_rank_batch():
    with pytest.raises(ValueError, match='complete prompt groups'):
        validate_context_batch_config(
            'tenant/run/adapter',
            rollout_groups=6,
            num_generations=4,
            train=TrainBatchConfig(mini_batch_size=6, micro_batch_size=1),
            sampler_dp=2,
            model_dp=2,
        )

    with pytest.raises(ValueError, match='per-rank train batch'):
        validate_context_batch_config(
            'tenant/run/adapter',
            rollout_groups=8,
            num_generations=2,
            train=TrainBatchConfig(mini_batch_size=2, micro_batch_size=2),
            sampler_dp=2,
            model_dp=2,
        )


def test_context_batch_config_requires_token_limit_for_dynamic_batching():
    with pytest.raises(ValueError, match='max_tokens_per_micro_batch'):
        validate_context_batch_config(
            'tenant/run/adapter',
            rollout_groups=8,
            num_generations=4,
            train=TrainBatchConfig(
                mini_batch_size=8,
                micro_batch_size=2,
                dynamic_batching=True,
            ),
            sampler_dp=1,
            model_dp=1,
        )
