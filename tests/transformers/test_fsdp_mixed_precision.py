# Copyright (c) ModelScope Contributors. All rights reserved.
from unittest.mock import MagicMock, patch

import pytest
import torch

from twinkle.model.transformers.strategy.mixed_precision import parse_mp_overrides
from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy, _build_ep_mp_policy, _build_mp_policy


def test_none_and_empty_config_resolve_to_no_overrides():
    assert parse_mp_overrides(None) == {}
    assert parse_mp_overrides({}) == {}


def test_partial_config_resolves_only_set_fields():
    assert parse_mp_overrides({'reduce_dtype': 'fp32'}) == {'reduce_dtype': torch.float32}


def test_full_config_string_values():
    config = {'param_dtype': 'bf16', 'reduce_dtype': 'fp32', 'output_dtype': 'fp16'}
    assert parse_mp_overrides(config) == {
        'param_dtype': torch.bfloat16,
        'reduce_dtype': torch.float32,
        'output_dtype': torch.float16,
    }


def test_torch_dtype_values_accepted():
    assert parse_mp_overrides({'param_dtype': torch.bfloat16}) == {'param_dtype': torch.bfloat16}


def test_string_values_case_insensitive():
    assert parse_mp_overrides({'reduce_dtype': 'FP32'}) == {'reduce_dtype': torch.float32}


def test_none_value_counts_as_unset():
    assert parse_mp_overrides({'param_dtype': 'bf16', 'reduce_dtype': None}) == {'param_dtype': torch.bfloat16}


@pytest.mark.parametrize('config', [
    {'param_dtpe': 'bf16'},  # typo'd key
    {'buffer_dtype': 'fp32'},  # FSDP1-only field, not valid on the FSDP2 face
])
def test_unknown_key_raises(config):
    with pytest.raises(ValueError, match='Unknown key'):
        parse_mp_overrides(config)


@pytest.mark.parametrize('value', ['fp64', 'float32', 32, torch.float64])
def test_invalid_value_raises(value):
    with pytest.raises(ValueError, match='reduce_dtype'):
        parse_mp_overrides({'reduce_dtype': value})


def test_non_dict_config_raises():
    with pytest.raises(ValueError, match='must be a dict'):
        parse_mp_overrides('bf16')


# ── native FSDP path: base string + overrides → MixedPrecisionPolicy ──────────


def test_native_policy_base_string_only_matches_legacy_behavior():
    policy = _build_mp_policy('bf16')
    assert (policy.param_dtype, policy.reduce_dtype, policy.output_dtype) == (torch.bfloat16, ) * 3
    assert policy.cast_forward_inputs

    policy = _build_mp_policy('fp16')
    assert (policy.param_dtype, policy.reduce_dtype, policy.output_dtype) == (torch.float16, ) * 3
    assert policy.cast_forward_inputs

    # 'no' must stay byte-identical to the legacy default-constructed policy.
    from torch.distributed.fsdp import MixedPrecisionPolicy
    assert _build_mp_policy('no') == MixedPrecisionPolicy()


def test_native_policy_override_replaces_only_configured_fields():
    policy = _build_mp_policy('bf16', parse_mp_overrides({'reduce_dtype': 'fp32'}))
    assert policy.param_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.float32
    assert policy.output_dtype is torch.bfloat16
    assert policy.cast_forward_inputs


def test_native_policy_override_on_no_base_keeps_unset_fields_none():
    policy = _build_mp_policy('no', parse_mp_overrides({'reduce_dtype': 'fp32'}))
    assert policy.param_dtype is None
    assert policy.reduce_dtype is torch.float32
    assert policy.output_dtype is None
    # cast_forward_inputs is inert while param_dtype is None (FSDP2 skips the cast).
    assert policy.cast_forward_inputs


def test_native_policy_param_override_applies_on_no_base():
    policy = _build_mp_policy('no', parse_mp_overrides({'param_dtype': 'fp16'}))
    assert policy.param_dtype is torch.float16
    assert policy.reduce_dtype is None
    assert policy.cast_forward_inputs


def test_ep_policy_still_forces_fp32_reduce_over_explicit_bf16():
    # NCCL PreMulSum constraint: the expert group must reduce in fp32 even when the
    # user explicitly asked for bf16 reduction on the dense side.
    base = _build_mp_policy('bf16', parse_mp_overrides({'reduce_dtype': 'bf16'}))
    ep_policy = _build_ep_mp_policy(base)
    assert ep_policy.reduce_dtype is torch.float32
    assert ep_policy.param_dtype is torch.bfloat16
    assert ep_policy.output_dtype is torch.bfloat16

    # A user-configured fp32 reduce passes through untouched.
    base = _build_mp_policy('bf16', parse_mp_overrides({'reduce_dtype': 'fp32'}))
    assert _build_ep_mp_policy(base).reduce_dtype is torch.float32


def test_strategy_parses_overrides_at_construction():
    strategy = NativeFSDPStrategy(
        mixed_precision='bf16',
        fsdp_config={'mixed_precision': {'reduce_dtype': 'fp32'}},
        enable_ep=False,
    )
    assert strategy._mp_overrides == {'reduce_dtype': torch.float32}


def test_strategy_rejects_invalid_overrides_at_construction():
    with pytest.raises(ValueError, match='Unknown key'):
        NativeFSDPStrategy(
            mixed_precision='bf16',
            fsdp_config={'mixed_precision': {'buffer_dtype': 'fp32'}},
            enable_ep=False,
        )


def test_wrap_model_passes_overridden_policy_to_fully_shard():
    strategy = NativeFSDPStrategy(
        device_mesh=MagicMock(),
        mixed_precision='bf16',
        fsdp_config={'mixed_precision': {'reduce_dtype': 'fp32'}},
        enable_ep=False,
    )
    model = torch.nn.Linear(4, 4)
    with patch('twinkle.model.transformers.strategy.native_fsdp._build_fsdp_mesh',
               return_value=MagicMock()), \
         patch('twinkle.model.transformers.strategy.native_fsdp.fully_shard') as fully_shard:
        strategy.wrap_model(model)

    fully_shard.assert_called_once()
    policy = fully_shard.call_args.kwargs['mp_policy']
    assert policy.param_dtype is torch.bfloat16
    assert policy.reduce_dtype is torch.float32
    assert policy.output_dtype is torch.bfloat16
    assert policy.cast_forward_inputs


# ── param/output dtype boundary guard (validated on 4x NPU: Qwen3.5 SDPA crash) ────


@pytest.mark.parametrize('mixed_precision,overrides', [
    ('bf16', {'param_dtype': 'fp32'}),  # output falls back to bf16: the 4x NPU crash config
    ('bf16', {'param_dtype': 'fp32', 'output_dtype': 'bf16'}),
    ('bf16', {'output_dtype': 'fp32'}),  # param stays bf16
    ('fp16', {'param_dtype': 'bf16'}),  # output falls back to fp16
    ('no', {'output_dtype': 'bf16'}),  # output cast active while param_dtype stays None
])
def test_strategy_rejects_param_output_dtype_boundary(mixed_precision, overrides):
    with pytest.raises(ValueError, match='output_dtype'):
        NativeFSDPStrategy(
            mixed_precision=mixed_precision,
            fsdp_config={'mixed_precision': overrides},
            enable_ep=False,
        )


@pytest.mark.parametrize('mixed_precision,overrides', [
    ('bf16', {}),  # legacy default
    ('bf16', {'reduce_dtype': 'fp32'}),  # the production case: BF16 compute, FP32 reduce
    ('bf16', {'param_dtype': 'fp32', 'reduce_dtype': 'fp32', 'output_dtype': 'fp32'}),  # uniform FP32
    ('no', {'reduce_dtype': 'fp32'}),  # effective param/output both None
    ('no', {'param_dtype': 'fp16'}),  # effective output None performs no cast: no boundary
])
def test_strategy_accepts_boundary_free_combinations(mixed_precision, overrides):
    strategy = NativeFSDPStrategy(
        mixed_precision=mixed_precision,
        fsdp_config={'mixed_precision': overrides},
        enable_ep=False,
    )
    policy = _build_mp_policy(mixed_precision, strategy._mp_overrides)
    assert policy.output_dtype is None or policy.output_dtype == policy.param_dtype
