"""Unit tests for _patch_adapter entry assertions and default target_parameters."""
from unittest import mock

import pytest
from peft import LoraConfig


def test_lora_dropout_with_target_parameters_raises():
    from twinkle.model.transformers.transformers import TransformersModel
    from twinkle.model.transformers.strategy.native_fsdp import NativeFSDPStrategy

    cfg = LoraConfig(
        r=4,
        target_modules=['q_proj'],
        target_parameters=['mlp.experts.gate_up_proj'],
        lora_dropout=0.1,
    )
    instance = mock.MagicMock(spec=TransformersModel)
    instance._enable_expert_parallel = True
    instance.strategy = NativeFSDPStrategy(enable_ep=True)

    with pytest.raises(ValueError, match='ParamWrapper'):
        TransformersModel._validate_ep_lora_config(instance, cfg)


def test_no_target_parameters_autofilled():
    from twinkle.model.transformers.transformers import TransformersModel

    cfg = LoraConfig(r=4, target_modules='all-linear')
    cfg = TransformersModel._maybe_autofill_target_parameters(cfg, enable_ep=True)
    assert cfg.target_parameters == ['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']


def test_target_parameters_passthrough_when_set():
    from twinkle.model.transformers.transformers import TransformersModel

    cfg = LoraConfig(r=4, target_modules='all-linear', target_parameters=['custom.param'])
    cfg2 = TransformersModel._maybe_autofill_target_parameters(cfg, enable_ep=True)
    assert cfg2.target_parameters == ['custom.param']
