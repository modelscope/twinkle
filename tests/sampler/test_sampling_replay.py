# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from types import ModuleType, SimpleNamespace

import pytest

from twinkle.data_format import SamplingMask
from twinkle.sampler.vllm_sampler.vllm_engine import (
    _copy_sampling_mask,
    _filter_engine_config,
    _set_sampling_replay_output_kind,
)


def test_copy_sampling_mask_converts_vllm_values_and_detaches_storage():
    source_token_ids = ["1", 3, 4]
    source_offsets = [0, 2, 3]
    copied = _copy_sampling_mask(
        SimpleNamespace(token_ids=source_token_ids, offsets=source_offsets),
        num_tokens=2,
        required=True,
    )

    assert copied == SamplingMask(token_ids=[1, 3, 4], offsets=[0, 2, 3])
    source_token_ids[0] = 99
    source_offsets[-1] = 99
    assert copied == SamplingMask(token_ids=[1, 3, 4], offsets=[0, 2, 3])


def test_missing_sampling_mask_is_only_allowed_when_replay_is_disabled():
    assert _copy_sampling_mask(None, num_tokens=2, required=False) is None
    with pytest.raises(RuntimeError, match="missing sampling mask"):
        _copy_sampling_mask(None, num_tokens=2, required=True)


@pytest.mark.parametrize(
    ("mask", "num_tokens", "message"),
    [
        (
            SimpleNamespace(token_ids=[1], offsets=[0, 1]),
            2,
            "1 rows for 2 sampled tokens",
        ),
        (SimpleNamespace(token_ids=[1], offsets=[1, 1]), 1, "invalid CSR endpoints"),
        (SimpleNamespace(token_ids=[1], offsets=[0, 0]), 1, "invalid CSR endpoints"),
        (
            SimpleNamespace(token_ids=[1, 2], offsets=[0, 2, 1]),
            2,
            "invalid CSR endpoints",
        ),
        (
            SimpleNamespace(token_ids=[1], offsets=[0, 0, 1]),
            2,
            "empty or invalid CSR row",
        ),
    ],
)
def test_copy_sampling_mask_rejects_invalid_csr(mask, num_tokens, message):
    with pytest.raises(RuntimeError, match=message):
        _copy_sampling_mask(mask, num_tokens=num_tokens, required=True)


def test_filter_engine_config_preserves_supported_replay_flag():
    filtered, invalid = _filter_engine_config(
        {"dtype": "bfloat16", "enable_return_sampling_mask": True, "unknown": 1},
        {"dtype", "enable_return_sampling_mask"},
        enable_sampling_replay=True,
    )

    assert filtered == {"dtype": "bfloat16", "enable_return_sampling_mask": True}
    assert invalid == {"unknown"}


def test_filter_engine_config_fails_fast_for_incompatible_vllm():
    with pytest.raises(
        RuntimeError, match="AsyncEngineArgs accepts enable_return_sampling_mask"
    ):
        _filter_engine_config(
            {"dtype": "bfloat16", "enable_return_sampling_mask": True},
            {"dtype"},
            enable_sampling_replay=True,
        )


def test_replay_forces_final_only_output_kind(monkeypatch):
    request_output_kind = SimpleNamespace(FINAL_ONLY=object())
    sampling_params_module = ModuleType("vllm.sampling_params")
    sampling_params_module.RequestOutputKind = request_output_kind
    vllm_module = ModuleType("vllm")
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params_module)
    params = SimpleNamespace(output_kind="unchanged")

    _set_sampling_replay_output_kind(params, enable_sampling_replay=False)
    assert params.output_kind == "unchanged"
    _set_sampling_replay_output_kind(params, enable_sampling_replay=True)
    assert params.output_kind is request_output_kind.FINAL_ONLY
