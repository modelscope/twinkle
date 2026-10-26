# Copyright (c) ModelScope Contributors. All rights reserved.
"""Unit tests for gradient normalization + clipping (single-process standard path)."""
import pytest
import torch
from torch import nn

from twinkle.utils.grad_clip import normalize_and_clip_grad_norm


def _parameter(grad_value):
    parameter = nn.Parameter(torch.tensor([0.0]))
    parameter.grad = torch.tensor([float(grad_value)])
    return parameter


class TestNormalizeAndClipGradNorm:

    def test_grads_are_divided_by_num_tokens(self):
        parameters = [_parameter(3.0), _parameter(4.0)]
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=2, max_grad_norm=10.0, norm_type=2.0)
        # 3/2 and 4/2 -> L2 norm 2.5, below the clip bound.
        assert norm == pytest.approx(2.5)
        assert parameters[0].grad.item() == pytest.approx(1.5)
        assert parameters[1].grad.item() == pytest.approx(2.0)

    def test_norm_is_reported_before_clipping(self):
        parameters = [_parameter(3.0), _parameter(4.0)]
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=1, max_grad_norm=0.5, norm_type=2.0)
        assert norm == pytest.approx(5.0)

    def test_clipping_scales_grads_to_the_bound(self):
        parameters = [_parameter(3.0), _parameter(4.0)]
        normalize_and_clip_grad_norm(parameters, num_tokens=2, max_grad_norm=1.25, norm_type=2.0)
        # coefficient = 1.25 / 2.5 = 0.5
        assert parameters[0].grad.item() == pytest.approx(0.75)
        assert parameters[1].grad.item() == pytest.approx(1.0)

    def test_inf_norm_clips_by_max_abs(self):
        parameters = [_parameter(3.0), _parameter(4.0)]
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=2, max_grad_norm=10.0, norm_type=float('inf'))
        assert norm == pytest.approx(2.0)

    def test_non_positive_num_tokens_is_treated_as_one(self):
        parameters = [_parameter(4.0)]
        norm = normalize_and_clip_grad_norm(
            parameters, num_tokens=0, max_grad_norm=10.0, norm_type=2.0)
        assert norm == pytest.approx(4.0)

    def test_parameters_without_grads_are_skipped(self):
        plain = nn.Parameter(torch.tensor([1.0]))
        with_grad = _parameter(4.0)
        norm = normalize_and_clip_grad_norm(
            [plain, with_grad], num_tokens=2, max_grad_norm=10.0, norm_type=2.0)
        assert norm == pytest.approx(2.0)
        assert plain.grad is None

    def test_no_grads_at_all_returns_zero(self):
        plain = nn.Parameter(torch.tensor([1.0]))
        assert normalize_and_clip_grad_norm(
            [plain], num_tokens=2, max_grad_norm=1.0, norm_type=2.0) == 0.0
