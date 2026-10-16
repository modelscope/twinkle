# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for OPSDLoss (On-Policy Self-Distillation, arXiv:2601.18734)."""
import pytest
import torch
import torch.nn.functional as F

from twinkle.loss import OPSDLoss
from twinkle.loss import torch_loss_mapping


def _make_opsd_batch(batch_size=4, seq_len=8, vocab_size=20, gap=0.0):
    """Synthetic batch: student logps + teacher logps shifted by `gap` on valid tokens."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    for i in range(batch_size):
        labels[i, seq_len // 2:] = -100  # first half = response tokens, rest ignored

    loss_mask = (labels != -100)
    masked_labels = labels.clone()
    masked_labels[~loss_mask] = 0
    logps = F.log_softmax(logits, dim=-1).gather(-1, masked_labels.unsqueeze(-1)).squeeze(-1)
    teacher_logps = logps.detach() + gap

    inputs = {'labels': labels}
    outputs = {'logps': logps}
    return inputs, outputs, teacher_logps, loss_mask


class TestOPSDLoss:

    def test_basic_finite_scalar(self):
        loss_fn = OPSDLoss()
        inputs, outputs, teacher, _ = _make_opsd_batch(gap=0.3)
        result = loss_fn(inputs, outputs, teacher_logps=teacher)
        assert isinstance(result, dict) and 'loss' in result
        assert result['loss'].dim() == 0
        assert torch.isfinite(result['loss'])

    def test_zero_loss_when_teacher_equals_student(self):
        """k3 estimate exp(r) - r - 1 == 0 exactly when r == 0."""
        loss_fn = OPSDLoss()
        inputs, outputs, teacher, _ = _make_opsd_batch(gap=0.0)
        result = loss_fn(inputs, outputs, teacher_logps=teacher)
        assert torch.allclose(result['loss'], torch.tensor(0.0), atol=1e-6)

    def test_loss_positive_when_gap_nonzero(self):
        loss_fn = OPSDLoss()
        for gap in (0.5, -0.5):
            inputs, outputs, teacher, _ = _make_opsd_batch(gap=gap)
            result = loss_fn(inputs, outputs, teacher_logps=teacher)
            assert result['loss'].item() > 0.0

    def test_gradient_pulls_student_toward_teacher(self):
        """teacher logp higher (r>0) -> d(loss)/d(student_logp) < 0 -> SGD raises student logp."""
        logps = torch.zeros(1, 4, requires_grad=True)
        labels = torch.tensor([[1, 1, -100, -100]])
        teacher = torch.full((1, 4), 0.0)
        teacher[0, :2] = 0.7  # teacher more confident on the two valid tokens
        loss_fn = OPSDLoss()
        out = loss_fn({'labels': labels}, {'logps': logps}, teacher_logps=teacher)
        out['loss'].backward()
        # gradient on valid tokens must be negative (increase logps), zero on masked tokens
        assert (logps.grad[0, :2] < 0).all()
        assert torch.allclose(logps.grad[0, 2:], torch.zeros(2))

    def test_gradient_direction_flips_when_teacher_lower(self):
        logps = torch.zeros(1, 4, requires_grad=True)
        labels = torch.tensor([[1, 1, -100, -100]])
        teacher = torch.full((1, 4), -0.7)  # teacher LESS confident
        loss_fn = OPSDLoss()
        out = loss_fn({'labels': labels}, {'logps': logps}, teacher_logps=teacher)
        out['loss'].backward()
        assert (logps.grad[0, :2] > 0).all()  # SGD lowers student logp

    def test_masked_tokens_do_not_contribute(self):
        """Changing teacher values on ignored positions must not change the loss."""
        loss_fn = OPSDLoss()
        inputs, outputs, teacher, loss_mask = _make_opsd_batch(gap=0.3)
        r1 = loss_fn(inputs, outputs, teacher_logps=teacher.clone())
        teacher2 = teacher.clone()
        teacher2[~loss_mask] += 123.0
        r2 = loss_fn(inputs, outputs, teacher_logps=teacher2)
        assert torch.allclose(r1['loss'], r2['loss'])

    def test_response_only_ragged_list_form(self):
        """Teacher logps as ragged per-sample lists (response tokens only) must align to the mask.

        This is the production form: the teacher forward uses a DIFFERENT (rubric) prompt, so
        only the response-token log-probs are extracted and passed per sample."""
        loss_fn = OPSDLoss()
        inputs, outputs, teacher, loss_mask = _make_opsd_batch(gap=0.4)
        ragged = [teacher[i][loss_mask[i]].tolist() for i in range(teacher.shape[0])]
        r_full = loss_fn(inputs, outputs, teacher_logps=teacher)
        r_ragged = loss_fn(inputs, outputs, teacher_logps=ragged)
        assert torch.allclose(r_full['loss'], r_ragged['loss'], atol=1e-5)

    def test_ref_logps_channel_fallback(self):
        """teacher_logps may ride the existing ref_logps channel (zero new tensor plumbing)."""
        loss_fn = OPSDLoss()
        inputs, outputs, teacher, _ = _make_opsd_batch(gap=0.3)
        r_kw = loss_fn(inputs, outputs, teacher_logps=teacher)
        r_ref = loss_fn(inputs, outputs, ref_logps=teacher)
        assert torch.allclose(r_kw['loss'], r_ref['loss'])

    def test_no_teacher_returns_zero_flowing_loss(self):
        """No teacher -> zero loss that still flows through autograd (ref-only forwards)."""
        logps = torch.randn(2, 6, requires_grad=True)
        labels = torch.randint(0, 10, (2, 6))
        loss_fn = OPSDLoss()
        out = loss_fn({'labels': labels}, {'logps': logps})
        assert out['loss'].item() == 0.0
        out['loss'].backward()  # must not raise
        assert logps.grad is not None

    def test_clamp_guards_extreme_gap(self):
        loss_fn = OPSDLoss()
        inputs, outputs, teacher, _ = _make_opsd_batch(gap=50.0)
        result = loss_fn(inputs, outputs, teacher_logps=teacher)
        assert torch.isfinite(result['loss'])

    def test_registered_in_mapping(self):
        assert torch_loss_mapping.get('opsd') is OPSDLoss

    def test_requires_logps_not_logits(self):
        assert OPSDLoss.require_logps is True
        assert OPSDLoss.require_logits is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
