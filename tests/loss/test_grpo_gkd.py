# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for GRPO family losses and GKD loss."""
import pytest
import torch
import torch.nn.functional as F

from twinkle.loss import GRPOLoss, GSPOLoss, SAPOLoss, CISPOLoss, BNPOLoss, DRGRPOLoss, GKDLoss


def _make_rl_batch(batch_size=4, seq_len=8, vocab_size=20):
    """Create a synthetic RL training batch with labels, logps, old_logps, ref_logps, advantages."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, seq_len, vocab_size)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    for i in range(batch_size):
        labels[i, seq_len // 2:] = -100

    loss_mask = (labels != -100)
    masked_labels = labels.clone()
    masked_labels[~loss_mask] = 0
    logps = F.log_softmax(logits, dim=-1).gather(-1, masked_labels.unsqueeze(-1)).squeeze(-1)

    old_logps = logps.detach() + torch.randn_like(logps) * 0.1
    ref_logps = logps.detach() + torch.randn_like(logps) * 0.05
    advantages = torch.randn(batch_size, 1)

    inputs = {'labels': labels}
    outputs = {'logps': logps, 'logits': logits}
    return inputs, outputs, old_logps, ref_logps, advantages


class TestGRPOLoss:

    def test_basic_grpo_loss(self):
        loss_fn = GRPOLoss(epsilon=0.2)
        inputs, outputs, old_logps, ref_logps, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages)
        assert isinstance(result, dict) and 'loss' in result
        assert result['loss'].dim() == 0
        assert torch.isfinite(result['loss'])

    def test_grpo_no_advantages_returns_zero(self):
        """Without advantages, should return a zero loss."""
        loss_fn = GRPOLoss()
        inputs, outputs, _, _, _ = _make_rl_batch()
        result = loss_fn(inputs, outputs, advantages=None)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-6)

    def test_grpo_no_old_logps_uses_current(self):
        """Without old_logps, uses current logps → ratio=1."""
        loss_fn = GRPOLoss(epsilon=0.2)
        inputs, outputs, _, ref_logps, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=None, ref_logps=ref_logps, advantages=advantages)
        assert torch.isfinite(result['loss'])

    def test_grpo_kl_penalty(self):
        """With beta > 0 and ref_logps, KL penalty is added."""
        loss_fn = GRPOLoss(epsilon=0.2, beta=0.01)
        inputs, outputs, old_logps, ref_logps, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages)
        assert torch.isfinite(result['loss'])

    def test_grpo_no_kl_when_no_ref(self):
        loss_fn = GRPOLoss(epsilon=0.2, beta=0.01)
        inputs, outputs, old_logps, _, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=old_logps, ref_logps=None, advantages=advantages)
        assert torch.isfinite(result['loss'])

    def test_grpo_from_logits(self):
        """When logps is None, should compute from logits."""
        loss_fn = GRPOLoss(epsilon=0.2)
        inputs, outputs, _, _, advantages = _make_rl_batch()
        outputs_no_logps = {'logits': outputs['logits']}
        result = loss_fn(inputs, outputs_no_logps, advantages=advantages)
        assert torch.isfinite(result['loss'])

    def test_grpo_list_advantages(self):
        """Advantages can be a list of floats."""
        loss_fn = GRPOLoss(epsilon=0.2)
        inputs, outputs, old_logps, _, _ = _make_rl_batch()
        adv_list = [1.0, -1.0, 0.5, -0.5]
        result = loss_fn(inputs, outputs, old_logps=old_logps, advantages=adv_list)
        assert torch.isfinite(result['loss'])

    def test_grpo_entropy_coef(self):
        loss_fn = GRPOLoss(epsilon=0.2, entropy_coef=0.01)
        inputs, outputs, old_logps, _, advantages = _make_rl_batch()
        outputs_with_ent = {**outputs, 'entropies': torch.rand(4, 8)}
        result = loss_fn(inputs, outputs_with_ent, old_logps=old_logps, advantages=advantages)
        assert torch.isfinite(result['loss'])

    def test_grpo_entropy_coef_requires_entropies(self):
        loss_fn = GRPOLoss(entropy_coef=0.01)
        inputs, outputs, old_logps, _, advantages = _make_rl_batch()
        with pytest.raises(AssertionError, match='entropies'):
            loss_fn(inputs, outputs, old_logps=old_logps, advantages=advantages)


class TestGSPOLoss:

    def test_basic_gspo(self):
        loss_fn = GSPOLoss(epsilon=0.2)
        inputs, outputs, old_logps, ref_logps, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages)
        assert isinstance(result, dict) and 'loss' in result
        assert torch.isfinite(result['loss'])


class TestSAPOLoss:

    def test_basic_sapo(self):
        loss_fn = SAPOLoss(epsilon=0.2, tau_pos=1.0, tau_neg=1.0)
        inputs, outputs, old_logps, ref_logps, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages)
        assert isinstance(result, dict) and 'loss' in result
        assert torch.isfinite(result['loss'])


class TestCISPOLoss:

    def test_basic_cispo(self):
        loss_fn = CISPOLoss(epsilon=0.2)
        inputs, outputs, old_logps, ref_logps, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages)
        assert isinstance(result, dict) and 'loss' in result
        assert torch.isfinite(result['loss'])


class TestBNPOLoss:

    def test_basic_bnpo(self):
        loss_fn = BNPOLoss(epsilon=0.2)
        inputs, outputs, old_logps, ref_logps, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages)
        assert isinstance(result, dict) and 'loss' in result
        assert torch.isfinite(result['loss'])


class TestDRGRPOLoss:

    def test_basic_dr_grpo(self):
        loss_fn = DRGRPOLoss(epsilon=0.2, max_completion_length=10)
        inputs, outputs, old_logps, ref_logps, advantages = _make_rl_batch()
        result = loss_fn(inputs, outputs, old_logps=old_logps, ref_logps=ref_logps, advantages=advantages)
        assert isinstance(result, dict) and 'loss' in result
        assert torch.isfinite(result['loss'])


class TestGKDLoss:

    def test_basic_gkd_full_teacher(self):
        """GKD with full-vocabulary teacher logits."""
        loss_fn = GKDLoss(beta=0.5, temperature=2.0, chunk_size=64)
        torch.manual_seed(42)
        batch, seq, vocab = 2, 4, 16
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        labels[:, seq // 2:] = -100

        inputs = {'labels': labels}
        outputs = {'logits': student_logits}
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert isinstance(result, dict) and 'loss' in result
        assert torch.isfinite(result['loss'])

    def test_gkd_topk_teacher(self):
        """GKD with top-k reduced teacher logits."""
        loss_fn = GKDLoss(beta=0.5, temperature=2.0, chunk_size=64, topk=5)
        torch.manual_seed(42)
        batch, seq, vocab = 2, 4, 16
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))
        labels[:, seq // 2:] = -100

        inputs = {'labels': labels}
        outputs = {'logits': student_logits}
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits, topk=5)
        assert torch.isfinite(result['loss'])

    def test_gkd_remote_api_teacher(self):
        """GKD with remote API teacher (topk logprobs + indices)."""
        loss_fn = GKDLoss(beta=0.5, temperature=2.0, chunk_size=64)
        torch.manual_seed(42)
        batch, seq, vocab, k = 2, 4, 16, 5
        student_logits = torch.randn(batch, seq, vocab)
        teacher_topk_indices = torch.randint(0, vocab, (batch, seq, k))
        teacher_topk_logprobs = torch.randn(batch, seq, k)
        labels = torch.randint(0, vocab, (batch, seq))
        labels[:, seq // 2:] = -100

        inputs = {'labels': labels}
        outputs = {'logits': student_logits}
        result = loss_fn(inputs, outputs,
                         teacher_topk_logprobs=teacher_topk_logprobs,
                         teacher_topk_indices=teacher_topk_indices)
        assert torch.isfinite(result['loss'])

    def test_gkd_forward_kl(self):
        """beta=0 → forward KL(S || T)."""
        loss_fn = GKDLoss(beta=0.0, temperature=1.0, chunk_size=64)
        torch.manual_seed(42)
        batch, seq, vocab = 2, 4, 16
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        result = loss_fn({'labels': labels}, {'logits': student_logits}, teacher_logits=teacher_logits)
        assert torch.isfinite(result['loss'])

    def test_gkd_reverse_kl(self):
        """beta=1 → reverse KL(T || S)."""
        loss_fn = GKDLoss(beta=1.0, temperature=1.0, chunk_size=64)
        torch.manual_seed(42)
        batch, seq, vocab = 2, 4, 16
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        result = loss_fn({'labels': labels}, {'logits': student_logits}, teacher_logits=teacher_logits)
        assert torch.isfinite(result['loss'])

    def test_gkd_invalid_beta(self):
        with pytest.raises(ValueError, match='beta'):
            GKDLoss(beta=-0.1)

    def test_gkd_invalid_temperature(self):
        with pytest.raises(ValueError, match='temperature'):
            GKDLoss(temperature=0)

    def test_gkd_no_teacher_raises(self):
        loss_fn = GKDLoss()
        labels = torch.randint(0, 10, (2, 4))
        with pytest.raises(AssertionError):
            loss_fn({'labels': labels}, {'logits': torch.randn(2, 4, 10)})

    def test_gkd_no_ignored_tokens(self):
        """When all tokens are valid."""
        loss_fn = GKDLoss(beta=0.5, temperature=2.0, chunk_size=64)
        torch.manual_seed(42)
        batch, seq, vocab = 2, 4, 16
        student_logits = torch.randn(batch, seq, vocab)
        teacher_logits = torch.randn(batch, seq, vocab)
        labels = torch.randint(0, vocab, (batch, seq))

        result = loss_fn({'labels': labels}, {'logits': student_logits}, teacher_logits=teacher_logits)
        assert torch.isfinite(result['loss'])
