# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for loss functions: CrossEntropy, ChunkedCrossEntropy, MSE."""
import pytest
import torch
import torch.nn.functional as F

from twinkle.loss import CrossEntropyLoss, ChunkedCrossEntropyLoss, MSELoss


class TestCrossEntropyLoss:

    def test_basic_ce_loss(self):
        loss_fn = CrossEntropyLoss()
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        result = loss_fn(inputs, outputs)
        assert isinstance(result, dict) and 'loss' in result
        assert result['loss'].dim() == 0  # scalar
        assert result['loss'].item() >= 0

    def test_ce_from_logps(self):
        loss_fn = CrossEntropyLoss()
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        inputs = {'labels': labels}
        outputs = {'logps': logps}
        result = loss_fn(inputs, outputs)
        assert result['loss'].item() >= 0

    def test_ce_with_ignore_index(self):
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        logits = torch.randn(4, 10)
        labels = torch.tensor([0, -100, 5, -100])
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        result = loss_fn(inputs, outputs)
        assert result['loss'].item() >= 0

    def test_ce_all_ignored(self):
        """When all labels are ignored, denominator clamps to 1, loss should be 0."""
        loss_fn = CrossEntropyLoss(ignore_index=-100)
        logits = torch.randn(4, 10)
        labels = torch.tensor([-100, -100, -100, -100])
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        result = loss_fn(inputs, outputs)
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-6)

    def test_ce_reduction_sum(self):
        loss_fn = CrossEntropyLoss(reduction='sum')
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        result = loss_fn(inputs, outputs)
        assert result['num_tokens'].item() > 0

    def test_ce_dft_weighting(self):
        loss_fn = CrossEntropyLoss(dft=True)
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        inputs = {'labels': labels}
        outputs = {'logits': logits}
        result = loss_fn(inputs, outputs)
        # DFT: -p*log(p) should always be non-negative
        assert result['loss'].item() >= 0

    def test_ce_logps_vs_logits_match(self):
        """Results from logits path and pre-computed logps path should match."""
        loss_fn = CrossEntropyLoss()
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))

        # From logits
        result_logits = loss_fn({'labels': labels}, {'logits': logits.clone()})

        # From logps
        logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        result_logps = loss_fn({'labels': labels}, {'logps': logps})

        assert torch.allclose(result_logits['loss'], result_logps['loss'], atol=1e-5)


class TestChunkedCrossEntropyLoss:

    def test_chunked_matches_standard(self):
        """Chunked CE should produce the same loss as standard CE."""
        torch.manual_seed(42)
        logits = torch.randn(8, 20)
        labels = torch.randint(0, 20, (8,))

        standard = CrossEntropyLoss()
        chunked = ChunkedCrossEntropyLoss(chunk_size=3)

        r_std = standard({'labels': labels}, {'logits': logits.clone()})
        r_chunked = chunked({'labels': labels.clone()}, {'logits': logits.clone()})

        assert torch.allclose(r_std['loss'], r_chunked['loss'], atol=1e-4)

    def test_chunked_with_ignore_index(self):
        chunked = ChunkedCrossEntropyLoss(chunk_size=2, ignore_index=-100)
        logits = torch.randn(6, 10)
        labels = torch.tensor([0, -100, 5, 3, -100, 1])
        result = chunked({'labels': labels}, {'logits': logits})
        assert result['loss'].item() >= 0

    def test_chunked_reduction_sum(self):
        chunked = ChunkedCrossEntropyLoss(chunk_size=4, reduction='sum')
        logits = torch.randn(8, 10)
        labels = torch.randint(0, 10, (8,))
        result = chunked({'labels': labels}, {'logits': logits})
        assert result['num_tokens'].item() > 0

    def test_chunked_from_logps(self):
        """Fast path: when logps is provided, should match standard CE logps path."""
        torch.manual_seed(42)
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        standard = CrossEntropyLoss()
        chunked = ChunkedCrossEntropyLoss(chunk_size=2)

        r_std = standard({'labels': labels}, {'logps': logps.clone()})
        r_chunked = chunked({'labels': labels.clone()}, {'logps': logps.clone()})

        assert torch.allclose(r_std['loss'], r_chunked['loss'], atol=1e-5)

    def test_chunked_dft(self):
        chunked = ChunkedCrossEntropyLoss(chunk_size=2, dft=True)
        logits = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        result = chunked({'labels': labels}, {'logits': logits})
        assert result['loss'].item() >= 0

    def test_chunked_invalid_chunk_size(self):
        with pytest.raises(AssertionError):
            ChunkedCrossEntropyLoss(chunk_size=0)

    def test_chunked_invalid_reduction(self):
        with pytest.raises(AssertionError):
            ChunkedCrossEntropyLoss(chunk_size=1, reduction='max')

    def test_chunked_gradient_flow(self):
        """Ensure gradients flow through the chunked autograd function."""
        chunked = ChunkedCrossEntropyLoss(chunk_size=2)
        logits = torch.randn(4, 10, requires_grad=True)
        labels = torch.randint(0, 10, (4,))
        result = chunked({'labels': labels}, {'logits': logits})
        result['loss'].backward()
        assert logits.grad is not None
        assert logits.grad.shape == logits.shape


class TestMSELoss:

    def test_basic_mse(self):
        loss_fn = MSELoss()
        preds = torch.randn(4, 3)
        labels = torch.randn(4, 3)
        result = loss_fn({'labels': labels}, {'logits': preds})
        assert isinstance(result, dict) and 'loss' in result
        assert result['loss'].item() >= 0

    def test_mse_zero_when_equal(self):
        loss_fn = MSELoss()
        vals = torch.randn(4, 3)
        result = loss_fn({'labels': vals}, {'logits': vals})
        assert result['loss'].item() == pytest.approx(0.0, abs=1e-6)

    def test_mse_known_value(self):
        loss_fn = MSELoss()
        preds = torch.tensor([[1.0, 2.0]])
        labels = torch.tensor([[3.0, 5.0]])
        result = loss_fn({'labels': labels}, {'logits': preds})
        expected = ((1 - 3) ** 2 + (2 - 5) ** 2) / 2  # = 6.5
        assert result['loss'].item() == pytest.approx(expected, abs=1e-5)
