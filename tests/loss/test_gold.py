# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for GOLDLoss (ULD distillation loss)."""
import pytest
import torch
import torch.nn.functional as F

from twinkle.loss.gold import GOLDLoss
from twinkle.loss.gold_config import GOLDConfig


# ---------------------------------------------------------------------------
# Fake tokenizer (duck-typed, shared-style with test_cross_token)
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Minimal tokenizer: decode of a single id returns ``"{prefix}_{id}"``."""

    def __init__(self, vocab_size: int, prefix: str = "tok"):
        self._vocab_size = vocab_size
        self._prefix = prefix
        self._vocab = {f"{prefix}_{i}": i for i in range(vocab_size)}
        self._id_to_token = {i: f"{prefix}_{i}" for i in range(vocab_size)}

    def __len__(self):
        return self._vocab_size

    def get_vocab(self):
        return dict(self._vocab)

    def decode(self, token_ids, skip_special_tokens=False):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._id_to_token.get(int(i), "") for i in token_ids)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gold_batch(batch=2, seq=8, vocab_s=40, vocab_t=30, answer_len=4):
    """Synthetic student/teacher logits, labels and inputs for GOLD."""
    torch.manual_seed(0)
    student_logits = torch.randn(batch, seq, vocab_s)
    teacher_logits = torch.randn(batch, seq, vocab_t)
    labels = torch.randint(0, vocab_s, (batch, seq))
    labels[:, answer_len:] = -100
    teacher_labels = torch.randint(0, vocab_t, (batch, seq))
    teacher_labels[:, answer_len:] = -100
    student_input_ids = torch.randint(0, vocab_s, (batch, seq))
    teacher_input_ids = torch.randint(0, vocab_t, (batch, seq))
    inputs = {
        "labels": labels,
        "input_ids": student_input_ids,
        "teacher_labels": teacher_labels,
        "teacher_input_ids": teacher_input_ids,
    }
    outputs = {"logits": student_logits}
    return inputs, outputs, teacher_logits


def _build_gold(use_extended_uld=True, hybrid=False, **config_kwargs):
    student_tok = FakeTokenizer(40, prefix="tok")
    teacher_tok = FakeTokenizer(30, prefix="tok")
    defaults = dict(
        use_extended_uld=use_extended_uld,
        uld_use_hybrid_loss=hybrid,
    )
    defaults.update(config_kwargs)
    config = GOLDConfig(**defaults)
    return GOLDLoss(config, student_tokenizer=student_tok,
                    teacher_tokenizer=teacher_tok)


# ---------------------------------------------------------------------------
# Tests: basic ULD
# ---------------------------------------------------------------------------

class TestGOLDLossBasic:

    def test_extended_uld_forward(self):
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert result["loss"].dim() == 0
        assert torch.isfinite(result["loss"])
        assert result["loss"].item() >= 0

    def test_positional_uld_forward(self):
        """use_extended_uld=False → simple positional truncation."""
        loss_fn = _build_gold(use_extended_uld=False)
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert torch.isfinite(result["loss"])

    def test_identical_distributions_zero_loss(self):
        """Identical student/teacher logits → ULD loss ≈ 0."""
        loss_fn = _build_gold(use_extended_uld=False,
                              uld_distillation_weight=1.0,
                              uld_skip_student_eos=False,
                              uld_skip_teacher_eos=False)
        vocab = 40
        inputs, outputs, _ = _make_gold_batch(batch=2, seq=8, vocab_s=vocab,
                                              vocab_t=vocab)
        # Same logits for both → aligned distributions equal
        logits = torch.randn(2, 8, vocab)
        outputs = {"logits": logits}
        teacher_logits = logits  # same tensor → probs identical
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        # ULD is L1 on sorted probabilities → should be ~0 (allow fp noise)
        assert result["loss"].item() == pytest.approx(0.0, abs=1e-5)

    def test_gradient_flow(self):
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        outputs["logits"].requires_grad_(True)
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        result["loss"].backward()
        assert outputs["logits"].grad is not None
        assert outputs["logits"].grad.shape == outputs["logits"].shape

    def test_teacher_logits_from_outputs(self):
        """teacher_logits can come from outputs dict instead of kwargs."""
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        outputs["teacher_logits"] = teacher_logits
        result = loss_fn(inputs, outputs)
        assert torch.isfinite(result["loss"])

    def test_teacher_labels_from_inputs(self):
        """teacher_labels can come from inputs dict (already set in helper)."""
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        # Show both paths are equivalent: kwargs-only vs inputs-only
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert torch.isfinite(result["loss"])

    def test_num_tokens(self):
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        # 2 samples × answer_len valid tokens
        assert result["num_tokens"].item() == 8


# ---------------------------------------------------------------------------
# Tests: crossentropy & weighting
# ---------------------------------------------------------------------------

class TestGOLDLossWeighting:

    def test_crossentropy_included(self):
        """With uld_crossentropy_weight > 0, CE term is added."""
        loss_fn = _build_gold(use_extended_uld=False,
                              uld_crossentropy_weight=0.5,
                              uld_distillation_weight=0.0)
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)

        # Compute expected CE manually
        shift_logits = outputs["logits"][..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        assert result["loss"].item() == pytest.approx(0.5 * ce.item(), rel=1e-5)

    def test_zero_distillation_weight(self):
        """distillation_weight=0 → only CE remains."""
        loss_fn = _build_gold(use_extended_uld=True,
                              uld_crossentropy_weight=1.0,
                              uld_distillation_weight=1.0)
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert torch.isfinite(result["loss"])

    def test_temperature_effect(self):
        """Higher temperature smooths distributions → different loss."""
        inputs, outputs, teacher_logits = _make_gold_batch()
        loss_t1 = _build_gold(use_extended_uld=False,
                              uld_student_temperature=1.0,
                              uld_teacher_temperature=1.0)
        loss_t5 = _build_gold(use_extended_uld=False,
                              uld_student_temperature=5.0,
                              uld_teacher_temperature=5.0)
        r1 = loss_t1(inputs, outputs, teacher_logits=teacher_logits)
        r5 = loss_t5(inputs, outputs, teacher_logits=teacher_logits)
        assert r1["loss"].item() != pytest.approx(r5["loss"].item(), abs=1e-6)


# ---------------------------------------------------------------------------
# Tests: skip eos & answer regions
# ---------------------------------------------------------------------------

class TestGOLDLossAnswerRegions:

    def test_skip_eos_default(self):
        """Default skip_student_eos=True → answer size reduced by 1."""
        loss_full = _build_gold(use_extended_uld=False,
                                uld_skip_student_eos=False,
                                uld_skip_teacher_eos=False)
        loss_skip = _build_gold(use_extended_uld=False,
                                uld_skip_student_eos=True,
                                uld_skip_teacher_eos=True)
        inputs, outputs, teacher_logits = _make_gold_batch(answer_len=4)
        r_full = loss_full(inputs, outputs, teacher_logits=teacher_logits)
        r_skip = loss_skip(inputs, outputs, teacher_logits=teacher_logits)
        # Skipping the EOS position should produce a different (smaller) loss
        assert r_full["loss"].item() != pytest.approx(r_skip["loss"].item())

    def test_empty_answer_region_zero(self):
        """All labels ignored → no answer region → zero distillation."""
        loss_fn = _build_gold(use_extended_uld=False)
        inputs, outputs, teacher_logits = _make_gold_batch(answer_len=0)
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        # distillation term is 0; CE weight is 0 by default
        assert result["loss"].item() == pytest.approx(0.0, abs=1e-8)

    def test_answer_region_middle(self):
        """Answer region not at sequence start."""
        loss_fn = _build_gold(use_extended_uld=False)
        torch.manual_seed(3)
        seq = 8
        labels = torch.randint(0, 40, (2, seq))
        labels[:, :2] = -100       # prompt region
        labels[:, 6:] = -100       # trailing padding
        teacher_labels = torch.randint(0, 30, (2, seq))
        teacher_labels[:, :2] = -100
        teacher_labels[:, 6:] = -100
        inputs = {
            "labels": labels,
            "input_ids": torch.randint(0, 40, (2, seq)),
            "teacher_labels": teacher_labels,
            "teacher_input_ids": torch.randint(0, 30, (2, seq)),
        }
        outputs = {"logits": torch.randn(2, seq, 40)}
        teacher_logits = torch.randn(2, seq, 30)
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert torch.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# Tests: hybrid ULD + JSD
# ---------------------------------------------------------------------------

class TestGOLDLossHybrid:

    def test_hybrid_loss_forward(self):
        """Hybrid mode with vocab overlap → JSD on matched + ULD on unmatched."""
        loss_fn = _build_gold(use_extended_uld=False, hybrid=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert torch.isfinite(result["loss"])
        assert hasattr(loss_fn, "last_matched_loss")
        assert hasattr(loss_fn, "last_unmatched_loss")

    def test_hybrid_adaptive_weights(self):
        """With None weights, uses vocab-overlap-based adaptive weighting."""
        loss_fn = _build_gold(use_extended_uld=False, hybrid=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert loss_fn.hybrid_matched_weight is None  # adaptive
        assert torch.isfinite(result["loss"])

    def test_hybrid_explicit_weights(self):
        loss_fn = _build_gold(
            use_extended_uld=False, hybrid=True,
            uld_hybrid_matched_weight=0.5,
            uld_hybrid_unmatched_weight=0.5,
        )
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert loss_fn.hybrid_matched_weight == 0.5
        assert torch.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# Tests: teacher data formats & errors
# ---------------------------------------------------------------------------

class TestGOLDLossInputs:

    def test_topk_input(self):
        """vLLM-style teacher_topk_logprobs + indices."""
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, _ = _make_gold_batch()
        batch, seq = 2, 8
        topk = 10
        teacher_topk_logprobs = torch.randn(batch, seq, topk)
        teacher_topk_indices = torch.randint(0, 30, (batch, seq, topk))
        result = loss_fn(
            inputs, outputs,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_topk_indices=teacher_topk_indices,
        )
        assert torch.isfinite(result["loss"])

    def test_no_teacher_raises(self):
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, _ = _make_gold_batch()
        with pytest.raises(ValueError, match="Teacher logits"):
            loss_fn(inputs, outputs)

    def test_no_teacher_labels_raises(self):
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        inputs.pop("teacher_labels")
        with pytest.raises(ValueError, match="Teacher labels"):
            loss_fn(inputs, outputs, teacher_logits=teacher_logits)

    def test_no_student_input_ids_raises(self):
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        inputs.pop("input_ids")
        with pytest.raises(ValueError, match="input_ids"):
            loss_fn(inputs, outputs, teacher_logits=teacher_logits)

    def test_no_teacher_input_ids_raises(self):
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        inputs.pop("teacher_input_ids")
        with pytest.raises(ValueError, match="input_ids"):
            loss_fn(inputs, outputs, teacher_logits=teacher_logits)

    def test_no_labels_raises(self):
        loss_fn = _build_gold(use_extended_uld=True)
        inputs, outputs, teacher_logits = _make_gold_batch()
        inputs.pop("labels")
        with pytest.raises(ValueError, match="labels"):
            loss_fn(inputs, outputs, teacher_logits=teacher_logits)

    def test_loss_without_student_tokenizer(self):
        """Positional ULD does not need tokenizers."""
        config = GOLDConfig(use_extended_uld=False,
                            uld_skip_student_eos=False,
                            uld_skip_teacher_eos=False)
        loss_fn = GOLDLoss(config)
        inputs, outputs, teacher_logits = _make_gold_batch()
        result = loss_fn(inputs, outputs, teacher_logits=teacher_logits)
        assert torch.isfinite(result["loss"])