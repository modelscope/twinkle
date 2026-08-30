# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for CrossTokenLoss (X-Token cross-tokenizer distillation loss)."""
import pytest
import torch

from twinkle.loss.cross_token import CrossTokenLoss, _PROJECTION_MATRIX_CACHE


# ---------------------------------------------------------------------------
# Fake tokenizer (duck-typed, no transformers dependency)
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """Minimal tokenizer with deterministic decode/encode for unit tests.

    Token text is ``"{prefix}_{id}"`` so two tokenizers sharing a prefix
    produce exact text matches on overlapping ids, while different prefixes
    force multi-token fallback mapping.
    """

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

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for length in range(min(len(text) - i, 20), 0, -1):
                candidate = text[i:i + length]
                if candidate in self._vocab:
                    ids.append(self._vocab[candidate])
                    i += length
                    matched = True
                    break
            if not matched:
                ids.append(0)
                i += 1
        return ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(batch=2, seq_s=6, seq_t=8, vocab_s=40, vocab_t=30):
    """Synthetic student/teacher logits + labels."""
    torch.manual_seed(0)
    student_logits = torch.randn(batch, seq_s, vocab_s)
    teacher_logits = torch.randn(batch, seq_t, vocab_t)
    labels = torch.randint(0, vocab_s, (batch, seq_s))
    labels[:, -2:] = -100
    student_input_ids = torch.randint(0, vocab_s, (batch, seq_s))
    teacher_input_ids = torch.randint(0, vocab_t, (batch, seq_t))
    return student_logits, teacher_logits, labels, student_input_ids, teacher_input_ids


def _build_loss(loss_type="pkl", **kwargs):
    student_tok = FakeTokenizer(40, prefix="tok")
    teacher_tok = FakeTokenizer(30, prefix="tok")
    defaults = dict(
        student_tokenizer=student_tok,
        teacher_tokenizer_group=[teacher_tok],
        max_length=3,
        vocab_topk=16,
        uncommon_topk=16,
        device=torch.device("cpu"),
    )
    defaults.update(kwargs)
    return CrossTokenLoss(loss_type=loss_type, **defaults)


# ---------------------------------------------------------------------------
# Tests: construction & validation
# ---------------------------------------------------------------------------

class TestCrossTokenLossInit:

    def test_invalid_loss_type_raises(self):
        with pytest.raises(ValueError, match="loss_type"):
            _build_loss(loss_type="xyz")

    def test_teacher_weights_length_mismatch(self):
        student_tok = FakeTokenizer(20, prefix="tok")
        t1 = FakeTokenizer(15, prefix="tok")
        t2 = FakeTokenizer(15, prefix="tok")
        with pytest.raises(ValueError, match="weights"):
            CrossTokenLoss(
                student_tokenizer=student_tok,
                teacher_tokenizer_group=[t1, t2],
                teacher_weights=[1.0],  # wrong length
                device=torch.device("cpu"),
            )

    def test_default_weights_sum_to_one(self):
        loss = _build_loss()
        assert sum(loss.teacher_weights) == pytest.approx(1.0)

    def test_custom_weights_normalized(self):
        loss = _build_loss(teacher_weights=[3.0])
        assert loss.teacher_weights[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Tests: P-KL mode
# ---------------------------------------------------------------------------

class TestCrossTokenLossPKL:

    def test_basic_forward(self):
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, t_logits, labels, s_ids, t_ids = _make_batch()
        result = loss_fn(
            {"labels": labels, "input_ids": s_ids},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
            teacher_input_ids_group=[t_ids],
        )
        assert result["loss"].dim() == 0
        assert torch.isfinite(result["loss"])
        assert result["loss"].item() >= 0

    def test_without_chunk_alignment(self):
        """No input_ids → fallback to min(seq_len) truncation."""
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, t_logits, labels, _, _ = _make_batch()
        result = loss_fn(
            {"labels": labels},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
        )
        assert torch.isfinite(result["loss"])

    def test_reverse_kl(self):
        loss_fn = _build_loss(loss_type="pkl", reverse_kl=True)
        s_logits, t_logits, labels, s_ids, t_ids = _make_batch()
        result = loss_fn(
            {"labels": labels, "input_ids": s_ids},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
            teacher_input_ids_group=[t_ids],
        )
        assert torch.isfinite(result["loss"])

    def test_dynamic_loss_scaling(self):
        loss_fn = _build_loss(loss_type="pkl", dynamic_loss_scaling=True)
        s_logits, t_logits, labels, s_ids, t_ids = _make_batch()
        result = loss_fn(
            {"labels": labels, "input_ids": s_ids},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
            teacher_input_ids_group=[t_ids],
        )
        assert torch.isfinite(result["loss"])

    def test_gradient_flow(self):
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, t_logits, labels, s_ids, t_ids = _make_batch()
        s_logits.requires_grad_(True)
        result = loss_fn(
            {"labels": labels, "input_ids": s_ids},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
            teacher_input_ids_group=[t_ids],
        )
        result["loss"].backward()
        assert s_logits.grad is not None
        assert s_logits.grad.shape == s_logits.shape

    def test_num_tokens_is_tensor(self):
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, t_logits, labels, s_ids, t_ids = _make_batch()
        result = loss_fn(
            {"labels": labels, "input_ids": s_ids},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
            teacher_input_ids_group=[t_ids],
        )
        assert isinstance(result["num_tokens"], torch.Tensor)
        assert result["num_tokens"].item() > 0


# ---------------------------------------------------------------------------
# Tests: H-KL mode
# ---------------------------------------------------------------------------

class TestCrossTokenLossHKL:

    def test_basic_forward(self):
        loss_fn = _build_loss(loss_type="hkl")
        s_logits, t_logits, labels, s_ids, t_ids = _make_batch()
        result = loss_fn(
            {"labels": labels, "input_ids": s_ids},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
            teacher_input_ids_group=[t_ids],
        )
        assert torch.isfinite(result["loss"])
        assert result["loss"].item() >= 0

    def test_without_chunk_alignment(self):
        loss_fn = _build_loss(loss_type="hkl")
        s_logits, t_logits, labels, _, _ = _make_batch()
        result = loss_fn(
            {"labels": labels},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
        )
        assert torch.isfinite(result["loss"])

    def test_gradient_flow(self):
        loss_fn = _build_loss(loss_type="hkl")
        s_logits, t_logits, labels, s_ids, t_ids = _make_batch()
        s_logits.requires_grad_(True)
        result = loss_fn(
            {"labels": labels, "input_ids": s_ids},
            {"logits": s_logits},
            teacher_logits_group=[t_logits],
            teacher_input_ids_group=[t_ids],
        )
        result["loss"].backward()
        assert s_logits.grad is not None


# ---------------------------------------------------------------------------
# Tests: teacher data formats
# ---------------------------------------------------------------------------

class TestCrossTokenLossTeacherFormats:

    def test_topk_logprobs_input(self):
        """vLLM-style top-k logprobs + indices → converted to full logits."""
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, _, labels, s_ids, t_ids = _make_batch()
        batch, seq_t, topk = 2, 8, 10
        teacher_topk_logprobs = torch.randn(batch, seq_t, topk)
        teacher_topk_indices = torch.randint(0, 30, (batch, seq_t, topk))
        result = loss_fn(
            {"labels": labels, "input_ids": s_ids},
            {"logits": s_logits},
            teacher_topk_logprobs_group=[teacher_topk_logprobs],
            teacher_topk_indices_group=[teacher_topk_indices],
            teacher_input_ids_group=[t_ids],
        )
        assert torch.isfinite(result["loss"])

    def test_no_teacher_raises(self):
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, _, labels, _, _ = _make_batch()
        with pytest.raises(ValueError, match="teacher"):
            loss_fn({"labels": labels}, {"logits": s_logits})

    def test_wrong_teacher_count_raises(self):
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, t_logits, labels, _, _ = _make_batch()
        with pytest.raises(ValueError, match="expected 1"):
            loss_fn(
                {"labels": labels},
                {"logits": s_logits},
                teacher_logits_group=[t_logits, t_logits],
            )

    def test_no_labels_raises(self):
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, t_logits, _, _, _ = _make_batch()
        with pytest.raises(ValueError, match="labels"):
            loss_fn({}, {"logits": s_logits}, teacher_logits_group=[t_logits])


# ---------------------------------------------------------------------------
# Tests: multi-teacher
# ---------------------------------------------------------------------------

class TestCrossTokenLossMultiTeacher:

    def test_two_teachers(self):
        student_tok = FakeTokenizer(30, prefix="tok")
        t1 = FakeTokenizer(25, prefix="tok")
        t2 = FakeTokenizer(20, prefix="tok")
        loss_fn = CrossTokenLoss(
            student_tokenizer=student_tok,
            teacher_tokenizer_group=[t1, t2],
            teacher_weights=[0.6, 0.4],
            vocab_topk=10,
            device=torch.device("cpu"),
        )
        torch.manual_seed(1)
        s_logits = torch.randn(2, 6, 30)
        t1_logits = torch.randn(2, 7, 25)
        t2_logits = torch.randn(2, 8, 20)
        labels = torch.randint(0, 30, (2, 6))
        labels[:, -1:] = -100
        result = loss_fn(
            {"labels": labels},
            {"logits": s_logits},
            teacher_logits_group=[t1_logits, t2_logits],
        )
        assert torch.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# Tests: projection matrix & statistics
# ---------------------------------------------------------------------------

class TestCrossTokenLossProjection:

    def test_mapping_statistics(self):
        loss_fn = _build_loss(loss_type="pkl")
        s_logits, t_logits, labels, _, _ = _make_batch()
        # Trigger lazy build
        loss_fn({"labels": labels}, {"logits": s_logits},
                teacher_logits_group=[t_logits])
        stats = loss_fn.get_mapping_statistics(0)
        assert "exact_matched" in stats
        assert "multi_token_matched" in stats
        assert "unmatched" in stats
        assert "sparsity" in stats
        assert stats["total_student_tokens"] == 40
        # Same prefix → overlap = min(40, 30) = 30 exact matches
        assert stats["exact_matched"] == 30

    def test_different_prefix_multi_token(self):
        """Different prefixes → no exact match, multi-token fallback."""
        student_tok = FakeTokenizer(20, prefix="s")
        teacher_tok = FakeTokenizer(15, prefix="t")
        loss_fn = CrossTokenLoss(
            student_tokenizer=student_tok,
            teacher_tokenizer_group=[teacher_tok],
            max_length=4,
            vocab_topk=10,
            device=torch.device("cpu"),
        )
        s_logits = torch.randn(1, 4, 20)
        t_logits = torch.randn(1, 5, 15)
        labels = torch.randint(0, 20, (1, 4))
        loss_fn({"labels": labels}, {"logits": s_logits},
                teacher_logits_group=[t_logits])
        stats = loss_fn.get_mapping_statistics(0)
        # No exact text match between "s_X" and "t_Y"
        assert stats["exact_matched"] == 0

    def test_projection_caching(self):
        """Second instance with same tokenizers should hit the cache."""
        loss1 = _build_loss(loss_type="pkl")
        loss2 = _build_loss(loss_type="pkl")
        # Same tokenizer config → same cache key
        assert loss1._generate_cache_key() == loss2._generate_cache_key()

        s_logits, t_logits, labels, _, _ = _make_batch()
        loss1({"labels": labels}, {"logits": s_logits},
              teacher_logits_group=[t_logits])
        cache_size = len(_PROJECTION_MATRIX_CACHE)

        loss2({"labels": labels}, {"logits": s_logits},
              teacher_logits_group=[t_logits])
        # Requires no new cache entry and still builds correctly
        assert len(_PROJECTION_MATRIX_CACHE) == cache_size
        assert loss2._projection_matrices_built
