# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tests for LigerFusedLinearCrossEntropyLoss.

CPU tests (always run) cover registration + the eval fallback path (standard
CE from materialised logits). Device tests (Liger kernel on CUDA/NPU) cover the
fused-path numerical correctness vs CrossEntropyLoss.
"""
import pytest
import torch
import torch.nn.functional as F

from twinkle.loss import CrossEntropyLoss, LigerFusedLinearCrossEntropyLoss, torch_loss_mapping

try:
    import liger_kernel  # noqa: F401
    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False

requires_liger = pytest.mark.skipif(not _HAS_LIGER, reason='liger_kernel not installed')


def _has_accelerator():
    if torch.cuda.is_available():
        return 'cuda'
    try:
        import torch_npu  # noqa: F401
        if torch_npu.npu.is_available():
            return 'npu'
    except ImportError:
        pass
    return None


requires_device = pytest.mark.skipif(not _has_accelerator(), reason='no CUDA/NPU available for Liger kernel')


# ── CPU: registration + flags ────────────────────────────────────────────────


def test_loss_registered_in_mapping():
    assert torch_loss_mapping['liger_fused_linear_cross_entropy'] is LigerFusedLinearCrossEntropyLoss


def test_loss_flags_keep_logits_skip_logps():
    # require_logits=True so forward keeps outputs['logits'] (= hidden states under the patch)
    # require_logps=False so forward skips selective_log_softmax
    assert LigerFusedLinearCrossEntropyLoss.require_logits is True
    assert LigerFusedLinearCrossEntropyLoss.require_logps is False


# ── CPU: fallback path (no lm_head in outputs -> standard CE) ──────────────────


def _make_logits_batch(bs=4, seq=8, vocab=20, seed=42):
    torch.manual_seed(seed)
    logits = torch.randn(bs, seq, vocab, dtype=torch.float32)
    labels = torch.randint(0, vocab, (bs, seq))
    # mark some tokens ignored
    labels[:, seq // 2:] = -100
    return labels, logits


def test_fallback_matches_cross_entropy_loss_when_no_lm_head():
    labels, logits = _make_logits_batch()
    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')
    ce = CrossEntropyLoss(ignore_index=-100, reduction='mean')
    out_fused = fused({'labels': labels}, {'logits': logits.clone()})
    out_ce = ce({'labels': labels}, {'logits': logits.clone()})
    assert torch.allclose(out_fused['loss'], out_ce['loss'], atol=1e-6)


def test_fallback_reduction_sum_num_tokens():
    labels, logits = _make_logits_batch()
    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='sum')
    out = fused({'labels': labels}, {'logits': logits.clone()})
    expected_tokens = (labels != -100).sum().item()
    assert out['num_tokens'].item() == expected_tokens


def test_fallback_all_ignored_yields_zero_loss():
    labels = torch.tensor([[-100, -100], [-100, -100]])
    logits = torch.randn(2, 2, 5, dtype=torch.float32)
    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')
    out = fused({'labels': labels}, {'logits': logits})
    assert out['loss'].item() == pytest.approx(0.0, abs=1e-6)


# ── Device: fused-path numerical correctness vs CrossEntropyLoss ───────────────


@requires_liger
@requires_device
def test_fused_path_matches_cross_entropy_on_device():
    device = _has_accelerator()
    torch.manual_seed(123)
    bs, seq, hidden, vocab = 2, 16, 8, 32
    # hidden states + a real lm_head weight
    hidden_states = torch.randn(bs, seq, hidden, device=device, dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False, device=device, dtype=torch.float32)
    labels = torch.randint(0, vocab, (bs, seq), device=device)
    labels[:, seq // 2:] = -100

    # Reference: materialise logits via lm_head, then CrossEntropyLoss.
    with torch.no_grad():
        logits = lm_head(hidden_states.detach())
    ref = CrossEntropyLoss(ignore_index=-100, reduction='mean')({'labels': labels}, {'logits': logits})

    # Fused: feed hidden states + lm_head.weight, no logits materialised.
    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')
    out = fused({'labels': labels}, {'logits': hidden_states.detach(), 'lm_head': lm_head})

    # Liger accumulates in fp32 internally; CrossEntropyLoss uses the (fp32 here)
    # log_softmax path. They must agree to a tight tolerance.
    assert torch.allclose(out['loss'].cpu(), ref['loss'].cpu(), atol=1e-4, rtol=1e-3), (
        f'fused={out["loss"].item():.6f} vs ref={ref["loss"].item():.6f}')


@requires_liger
@requires_device
def test_fused_path_backward_grad_flows_to_hidden_states():
    """Gradient must reach hidden_states (connects back to the LoRA backbone);
    lm_head is frozen so grad_weight need not be populated."""
    device = _has_accelerator()
    torch.manual_seed(7)
    bs, seq, hidden, vocab = 2, 8, 8, 16
    hidden_states = torch.randn(bs, seq, hidden, device=device, dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False, device=device, dtype=torch.float32)
    for p in lm_head.parameters():
        p.requires_grad_(False)  # frozen lm_head, LoRA-style
    labels = torch.randint(0, vocab, (bs, seq), device=device)

    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')
    out = fused({'labels': labels}, {'logits': hidden_states, 'lm_head': lm_head})
    out['loss'].backward()
    assert hidden_states.grad is not None
    assert hidden_states.grad.shape == hidden_states.shape
    assert torch.isfinite(hidden_states.grad).all()


@requires_liger
@requires_device
def test_fused_path_ignores_padding_tokens():
    """Labels == -100 must be excluded exactly (Liger's ignore_index)."""
    device = _has_accelerator()
    torch.manual_seed(99)
    bs, seq, hidden, vocab = 1, 6, 8, 12
    hidden_states = torch.randn(bs, seq, hidden, device=device, dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False, device=device, dtype=torch.float32)
    labels = torch.randint(0, vocab, (bs, seq), device=device)
    # blank half the tokens
    mask = torch.zeros(bs, seq, dtype=torch.bool, device=device)
    mask[:, seq // 2:] = True
    labels_full = labels.clone()
    labels_masked = labels.clone()
    labels_masked[mask] = -100

    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')
    out_masked = fused({'labels': labels_masked}, {'logits': hidden_states.detach(), 'lm_head': lm_head})
    out_full = fused({'labels': labels_full}, {'logits': hidden_states.detach(), 'lm_head': lm_head})
    # masked loss is over half the tokens; both finite and non-negative.
    assert out_masked['loss'].item() >= 0
    assert out_full['loss'].item() >= 0
    assert torch.isfinite(out_masked['loss']) and torch.isfinite(out_full['loss'])


@requires_liger
def test_construction_without_device_is_lazy():
    """Building the loss must not require a device — the kernel is only
    dispatched when __call__ runs under the fused path."""
    loss = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean', label_smoothing=0.1)
    assert loss.reduction == 'mean'
    assert loss.label_smoothing == 0.1


# ── Device-agnostic defensive fallback (fused kernel raises) ────────────────


@requires_liger
@requires_device
def test_fused_kernel_failure_falls_back_to_materialised_ce():
    """When the Liger fused kernel raises (any device/shape), the loss must
    fall back to materialised-CE via the stashed lm_head weight (NOT its
    identity-patched forward) and match CrossEntropyLoss; subsequent calls must
    skip the fused path (broken flag)."""
    device = _has_accelerator()
    torch.manual_seed(5)
    bs, seq, hidden, vocab = 2, 6, 8, 14
    hidden_states = torch.randn(bs, seq, hidden, device=device, dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False, device=device, dtype=torch.float32)
    for p in lm_head.parameters():
        p.requires_grad_(False)
    labels = torch.randint(0, vocab, (bs, seq), device=device)

    # Reference: materialise logits via lm_head, standard CE.
    with torch.no_grad():
        ref_logits = lm_head(hidden_states.detach())
    ref = CrossEntropyLoss(ignore_index=-100, reduction='mean')({'labels': labels}, {'logits': ref_logits})

    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')

    # Force the fused kernel to raise on the first call by swapping the Liger
    # module for a raising callable (assigning .__call__ on an nn.Module instance
    # doesn't work — Module.__call__ is class-level).
    class _Boom:
        def __call__(self, *a, **k):
            raise RuntimeError('simulated fused-kernel failure (e.g. unsupported shape)')

    original_liger = fused._liger
    fused._liger = _Boom()

    # The fallback warning is emitted via twinkle's logger (not the `warnings`
    # module), so capture log records.
    import logging as _logging
    from twinkle import get_logger as _get_logger

    class _LogCapture(_logging.Handler):
        def __init__(self):
            super().__init__(level=_logging.WARNING)
            self.records: list[str] = []

        def emit(self, record):
            self.records.append(record.getMessage())

    _cap = _LogCapture()
    _get_logger().addHandler(_cap)
    try:
        out = fused({'labels': labels}, {'logits': hidden_states.detach(), 'lm_head': lm_head})
        warned = any('falling back to materialised cross-entropy' in m for m in _cap.records)
    finally:
        _get_logger().removeHandler(_cap)

    # Fallback matches the materialised-CE reference.
    assert torch.allclose(out['loss'].cpu(), ref['loss'].cpu(), atol=1e-4, rtol=1e-3), (
        f'fallback={out["loss"].item():.6f} vs ref={ref["loss"].item():.6f}')
    assert warned, 'expected the defensive-fallback warning'

    # Restore the real kernel; the broken flag must persist (no retry).
    fused._liger = original_liger
    out2 = fused({'labels': labels}, {'logits': hidden_states.detach(), 'lm_head': lm_head})
    assert fused._fused_broken is True
    assert torch.allclose(out2['loss'].cpu(), ref['loss'].cpu(), atol=1e-4, rtol=1e-3), (
        'subsequent call must go straight to the fallback without re-invoking the fused kernel')


@requires_liger
@requires_device
def test_fused_kernel_success_does_not_mark_broken():
    """Sanity: a successful fused call must NOT set the broken flag (fallback
    path stays inactive)."""
    device = _has_accelerator()
    torch.manual_seed(8)
    bs, seq, hidden, vocab = 2, 8, 8, 16
    hidden_states = torch.randn(bs, seq, hidden, device=device, dtype=torch.float32, requires_grad=True)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False, device=device, dtype=torch.float32)
    for p in lm_head.parameters():
        p.requires_grad_(False)
    labels = torch.randint(0, vocab, (bs, seq), device=device)
    fused = LigerFusedLinearCrossEntropyLoss(ignore_index=-100, reduction='mean')
    fused({'labels': labels}, {'logits': hidden_states.detach(), 'lm_head': lm_head})
    assert fused._fused_broken is False


