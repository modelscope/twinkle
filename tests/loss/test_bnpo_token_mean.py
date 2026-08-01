# Copyright (c) ModelScope Contributors. All rights reserved.
"""BNPO token-mean fix: 'global' scope must be invariant to how a batch is split into
micro/dp groups (strict global token-mean); 'micro' scope reproduces the pre-fix
double-average bias that skews toward short responses.

The framework combines groups per LossOutput semantics (transformers.py / metric/loss.py):
  effective_loss = Σ_g loss_g / Σ_g num_tokens_g
so for scope='global' (loss_g = token sum, num_tokens_g = Σmask) this collapses to the
single-shot token-mean for ANY partition; for scope='micro' (loss_g = token-mean,
num_tokens_g = 0 -> treated as 1) it becomes the equal-weighted mean of per-group means.
"""
import torch

from twinkle.loss.grpo import BNPOLoss


def _combine(loss_fn, ptl, mask, groups):
    """Mimic the framework accumulation over `groups` (lists of row indices)."""
    tot_loss = 0.0
    tot_tok = 0.0
    for idx in groups:
        g_ptl, g_mask = ptl[idx], mask[idx]
        loss_g = loss_fn._aggregate_loss(g_ptl, g_mask)
        ntok = loss_fn._loss_num_tokens(g_mask)
        ntok = float(ntok if not torch.is_tensor(ntok) else ntok.item())
        if ntok <= 0:            # micro path: num_tokens=0 -> framework uses 1 per group
            ntok = 1.0
        tot_loss = tot_loss + loss_g
        tot_tok += ntok
    return float(tot_loss) / tot_tok


def _fixture():
    # two very different response lengths (short=2 tok, long=6 tok) -> maximally exposes bias
    ptl = torch.tensor([
        [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],   # short: mean per-token loss 1.0 over 2 tokens
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # long : mean per-token loss 0.5 over 6 tokens
    ])
    mask = torch.tensor([
        [1., 1., 0., 0., 0., 0.],
        [1., 1., 1., 1., 1., 1.],
    ])
    return ptl, mask


def test_global_is_split_invariant():
    ptl, mask = _fixture()
    loss = BNPOLoss(token_mean_scope='global')
    whole = _combine(loss, ptl, mask, [[0, 1]])
    split = _combine(loss, ptl, mask, [[0], [1]])
    true_token_mean = float((ptl * mask).sum() / mask.sum())  # (2*1 + 6*0.5)/8 = 0.625
    assert abs(whole - true_token_mean) < 1e-6
    assert abs(split - true_token_mean) < 1e-6           # <-- the fix: split == whole
    assert abs(whole - split) < 1e-6


def test_micro_is_biased_and_split_dependent():
    ptl, mask = _fixture()
    loss = BNPOLoss(token_mean_scope='micro')
    whole = _combine(loss, ptl, mask, [[0, 1]])          # one group -> token-mean 0.625
    split = _combine(loss, ptl, mask, [[0], [1]])        # per-group means (1.0, 0.5) -> 0.75
    assert abs(whole - 0.625) < 1e-6
    assert abs(split - 0.75) < 1e-6                       # short response over-weighted
    assert split > whole                                 # bias toward short is real
    # and it disagrees with the correct global answer
    assert abs(split - 0.625) > 1e-3


def test_seam_inherits_global_by_default():
    from twinkle.loss.grpo import SEAMBNPOLoss
    seam = SEAMBNPOLoss(epsilon=0.2, beta=0.001)
    assert seam.token_mean_scope == 'global'
    ptl, mask = _fixture()
    split = _combine(seam, ptl, mask, [[0], [1]])
    assert abs(split - 0.625) < 1e-6


if __name__ == '__main__':
    test_global_is_split_invariant()
    test_micro_is_biased_and_split_dependent()
    test_seam_inherits_global_by_default()
    print('OK: global split-invariant; micro reproduces the biased double-average')
