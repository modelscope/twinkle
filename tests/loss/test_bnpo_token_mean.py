# Copyright (c) ModelScope Contributors. All rights reserved.
"""BNPO ``token_mean_scope`` semantics.

'micro' (the DEFAULT) reproduces verl/SEAM: ``masked_mean`` inside each micro-batch, then
an equal-weighted average across micro/dp groups (see verl/workers/actor/dp_actor.py --
``pg_loss = agg_loss(..., 'token-mean')`` per micro, then ``* 1/gradient_accumulation``
before ``backward()``). It is deliberately NOT split-invariant.

'global' is the strict, split-invariant token-mean. It is available but NOT the default:
with group-relative advantages the token-weighted mean does not cancel (it equals
-cov(len, A)/mean(len)), which on skill2lora E13 produced a ~100x stronger coherent
"emit fewer tokens" gradient than verl and collapsed the response length. See BNPOLoss's
docstring for the measurements.

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


def test_micro_matches_verl_equal_weighted_micro_means():
    ptl, mask = _fixture()
    loss = BNPOLoss(token_mean_scope='micro')
    whole = _combine(loss, ptl, mask, [[0, 1]])          # one group -> token-mean 0.625
    split = _combine(loss, ptl, mask, [[0], [1]])        # per-group means (1.0, 0.5) -> 0.75
    assert abs(whole - 0.625) < 1e-6
    assert abs(split - 0.75) < 1e-6                      # equal weight per micro, as verl does
    # Not split-invariant, by design: this is exactly verl's behaviour.
    assert abs(split - 0.625) > 1e-3


def test_global_reports_sum_reduction_for_display():
    """'global' returns a token SUM, so LossMetric must be told reduction='sum' or the
    logged loss is inflated by the token count."""
    assert BNPOLoss(token_mean_scope='global').reduction == 'sum'


if __name__ == '__main__':
    test_global_is_split_invariant()
    test_micro_matches_verl_equal_weighted_micro_means()
    test_global_reports_sum_reduction_for_display()
    print('OK: micro (default) == verl equal-weighted micro-means; global is split-invariant')
