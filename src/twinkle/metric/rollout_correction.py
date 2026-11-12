# Copyright (c) ModelScope Contributors. All rights reserved.
"""How far the sampler's log-probs are from the trainer's on the same tokens."""
from typing import Dict, Optional, Sequence

import torch

from twinkle import remote_class, remote_function


@remote_class(execute='first')
class RolloutCorrection:
    """The ``rollout_correction/*`` diagnostics from a training-side and a sampler-side logp lane.

    Both lanes are per-sample response-token rows; a row pair is compared over its common length and
    a pair with a missing or empty side is skipped. ``kl`` is the direct estimator of
    ``KL(pi_rollout || pi_training)`` (a freshly synced sampler sits near 0); the ``*_ppl`` /
    ``log_ppl_*`` keys compare the per-sequence mean logps; ``chi2_token`` / ``chi2_seq`` are the
    token- and sequence-level chi-square divergences. Returns ``{}`` when no pair is comparable.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=False)
    def __call__(self,
                 training_logps: Sequence[Optional[Sequence[float]]],
                 rollout_logps: Sequence[Optional[Sequence[float]]],
                 **kwargs) -> Dict[str, float]:
        train_rows, rollout_rows = [], []
        for train, rollout in zip(training_logps, rollout_logps):
            if train is None or rollout is None:
                continue
            length = min(len(train), len(rollout))
            if length == 0:
                continue
            train_rows.append(torch.tensor([float(v) for v in train[:length]], dtype=torch.float32))
            rollout_rows.append(torch.tensor([float(v) for v in rollout[:length]], dtype=torch.float32))
        if not train_rows:
            return {}
        log_ratio = torch.cat(train_rows) - torch.cat(rollout_rows)
        clamped = torch.clamp(log_ratio, -20.0, 20.0)
        metrics: Dict[str, float] = {
            'kl': float((-log_ratio).mean()),
            'k3_kl': float((torch.exp(clamped) - clamped - 1.0).clamp(-10.0, 10.0).mean()),
            'chi2_token': float(torch.exp(clamped).square().mean() - 1.0),
        }
        train_seq_mean = torch.stack([row.mean() for row in train_rows])
        rollout_seq_mean = torch.stack([row.mean() for row in rollout_rows])
        metrics['training_log_ppl'] = float(-train_seq_mean.mean())
        metrics['rollout_log_ppl'] = float(-rollout_seq_mean.mean())
        metrics['training_ppl'] = float(torch.exp(-train_seq_mean).mean())
        metrics['rollout_ppl'] = float(torch.exp(-rollout_seq_mean).mean())
        log_ppl_diff = rollout_seq_mean - train_seq_mean
        metrics['log_ppl_diff'] = float(log_ppl_diff.mean())
        metrics['log_ppl_abs_diff'] = float(log_ppl_diff.abs().mean())
        metrics['log_ppl_diff_max'] = float(log_ppl_diff.max())
        metrics['log_ppl_diff_min'] = float(log_ppl_diff.min())
        metrics['ppl_ratio'] = float(torch.exp(log_ppl_diff).mean())
        # Sequence-level chi2: rho_geo = exp(mean over the sequence of log(training/rollout)).
        rho_geo = torch.exp(torch.clamp(train_seq_mean - rollout_seq_mean, -20.0, 20.0))
        metrics['chi2_seq'] = float(rho_geo.square().mean() - 1.0)
        return metrics
