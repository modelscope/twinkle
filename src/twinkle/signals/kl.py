# Copyright (c) ModelScope Contributors. All rights reserved.
"""Per-token KL estimates between a policy and a reference logp lane."""
from typing import Any, List, Literal, Sequence

import torch

from twinkle import remote_class, remote_function

KINDS = ('k1', 'k2', 'k3')


@remote_class(execute='first')
class KLEstimator:
    """KL(policy || reference) from two aligned per-token logp lanes (Schulman's k1 / k2 / k3).

    With ``d = ref_logp - logp`` per token: ``k1 = -d``, ``k2 = d**2 / 2``, ``k3 = exp(d) - d - 1``.
    ``reduce='mean'`` returns one sequence value per row (an empty row reads 0.0); ``'none'`` returns
    the per-token rows.
    """

    @remote_function(dispatch='all', collect='first', lazy_collect=True)
    def __call__(self,
                 logps: Sequence[Sequence[float]],
                 ref_logps: Sequence[Sequence[float]],
                 *,
                 kind: Literal['k1', 'k2', 'k3'] = 'k3',
                 reduce: Literal['mean', 'none'] = 'mean',
                 **kwargs) -> List[Any]:
        if kind not in KINDS:
            raise ValueError(f'kl kind {kind!r} not in {KINDS}.')
        if reduce not in ('mean', 'none'):
            raise ValueError(f"kl reduce {reduce!r} not in ('mean', 'none').")
        if len(logps) != len(ref_logps):
            raise ValueError(f'{len(logps)} policy rows vs {len(ref_logps)} reference rows.')
        out: List[Any] = []
        for index, (policy, reference) in enumerate(zip(logps, ref_logps)):
            if len(policy) != len(reference):
                raise RuntimeError(f'row {index}: {len(policy)} policy logps vs {len(reference)} reference logps; '
                                   'the two lanes must score the same response tokens.')
            delta = torch.tensor(list(reference), dtype=torch.float32) - torch.tensor(list(policy), dtype=torch.float32)
            if kind == 'k1':
                kl = -delta
            elif kind == 'k2':
                kl = delta.square() / 2
            else:
                kl = torch.exp(delta) - delta - 1.0
            if reduce == 'mean':
                out.append(float(kl.mean()) if kl.numel() else 0.0)
            else:
                out.append(kl.tolist())
        return out
