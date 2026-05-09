# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from typing import Any, Dict, List, Optional, Union
from twinkle.data_format import InputFeature, ModelOutput
from .base import Metric


def _align_logps_to_mask(
    ragged: Any,
    mask: 'torch.Tensor',  # noqa: F821
    dtype: 'torch.dtype',  # noqa: F821
) -> Optional['torch.Tensor']:  # noqa: F821
    import torch

    device = mask.device
    batch_size, seq_len = mask.shape

    if isinstance(ragged, torch.Tensor):
        t = ragged.to(device=device, dtype=dtype)
        if t.shape == (batch_size, seq_len):
            return t
        # Fall through to the list path (row-wise scatter).
        ragged = [t[i] for i in range(min(batch_size, t.shape[0]))]

    if not isinstance(ragged, (list, tuple)):
        return None

    result = torch.zeros((batch_size, seq_len), dtype=dtype, device=device)
    for i, sample in enumerate(ragged):
        if i >= batch_size:
            break
        pos = mask[i].nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            continue
        if isinstance(sample, (int, float)):
            result[i, pos] = float(sample)
            continue
        vals = torch.as_tensor(sample, dtype=dtype, device=device).flatten()
        n = min(len(pos), int(vals.numel()))
        if n > 0:
            result[i, pos[:n]] = vals[:n]
    return result


class GRPOMetric(Metric):

    def __init__(
        self,
        device_mesh=None,
        process_group=None,
        ignore_index: int = -100,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(device_mesh, process_group, **kwargs)
        self.has_old = None
        self.n_tokens = None
        self.sum_approx_kl = None
        self.sum_diff = None
        self.sum_old = None
        self.sum_new_sq = None
        self.sum_new = None
        self.ignore_index = ignore_index
        self.temperature = float(temperature)
        self.reset()

    def reset(self):
        self.sum_new: float = 0.0
        self.sum_new_sq: float = 0.0
        self.sum_old: float = 0.0
        self.sum_diff: float = 0.0
        self.sum_approx_kl: float = 0.0
        self.n_tokens: int = 0
        self.has_old: bool = False

    @staticmethod
    def _as_mb_list(logps_val) -> Optional[List]:
        import torch
        if logps_val is None:
            return None
        if isinstance(logps_val, list):
            return logps_val or None
        if torch.is_tensor(logps_val):
            if logps_val.numel() == 0:
                return None
            return [logps_val]
        return None

    def _accumulate_mb(
        self,
        labels: 'torch.Tensor',
        logps: 'torch.Tensor',
        old_slice: Any,
    ) -> int:
        """Reduce one microbatch into ``self.sum_*`` counters.

        Returns ``labels.shape[0]`` so the caller can advance the
        ``old_logps`` slicing cursor even when the microbatch had zero
        generated tokens (e.g. fully-masked prompt-only batch).
        """
        import torch

        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        if not torch.is_tensor(logps) or logps.numel() == 0:
            return labels.shape[0]
        if labels.device != logps.device:
            labels = labels.to(logps.device)

        # Safety-align seq_len (SP / packed edge cases may leave a
        # small off-by-one between labels and logps within a mb).
        if logps.shape[-1] != labels.shape[-1]:
            m = min(logps.shape[-1], labels.shape[-1])
            logps = logps[..., :m]
            labels = labels[..., :m]
        # Safety-align num_seq (mb-local; normally matches exactly).
        if logps.shape[0] != labels.shape[0]:
            n = min(logps.shape[0], labels.shape[0])
            logps = logps[:n]
            labels = labels[:n]

        mask = (labels != self.ignore_index)
        n_tok = int(mask.sum().item())
        num_seq = labels.shape[0]
        if n_tok == 0:
            return num_seq

        # Recover T=1 log-probs if user told us the sampler temperature.
        # At T=1 this is a no-op (temperature field defaults to 1.0).
        # Rescaling keeps ``logp_diff`` / ``approx_kl`` unchanged because
        # both new and old logps receive the same multiplier.
        scale = self.temperature
        logps_f = logps.float()
        if scale != 1.0:
            logps_f = logps_f * scale
        mask_f = mask.float()

        self.n_tokens += n_tok
        self.sum_new += float((logps_f * mask_f).sum().item())
        self.sum_new_sq += float(((logps_f ** 2) * mask_f).sum().item())

        if old_slice is None:
            return num_seq

        aligned = _align_logps_to_mask(old_slice, mask, logps_f.dtype)
        if aligned is None:
            return num_seq
        old_f = aligned.float()
        if scale != 1.0:
            old_f = old_f * scale

        d = logps_f - old_f  # new - old
        self.sum_old += float((old_f * mask_f).sum().item())
        self.sum_diff += float((d * mask_f).sum().item())
        # Schulman K3 estimator of KL(old || new):
        #   samples x ~ old,  r(x) = new(x) / old(x),
        #   k3 = r - 1 - log(r) = exp(new - old) - (new - old) - 1.
        kl = torch.exp(d) - d - 1.0
        self.sum_approx_kl += float((kl * mask_f).sum().item())
        self.has_old = True
        return num_seq

    def accumulate(
        self,
        inputs: Union[InputFeature, List[InputFeature]],
        outputs: ModelOutput,
        *,
        old_logps: Any = None,
        **kwargs,
    ):
        import torch
        if outputs is None:
            return
        assert 'logps' in outputs
        logps_val = outputs.get('logps')
        logps_list = self._as_mb_list(logps_val)
        inputs_list = inputs if isinstance(inputs, list) else [inputs]

        if (torch.is_tensor(logps_val) and len(inputs_list) > 1
                and all(isinstance(i, dict) and i.get('labels') is not None
                        for i in inputs_list)):
            label_tensors = [torch.as_tensor(i['labels']) for i in inputs_list]
            seq_lens = {t.shape[-1] for t in label_tensors}
            if len(seq_lens) == 1:
                merged = torch.cat(label_tensors, dim=0)
                inputs_list = [{'labels': merged}]

        flat_old: Optional[List] = None
        if old_logps is not None and isinstance(old_logps, (list, tuple)):
            flat_old = list(old_logps)

        cursor = 0
        n_mb = min(len(inputs_list), len(logps_list))
        for mb_idx in range(n_mb):
            mb_input = inputs_list[mb_idx]
            if not isinstance(mb_input, dict):
                continue
            labels = mb_input.get('labels')
            if labels is None:
                continue
            import torch
            labels = torch.as_tensor(labels)

            logps_mb = logps_list[mb_idx]

            if flat_old is not None:
                num_seq_est = (labels.shape[0] if labels.dim() >= 2 else 1)
                old_slice = flat_old[cursor:cursor + num_seq_est]
            elif old_logps is not None and hasattr(old_logps, 'shape'):
                # Uncommon: aligned global tensor. Only honour when it
                # exactly matches the single-mb shape; otherwise drop.
                import torch as _torch  # noqa: F811
                old_slice = old_logps if (_torch.is_tensor(old_logps) and old_logps.shape
                                          == logps_mb.shape) else None
            else:
                old_slice = None

            advanced = self._accumulate_mb(labels, logps_mb, old_slice)
            cursor += advanced

    def calculate(self) -> Dict[str, Any]:
        local = [{
            'sum_new': self.sum_new,
            'sum_new_sq': self.sum_new_sq,
            'sum_old': self.sum_old,
            'sum_diff': self.sum_diff,
            'sum_kl': self.sum_approx_kl,
            'n': self.n_tokens,
            'has_old': self.has_old,
        }]
        all_results = self.gather_results(local)

        n_total = sum(r['n'] for r in all_results)
        if n_total == 0:
            self.reset()
            return {}

        sum_new = sum(r['sum_new'] for r in all_results)
        sum_new_sq = sum(r['sum_new_sq'] for r in all_results)
        mean_new = sum_new / n_total
        var_new = max(0.0, sum_new_sq / n_total - mean_new * mean_new)

        results: Dict[str, Any] = {
            'train/policy_confidence': f'{math.exp(mean_new):.4f}',
            'train/mean_new_logp': f'{mean_new:.4f}',
            'train/logp_std': f'{math.sqrt(var_new):.4f}',
        }
        if any(r['has_old'] for r in all_results):
            mean_old = sum(r['sum_old'] for r in all_results) / n_total
            mean_diff = sum(r['sum_diff'] for r in all_results) / n_total
            mean_kl = sum(r['sum_kl'] for r in all_results) / n_total
            results['train/mean_old_logp'] = f'{mean_old:.4f}'
            results['train/logp_diff_mean'] = f'{mean_diff:+.4f}'
            results['train/approx_kl'] = f'{mean_kl:.6f}'

        self.reset()
        return results
