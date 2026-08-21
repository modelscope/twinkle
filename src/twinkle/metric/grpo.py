# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from typing import Any, Dict, List, Optional, Union

from twinkle.data_format import InputFeature, ModelOutput
from twinkle.utils import get_logger
from twinkle.utils.transformers_utils import align_logps_to_mask
from .base import Metric

logger = get_logger()


class GRPOMetric(Metric):

    def __init__(
        self,
        device_mesh=None,
        process_group=None,
        ignore_index: int = -100,
        temperature: float = 1.0,
        epsilon: float = 0.2,
        epsilon_high: Optional[float] = None,
        top_k_kl: int = 0,
        **kwargs,
    ):
        super().__init__(device_mesh, process_group, **kwargs)
        self.has_old = None
        self.n_tokens = None
        self.sum_approx_kl = None
        self.sum_diff = None
        self.sum_old = None
        self.sum_new = None
        self.ignore_index = ignore_index
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)
        self.epsilon_high = float(epsilon_high) if epsilon_high is not None else float(epsilon)
        self.top_k_kl = int(top_k_kl)
        self.reset()

    def reset(self):
        self.sum_new: float = 0.0
        self.sum_old: float = 0.0
        self.sum_diff: float = 0.0
        self.sum_approx_kl: float = 0.0
        self.max_token_kl: float = 0.0
        self.max_token_ratio: float = 0.0
        self.n_tokens: int = 0
        self.has_old: bool = False
        self.sum_entropy: float = 0.0
        self.n_entropy_tokens: int = 0
        self.sum_clip_low: float = 0.0
        self.sum_clip_high: float = 0.0
        self.clip_n_total: float = 0.0
        self.high_kl_records: list = []
        self._gsi_cursor: int = 0
        # 异常 token 探针：logp 尾部统计 + 与采样端的对账。
        self.min_new_logp: float = 0.0
        self.n_logp_lt5: int = 0
        self.n_logp_lt10: int = 0
        self.sum_sampler_abs: float = 0.0
        self.max_sampler_abs: float = 0.0
        self.n_sampler_matched: int = 0
        self.n_sampler_given: int = 0

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

    def _collect_high_kl(
        self,
        d: 'torch.Tensor',
        kl_masked: 'torch.Tensor',
        labels: 'torch.Tensor',
        logps_f: 'torch.Tensor',
        old_f: 'torch.Tensor',
        gsi_base: int,
    ) -> None:
        import torch
        if kl_masked.numel() == 0:
            return
        flat = kl_masked.flatten()
        n_pos = int((flat > 0).sum().item())
        k = min(self.top_k_kl, n_pos)
        if k <= 0:
            return
        topk_vals, topk_idx = torch.topk(flat, k)
        seq_len = kl_masked.shape[-1]
        for j in range(k):
            kl_v = float(topk_vals[j].item())
            if kl_v <= 0:
                continue
            idx = int(topk_idx[j].item())
            i = idx // seq_len
            pos = idx % seq_len
            self.high_kl_records.append({
                'gsi': gsi_base + i,
                'pos': pos,
                'token_id': int(labels[i, pos].item()),
                'kl': kl_v,
                'ratio': float(torch.exp(d[i, pos]).item()),
                'logp_new': float(logps_f[i, pos].item()),
                'logp_old': float(old_f[i, pos].item()),
            })

    def _accumulate_mb(
        self,
        labels: 'torch.Tensor',
        logps: 'torch.Tensor',
        old_slice: Any,
        entropies: Optional['torch.Tensor'] = None,
        adv_slice: Any = None,
        gsi_base: int = 0,
        sampler_slice: Any = None,
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
        logps_raw = logps.float()
        logps_f = logps_raw * scale if (scale > 0.0 and scale != 1.0) else logps_raw
        mask_f = mask.float()

        self.n_tokens += n_tok
        self.sum_new += float((logps_f * mask_f).sum().item())

        cur_min = float(logps_raw.masked_fill(~mask, 0.0).min().item())
        if cur_min < self.min_new_logp:
            self.min_new_logp = cur_min
        self.n_logp_lt5 += int(((logps_raw < -5.0) & mask).sum().item())
        self.n_logp_lt10 += int(((logps_raw < -10.0) & mask).sum().item())

        if sampler_slice is not None:
            self._accumulate_sampler(logps_raw, sampler_slice, mask, mask_f)

        # Entropy is loss-type-agnostic; aligned to logps shape by the model forward.
        if entropies is not None and torch.is_tensor(entropies) and entropies.numel() > 0:
            ent_f = entropies.float()
            if ent_f.shape[-1] != mask_f.shape[-1]:
                m_ent = min(ent_f.shape[-1], mask_f.shape[-1])
                ent_f = ent_f[..., :m_ent]
                ent_mask = mask_f[..., :m_ent]
            else:
                ent_mask = mask_f
            if ent_f.shape[0] != ent_mask.shape[0]:
                n_ent = min(ent_f.shape[0], ent_mask.shape[0])
                ent_f = ent_f[:n_ent]
                ent_mask = ent_mask[:n_ent]
            self.sum_entropy += float((ent_f * ent_mask).sum().item())
            self.n_entropy_tokens += int(ent_mask.sum().item())

        if old_slice is None:
            return num_seq

        aligned = align_logps_to_mask(old_slice, mask, logps_f.dtype)
        if aligned is None:
            return num_seq
        old_f = aligned.float()
        if scale > 0.0 and scale != 1.0:
            old_f = old_f * scale

        d = logps_f - old_f  # new - old
        self.sum_old += float((old_f * mask_f).sum().item())
        self.sum_diff += float((d * mask_f).sum().item())

        # Schulman K3 estimator of KL(old || new):
        #   samples x ~ old,  r(x) = new(x) / old(x),
        #   k3 = r - 1 - log(r) = exp(new - old) - (new - old) - 1.
        kl = torch.exp(d) - d - 1.0
        kl_masked = kl * mask_f
        self.sum_approx_kl += float(kl_masked.sum().item())
        # Per-token extremes for collapse detection
        if kl_masked.numel() > 0:
            cur_max_kl = float(kl_masked.max().item())
            if cur_max_kl > self.max_token_kl:
                self.max_token_kl = cur_max_kl
            # Track ratio extremes
            ratio_masked = torch.exp(d) * mask_f
            cur_max_ratio = float(ratio_masked.max().item())
            if cur_max_ratio > self.max_token_ratio:
                self.max_token_ratio = cur_max_ratio
        self.has_old = True

        # Clip stats: gated by subclass (token-level / seq-level / unconditional).
        if adv_slice is not None:
            adv_aligned = align_logps_to_mask(adv_slice, mask, logps_f.dtype)
            if adv_aligned is not None:
                self._accumulate_clip(d, adv_aligned, mask, mask_f)
        if self.top_k_kl > 0:
            self._collect_high_kl(d, kl_masked, labels, logps_f, old_f, gsi_base)
        return num_seq

    def _accumulate_clip(
        self,
        log_ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        mask: 'torch.Tensor',
        mask_f: 'torch.Tensor',
    ) -> None:
        """Token-level PPO clip rate, gated by advantage sign (default GRPO)."""
        import torch
        ratio = torch.exp(log_ratio)
        is_low = (ratio < 1 - self.epsilon) & (advantages < 0)
        is_high = (ratio > 1 + self.epsilon_high) & (advantages > 0)
        self.sum_clip_low += float((is_low.float() * mask_f).sum().item())
        self.sum_clip_high += float((is_high.float() * mask_f).sum().item())
        self.clip_n_total += float(mask_f.sum().item())

    def _accumulate_sampler(
        self,
        logps_raw: 'torch.Tensor',
        sampler_slice: Any,
        mask: 'torch.Tensor',
        mask_f: 'torch.Tensor',
    ) -> None:
        smp = align_logps_to_mask(sampler_slice, mask, logps_raw.dtype)
        if smp is None:
            return
        rows = sampler_slice if isinstance(sampler_slice, (list, tuple)) else [sampler_slice]
        lens = []
        for row in rows:
            try:
                lens.append(int(len(row)))
            except TypeError:
                lens.append(1)
        self.n_sampler_given += sum(lens)
        cov = align_logps_to_mask([[1.0] * n for n in lens], mask, logps_raw.dtype)
        if cov is None:
            return
        valid = cov * mask_f
        diff_abs = (logps_raw - smp).abs() * valid
        self.sum_sampler_abs += float(diff_abs.sum().item())
        self.n_sampler_matched += int(valid.sum().item())
        if diff_abs.numel() > 0:
            cur = float(diff_abs.max().item())
            if cur > self.max_sampler_abs:
                self.max_sampler_abs = cur

    def accumulate(
        self,
        inputs: Union[InputFeature, List[InputFeature]],
        outputs: ModelOutput,
        *,
        old_logps: Any = None,
        advantages: Any = None,
        sampler_logps: Any = None,
        **kwargs,
    ):
        import torch
        if outputs is None:
            return
        assert 'logps' in outputs
        logps_val = outputs.get('logps')
        logps_list = self._as_mb_list(logps_val)
        ent_val = outputs.get('entropies') if isinstance(outputs, dict) else None
        ent_list = self._as_mb_list(ent_val)
        inputs_list = inputs if isinstance(inputs, list) else [inputs]

        if (torch.is_tensor(logps_val) and len(inputs_list) > 1
                and all(isinstance(i, dict) and i.get('labels') is not None for i in inputs_list)):
            label_tensors = [torch.as_tensor(i['labels']) for i in inputs_list]
            seq_lens = {t.shape[-1] for t in label_tensors}
            if len(seq_lens) == 1:
                merged = torch.cat(label_tensors, dim=0)
                inputs_list = [{'labels': merged}]
            else:
                logger.warning(f'GRPOMetric: logps is a single tensor but inputs_list has '
                               f'{len(inputs_list)} mb with mismatched seq_lens={sorted(seq_lens)}. '
                               f'Only mb[0] will be accumulated; check the model forward path.')

        flat_old: Optional[List] = None
        if old_logps is not None and isinstance(old_logps, (list, tuple)):
            flat_old = list(old_logps)
        flat_adv: Optional[List] = None
        if advantages is not None and isinstance(advantages, (list, tuple)):
            flat_adv = list(advantages)
        flat_sampler: Optional[List] = None
        if sampler_logps is not None and isinstance(sampler_logps, (list, tuple)):
            flat_sampler = list(sampler_logps)

        cursor = 0
        n_mb = min(len(inputs_list), len(logps_list))
        for mb_idx in range(n_mb):
            mb_input = inputs_list[mb_idx]
            if not isinstance(mb_input, dict):
                continue
            labels = mb_input.get('labels')
            if labels is None:
                continue

            labels = torch.as_tensor(labels)

            logps_mb = logps_list[mb_idx]
            ent_mb = ent_list[mb_idx] if ent_list is not None and mb_idx < len(ent_list) else None

            num_seq_est = (labels.shape[0] if labels.dim() >= 2 else 1)
            if flat_old is not None:
                old_slice = flat_old[cursor:cursor + num_seq_est]
            elif old_logps is not None and hasattr(old_logps, 'shape'):
                # Aligned tensor from a ref/old model forward. Its seq width is the max over
                # the WHOLE micro batch (padded before the dp split), while ``logps_mb`` is
                # padded only to this rank's own max — so old is routinely LONGER, and
                # requiring exact equality here threw away ratio/kl on most steps whenever the
                # longest sample of the micro batch lived on another rank. The loss never had
                # this problem (GRPOLoss._pad_and_align_to_batch has the full-sequence branch),
                # so the gradients were right all along and only the panel went blank.
                # align_logps_to_mask now shares that branch; accept anything it can align.
                import torch as _torch  # noqa: F811
                usable = (_torch.is_tensor(old_logps)
                          and old_logps.dim() == logps_mb.dim()
                          and old_logps.shape[0] == logps_mb.shape[0]
                          and old_logps.shape[-1] >= logps_mb.shape[-1])
                if usable:
                    old_slice = old_logps
                else:
                    if mb_idx == 0:
                        # Warn once per accumulate call (not per mb) to avoid log spam.
                        old_shape = tuple(old_logps.shape) if _torch.is_tensor(old_logps) else 'unknown'
                        logger.warning(f'GRPOMetric: old_logps shape {old_shape} cannot be aligned to '
                                       f'logps_mb shape {tuple(logps_mb.shape)} (row count must match and '
                                       f'seq width must be >=); ratio/kl metrics will be skipped for '
                                       f'this step.')
                    old_slice = None
            else:
                old_slice = None

            adv_mb = flat_adv[cursor:cursor + num_seq_est] if flat_adv is not None else None
            smp_mb = flat_sampler[cursor:cursor + num_seq_est] if flat_sampler is not None else None
            gsi_base = self._gsi_cursor
            advanced = self._accumulate_mb(labels, logps_mb, old_slice, ent_mb, adv_mb,
                                           gsi_base=gsi_base, sampler_slice=smp_mb)
            self._gsi_cursor += advanced
            cursor += advanced

    def calculate(self) -> Dict[str, Any]:
        import torch
        local = [{
            'sum_new': self.sum_new,
            'sum_old': self.sum_old,
            'sum_diff': self.sum_diff,
            'sum_kl': self.sum_approx_kl,
            'max_token_kl': self.max_token_kl,
            'max_token_ratio': self.max_token_ratio,
            'n': self.n_tokens,
            'has_old': self.has_old,
            'sum_entropy': self.sum_entropy,
            'n_entropy_tokens': self.n_entropy_tokens,
            'sum_clip_low': self.sum_clip_low,
            'sum_clip_high': self.sum_clip_high,
            'clip_n_total': self.clip_n_total,
            'min_new_logp': self.min_new_logp,
            'n_logp_lt5': self.n_logp_lt5,
            'n_logp_lt10': self.n_logp_lt10,
            'sum_sampler_abs': self.sum_sampler_abs,
            'max_sampler_abs': self.max_sampler_abs,
            'n_sampler_matched': self.n_sampler_matched,
            'n_sampler_given': self.n_sampler_given,
        }]
        all_results = self.gather_results(local)

        n_total = sum(r['n'] for r in all_results)
        if n_total == 0:
            self.reset()
            return {}

        sum_new = sum(r['sum_new'] for r in all_results)
        mean_new = sum_new / n_total

        results: Dict[str, Any] = {
            'train/policy_confidence': math.exp(mean_new),
            'train/mean_new_logp': mean_new,
            'train/n_trainable_tokens': n_total,
            'train/logp_min': min(r.get('min_new_logp', 0.0) for r in all_results),
            'train/logp_frac_lt_5': sum(r.get('n_logp_lt5', 0) for r in all_results) / n_total,
            'train/logp_frac_lt_10': sum(r.get('n_logp_lt10', 0) for r in all_results) / n_total,
        }
        if any(r['has_old'] for r in all_results):
            mean_old = sum(r['sum_old'] for r in all_results) / n_total
            mean_diff = sum(r['sum_diff'] for r in all_results) / n_total
            mean_kl = sum(r['sum_kl'] for r in all_results) / n_total
            global_max_kl = max(r['max_token_kl'] for r in all_results)
            global_max_ratio = max(r['max_token_ratio'] for r in all_results)
            results['train/mean_old_logp'] = mean_old
            results['train/logp_diff_mean'] = mean_diff
            results['train/approx_kl'] = mean_kl
            results['train/token_kl_max'] = global_max_kl
            results['train/token_ratio_max'] = global_max_ratio

        n_ent = sum(r.get('n_entropy_tokens', 0) for r in all_results)
        if n_ent > 0:
            results['train/entropy'] = sum(r.get('sum_entropy', 0.0) for r in all_results) / n_ent

        clip_n = sum(r.get('clip_n_total', 0.0) for r in all_results)
        if clip_n > 0:
            sum_low = sum(r.get('sum_clip_low', 0.0) for r in all_results)
            sum_high = sum(r.get('sum_clip_high', 0.0) for r in all_results)
            results['train/clip_ratio_low'] = sum_low / clip_n
            results['train/clip_ratio_high'] = sum_high / clip_n
            results['train/clip_ratio'] = (sum_low + sum_high) / clip_n

        # 采样端对账（只在调用方传了 sampler_logps 时出现）。两条都是断言型指标：
        # sampler_logp_mae 应在引擎精度量级（bf16 约 1e-2），sampler_token_delta 应恒为 0。
        n_smp = sum(r.get('n_sampler_matched', 0) for r in all_results)
        if n_smp > 0:
            results['train/sampler_logp_mae'] = sum(r.get('sum_sampler_abs', 0.0) for r in all_results) / n_smp
            results['train/sampler_logp_max_abs'] = max(r.get('max_sampler_abs', 0.0) for r in all_results)
            results['train/sampler_token_delta'] = n_total - sum(r.get('n_sampler_given', 0) for r in all_results)

        # Underscore-prefixed key bypasses swanlab numeric coercion; script can pop and consume.
        if self.high_kl_records:
            results['_high_kl_records'] = list(self.high_kl_records)

        self.reset()
        return results


class PPOMetric(GRPOMetric):
    """PPO policy metric; shares token-level ratio and clipping statistics with GRPO."""


class GSPOMetric(GRPOMetric):
    """GRPOMetric variant for GSPO: clip applies to per-sequence geometric-mean ratio."""

    def _accumulate_clip(
        self,
        log_ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        mask: 'torch.Tensor',
        mask_f: 'torch.Tensor',
    ) -> None:
        import torch
        seq_tok = mask_f.sum(-1).clamp(min=1.0)
        seq_log_ratio = (log_ratio * mask_f).sum(-1) / seq_tok
        seq_ratio = torch.exp(seq_log_ratio)
        # Recover per-sample scalar from the scattered [B,L] tensor: value lives only
        # at mask positions, so masked-mean reproduces the original advantage exactly.
        seq_adv = (advantages * mask_f).sum(-1) / seq_tok
        is_low = (seq_ratio < 1 - self.epsilon) & (seq_adv < 0)
        is_high = (seq_ratio > 1 + self.epsilon_high) & (seq_adv > 0)
        valid = (mask_f.sum(-1) > 0).float()
        self.sum_clip_low += float((is_low.float() * valid).sum().item())
        self.sum_clip_high += float((is_high.float() * valid).sum().item())
        self.clip_n_total += float(valid.sum().item())


class CISPOMetric(GRPOMetric):
    """GRPOMetric variant for CISPO: clip rate is unconditional on advantage sign."""

    def _accumulate_clip(
        self,
        log_ratio: 'torch.Tensor',
        advantages: 'torch.Tensor',
        mask: 'torch.Tensor',
        mask_f: 'torch.Tensor',
    ) -> None:
        import torch
        ratio = torch.exp(log_ratio)
        is_low = ratio < 1 - self.epsilon
        is_high = ratio > 1 + self.epsilon_high
        self.sum_clip_low += float((is_low.float() * mask_f).sum().item())
        self.sum_clip_high += float((is_high.float() * mask_f).sum().item())
        self.clip_n_total += float(mask_f.sum().item())
