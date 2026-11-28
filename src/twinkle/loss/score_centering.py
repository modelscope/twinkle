# Copyright (c) ModelScope Contributors. All rights reserved.
"""Score centering for training-inference mismatch (arXiv:2609.20807)."""
from typing import Optional

import torch

TOPK_LOG_PROB_CHUNK_SIZE = 4096


def vocab_parallel_group() -> Optional[object]:
    """The Megatron TP group sharding these logits, or None for full-vocab backends.

    Same provenance rule as ``selective_log_softmax``: only a Megatron model's own TP state
    describes vocab shards; callers that know their logits are full-vocab pass a group explicitly.
    """
    try:
        from megatron.core import parallel_state
        if parallel_state.model_parallel_is_initialized() and parallel_state.get_tensor_model_parallel_world_size(
        ) > 1:
            return parallel_state.get_tensor_model_parallel_group()
    except Exception:
        return None
    return None


def _vocab_start(vocab_size: int, group) -> int:
    if group is None:
        return 0
    return torch.distributed.get_rank(group) * vocab_size


def _all_reduce(value: torch.Tensor, op, group) -> None:
    if group is not None and torch.distributed.get_world_size(group) > 1:
        torch.distributed.all_reduce(value, op=op, group=group)


class _TopKLogProbsFromLogits(torch.autograd.Function):
    """``log_softmax(logits).gather(ids)`` with a bounded fp32 workspace.

    Under vocab parallelism each rank holds a contiguous vocab shard: the row maximum and the
    normalizer are all-reduced over the group, and ids outside the local shard contribute zero.
    """

    @staticmethod
    def forward(ctx, logits, token_ids, chunk_size, group):
        n, vocab_size = logits.shape
        vocab_start = _vocab_start(vocab_size, group)
        chunk_size = chunk_size if chunk_size > 0 else max(n, 1)
        output = torch.empty(token_ids.shape, device=logits.device, dtype=torch.float32)
        maxima = torch.empty((n, 1), device=logits.device, dtype=torch.float32)
        denominators = torch.empty_like(maxima)
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            work = logits[start:end].to(dtype=torch.float32, copy=True)
            maximum = work.max(dim=-1, keepdim=True).values
            _all_reduce(maximum, torch.distributed.ReduceOp.MAX, group)
            local_ids = token_ids[start:end].long() - vocab_start
            local_mask = (local_ids >= 0) & (local_ids < vocab_size)
            targets = work.gather(-1, local_ids.clamp(0, vocab_size - 1)).masked_fill_(~local_mask, 0.0)
            _all_reduce(targets, torch.distributed.ReduceOp.SUM, group)
            work.sub_(maximum).exp_()
            denominator = work.sum(dim=-1, keepdim=True)
            _all_reduce(denominator, torch.distributed.ReduceOp.SUM, group)
            output[start:end] = (targets - maximum) - denominator.log()
            maxima[start:end] = maximum
            denominators[start:end] = denominator
            del work
        ctx.save_for_backward(logits, token_ids, maxima, denominators)
        ctx.chunk_size = chunk_size
        ctx.vocab_start = vocab_start
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        logits, token_ids, maxima, denominators = ctx.saved_tensors
        _, vocab_size = logits.shape
        grad_input = torch.empty_like(logits)
        for start in range(0, logits.shape[0], ctx.chunk_size):
            end = min(start + ctx.chunk_size, logits.shape[0])
            work = logits[start:end].to(dtype=torch.float32, copy=True)
            work.sub_(maxima[start:end]).exp_().div_(denominators[start:end])
            grad = grad_output[start:end].float()
            work.mul_(-grad.sum(dim=-1, keepdim=True))
            local_ids = token_ids[start:end].long() - ctx.vocab_start
            local_mask = (local_ids >= 0) & (local_ids < vocab_size)
            work.scatter_add_(-1, local_ids.clamp(0, vocab_size - 1), grad.masked_fill(~local_mask, 0.0))
            grad_input[start:end] = work
            del work
        # Every rank holds the same replicated output, so the gradient needs no all-reduce.
        return grad_input, None, None, None


def topk_log_probs_from_logits(logits: torch.Tensor,
                               token_ids: torch.Tensor,
                               chunk_size: int = TOPK_LOG_PROB_CHUNK_SIZE,
                               process_group=None) -> torch.Tensor:
    """Full-vocab-normalized log-probs of ``logits`` at ``token_ids``, chunked in fp32."""
    group = process_group if process_group is not None else vocab_parallel_group()
    return _TopKLogProbsFromLogits.apply(logits, token_ids.long(), chunk_size, group)


def align_rollout_topk(topk_ids, topk_logps, mask: torch.Tensor, expected_topk: int):
    """Flatten response-framed sampler heads onto the mask's trainable positions."""
    if topk_ids is None or topk_logps is None:
        raise ValueError('score_centering requires rollout_topk_ids and rollout_topk_logps from the sampler.')
    if len(topk_ids) != mask.shape[0] or len(topk_logps) != mask.shape[0]:
        raise ValueError('score_centering top-k batch size does not match the training batch.')
    aligned_ids, aligned_logps = [], []
    for index, (ids, logps) in enumerate(zip(topk_ids, topk_logps, strict=True)):
        n_pos = int(mask[index].sum().item())
        if len(ids) != n_pos or len(logps) != n_pos:
            raise ValueError(f'score_centering top-k/mask mismatch at sample {index}: expected {n_pos} rows, '
                             f'got ids={len(ids)}, logps={len(logps)}.')
        # The sampler head is ragged (k, or k+1 when the sampled token sits outside the top-k) and
        # is not sorted; keep the k most likely candidates so both sides compare the same ids.
        rows_ids, rows_logps = [], []
        for row_ids, row_logps in zip(ids, logps, strict=True):
            pairs = sorted(zip(map(int, row_ids), map(float, row_logps)), key=lambda pair: -pair[1])
            if len(pairs) < expected_topk:
                raise ValueError(f'score_centering needs >= {expected_topk} sampler head entries, '
                                 f'got {len(pairs)} at sample {index}.')
            rows_ids.append([token for token, _ in pairs[:expected_topk]])
            rows_logps.append([value for _, value in pairs[:expected_topk]])
        aligned_ids.append(torch.tensor(rows_ids, dtype=torch.long, device=mask.device))
        aligned_logps.append(torch.tensor(rows_logps, dtype=torch.float32, device=mask.device))
    return torch.cat(aligned_ids), torch.cat(aligned_logps)


def score_centering_correction(train_head_logps: torch.Tensor, sampler_head_logps: torch.Tensor, eps: float = 1e-6):
    """Additive REINFORCE centering term over the sampler head, plus both head masses."""
    train_head_logps = train_head_logps.float()
    with torch.no_grad():
        train_probs = train_head_logps.exp()
        sampler_probs = sampler_head_logps.float().exp()
        train_mass = train_probs.sum(-1)
        sampler_mass = sampler_probs.sum(-1)
        tail_ratio = (1 - sampler_mass).clamp_min(eps) / (1 - train_mass).clamp_min(eps)
        residual = sampler_probs - tail_ratio.unsqueeze(-1) * train_probs
    return (residual * train_head_logps).sum(-1), sampler_mass, train_mass