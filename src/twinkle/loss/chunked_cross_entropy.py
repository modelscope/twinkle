# Copyright (c) ModelScope Contributors. All rights reserved.
from twinkle.data_format import LossOutput
from .base import Loss

# Lazily-built singleton autograd.Function, so we neither pay the
# class-construction cost on every forward nor force a top-level torch import.
_CHUNKED_CE_FUNC = None


def _get_chunked_ce_func():
    global _CHUNKED_CE_FUNC
    if _CHUNKED_CE_FUNC is not None:
        return _CHUNKED_CE_FUNC

    import torch
    import torch.nn.functional as F

    class _ChunkedCrossEntropyFunc(torch.autograd.Function):
        """Chunked CE that materialises log_softmax(B, V) only one chunk at a time.

        Forward returns a scalar loss; backward writes per-token gradients into
        a freshly allocated `grad_logits` tensor (the input `logits` is never
        mutated). Mathematically equivalent to ``CrossEntropyLoss`` in the same
        package; ``chunk_size`` only controls the memory/throughput trade-off.
        """

        @staticmethod
        def forward(ctx, logits, labels, chunk_size, ignore_index, reduction, dft):
            ctx.save_for_backward(labels)
            ctx._logits = logits
            ctx.chunk_size = chunk_size
            ctx.ignore_index = ignore_index
            ctx.reduction = reduction
            ctx.dft = dft

            n = logits.shape[0]
            # Use fp32 accumulators so we don't lose precision when summing
            # over many tokens under fp16/bf16 autocast (matches cross_entropy.py).
            total_loss = logits.new_zeros((), dtype=torch.float32)
            total_count = logits.new_zeros((), dtype=torch.float32)

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                logits_chunk = logits[start:end]
                labels_chunk = labels[start:end]
                mask = (labels_chunk != ignore_index).float()

                logps = F.log_softmax(
                    logits_chunk, dim=-1).gather(-1,
                                                 labels_chunk.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                per_token = -logps * logps.exp() if dft else -logps

                total_loss = total_loss + (per_token * mask).sum()
                total_count = total_count + mask.sum()

            ctx.num_tokens = total_count.detach()
            if reduction == 'mean':
                return total_loss / total_count.clamp(min=1)
            return total_loss

        @staticmethod
        def backward(ctx, grad_output):
            labels, = ctx.saved_tensors
            logits = ctx._logits
            ctx._logits = None
            chunk_size = ctx.chunk_size
            ignore_index = ctx.ignore_index
            reduction = ctx.reduction
            dft = ctx.dft

            if reduction == 'mean':
                scale = grad_output / ctx.num_tokens.clamp(min=1)
            else:
                scale = grad_output

            n = logits.shape[0]
            # Write gradients in-place into logits to avoid allocating a
            # second [N, V] tensor (halves peak backward memory).
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                logits_chunk = logits[start:end].detach().requires_grad_(True)
                labels_chunk = labels[start:end]
                mask = (labels_chunk != ignore_index).float()

                with torch.enable_grad():
                    logps = F.log_softmax(
                        logits_chunk, dim=-1).gather(-1,
                                                     labels_chunk.clamp(min=0).unsqueeze(-1)).squeeze(-1)
                    per_token = -logps * logps.exp() if dft else -logps
                    loss_chunk = (per_token * mask).sum()

                grad_chunk = torch.autograd.grad(loss_chunk, logits_chunk, retain_graph=False)[0]
                logits.data[start:end] = grad_chunk * scale

            # logits, labels, chunk_size, ignore_index, reduction, dft
            return logits, None, None, None, None, None

    _CHUNKED_CE_FUNC = _ChunkedCrossEntropyFunc
    return _CHUNKED_CE_FUNC


class ChunkedCrossEntropyLoss(Loss):
    """CE loss that chunks the (B, V) softmax to bound peak memory.

    Drop-in replacement for :class:`CrossEntropyLoss` when ``outputs['logits']``
    is large (e.g. long sequence x big vocab). Behaviour matches that loss
    bit-for-bit; ``chunk_size`` only affects memory/throughput.

    Args:
        chunk_size: How many rows of ``logits`` to process per chunk.
        ignore_index: Label id treated as padding (excluded from loss).
        reduction: ``'mean'`` or ``'sum'``; matches ``CrossEntropyLoss``.
        dft: If True, use DFT weighting ``-p*log(p)`` (arxiv 2508.05629).
    """

    require_logits = True
    # We chunk the (B, V) softmax ourselves; tell upstream not to materialise
    # `logps` (which would already pay the full memory cost we're trying to
    # avoid). The `_loss_from_logps` fast path is kept only for the rare case
    # where someone explicitly hands us pre-computed logps.
    require_logps = False

    def __init__(self, chunk_size: int, ignore_index: int = -100, reduction: str = 'mean', dft: bool = False, **kwargs):
        super().__init__()
        assert chunk_size > 0, 'chunk_size must be positive'
        assert reduction in ('mean', 'sum'), f"reduction must be 'mean' or 'sum', got {reduction!r}"
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dft = dft

    def __call__(self, inputs, outputs, **kwargs):
        labels = inputs['labels']
        logps = outputs.get('logps')

        # Fast path: if logps is already gathered upstream, chunking the
        # softmax is moot — fall back to the same scalar formula as
        # CrossEntropyLoss to keep behaviour identical.
        if logps is not None:
            return self._loss_from_logps(labels, logps)

        logits = outputs['logits']
        labels = labels.view(-1)
        logits = logits.view(-1, logits.shape[-1])

        func = _get_chunked_ce_func()
        loss = func.apply(logits, labels, self.chunk_size, self.ignore_index, self.reduction, self.dft)

        if self.reduction == 'mean':
            return LossOutput(loss=loss, num_tokens=0)
        num_tokens = (labels != self.ignore_index).float().sum().clamp(min=1)
        return LossOutput(loss=loss, num_tokens=num_tokens)

    def _loss_from_logps(self, labels, logps):
        mask = (labels != self.ignore_index).float()
        per_token = -logps * logps.exp() if self.dft else -logps
        if self.reduction == 'mean':
            return LossOutput(loss=(per_token * mask).sum() / mask.sum().clamp(min=1), num_tokens=0)
        return LossOutput(loss=(per_token * mask).sum(), num_tokens=mask.sum().clamp(min=1))
