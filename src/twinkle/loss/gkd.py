# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Optional

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss

if TYPE_CHECKING:
    import torch

# Lazily-built singleton autograd.Function, mirroring chunked_cross_entropy.py: neither pay the
# class-construction cost on every forward nor force a top-level torch import.
_CHUNKED_JSD_FUNC = None


def _get_chunked_jsd_func():
    global _CHUNKED_JSD_FUNC
    if _CHUNKED_JSD_FUNC is not None:
        return _CHUNKED_JSD_FUNC

    import torch
    import torch.nn.functional as F

    def _chunk_jsd_sum(s_chunk, t_chunk, beta, beta_t, log_beta, log_1_minus_beta):
        """One chunk of the beta-JSD, summed. Identical ops in forward and backward recompute."""
        s_log_probs = F.log_softmax(s_chunk, dim=-1)
        t_log_probs = F.log_softmax(t_chunk, dim=-1)
        if beta == 0:
            # Forward KL: KL(T || S)
            jsd_chunk = F.kl_div(s_log_probs, t_log_probs, reduction='none', log_target=True)
        elif beta == 1:
            # Reverse KL: KL(S || T)
            jsd_chunk = F.kl_div(t_log_probs, s_log_probs, reduction='none', log_target=True)
        else:
            # Generalised JSD: β·KL(T||M) + (1-β)·KL(S||M)
            mixture_log_probs = torch.logsumexp(
                torch.stack([s_log_probs + log_1_minus_beta, t_log_probs + log_beta]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, t_log_probs, reduction='none', log_target=True)
            kl_student = F.kl_div(mixture_log_probs, s_log_probs, reduction='none', log_target=True)
            jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
        return jsd_chunk.sum()

    class _ChunkedJSDFunc(torch.autograd.Function):
        """Chunked JSD that keeps only ONE chunk's intermediates alive at a time.

        A plain chunk loop under autograd retains every chunk's saved tensors (log_softmax
        outputs, the stacked mixture input, the mixture itself) until backward, so peak memory
        still grows linearly with the number of valid tokens -- on long on-policy GKD rounds
        (thousands of response tokens x full vocab) that defeats the chunking and OOMs. Like
        ``ChunkedCrossEntropyLoss`` in this package, forward computes each chunk under no-grad
        and backward recomputes one chunk at a time, writing gradients into a fresh buffer.
        ``grad_outputs=scale`` reproduces the exact scalar the old single-graph backward
        propagated, so gradients match the naive loop bit-for-bit; ``chunk_size`` only controls
        the memory/throughput trade-off.
        """

        @staticmethod
        def forward(ctx, student_logits, teacher_logits, chunk_size, beta, num_valid):
            if torch.is_tensor(num_valid):
                ctx.save_for_backward(student_logits, teacher_logits, num_valid)
                ctx.num_valid = None
            else:
                ctx.save_for_backward(student_logits, teacher_logits)
                ctx.num_valid = num_valid
            ctx.chunk_size = chunk_size
            ctx.beta = beta

            n = student_logits.shape[0]
            if beta not in (0, 1):
                beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
                log_beta = torch.log(beta_t)
                log_1_minus_beta = torch.log1p(-beta_t)
            else:
                beta_t = log_beta = log_1_minus_beta = None

            total_loss = student_logits.new_zeros(())
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                total_loss = total_loss + _chunk_jsd_sum(
                    student_logits[start:end], teacher_logits[start:end], beta, beta_t, log_beta,
                    log_1_minus_beta)
            # Divide by the ORIGINAL num_valid object (int tensor under a label mask, python int
            # otherwise): tensor/tensor division casts the divisor to the logits dtype while
            # tensor/scalar keeps it exact -- switching forms would shift bf16 results by 1 ulp.
            return total_loss / num_valid

        @staticmethod
        def backward(ctx, grad_output):
            saved = ctx.saved_tensors
            student_logits, teacher_logits = saved[0], saved[1]
            num_valid = saved[2] if len(saved) > 2 else ctx.num_valid
            chunk_size = ctx.chunk_size
            beta = ctx.beta

            n = student_logits.shape[0]
            # Same scalar the single-graph backward propagated through the final division.
            scale = grad_output / num_valid
            need_s = student_logits.requires_grad
            need_t = teacher_logits.requires_grad
            if not (need_s or need_t):
                return None, None, None, None, None
            grad_student = torch.empty_like(student_logits) if need_s else None
            grad_teacher = torch.empty_like(teacher_logits) if need_t else None

            if beta not in (0, 1):
                beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
                log_beta = torch.log(beta_t)
                log_1_minus_beta = torch.log1p(-beta_t)
            else:
                beta_t = log_beta = log_1_minus_beta = None

            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                s_chunk = student_logits[start:end]
                t_chunk = teacher_logits[start:end]
                inputs = []
                if need_s:
                    s_chunk = s_chunk.detach().requires_grad_(True)
                    inputs.append(s_chunk)
                if need_t:
                    t_chunk = t_chunk.detach().requires_grad_(True)
                    inputs.append(t_chunk)
                with torch.enable_grad():
                    loss_chunk = _chunk_jsd_sum(s_chunk, t_chunk, beta, beta_t, log_beta, log_1_minus_beta)
                grads = torch.autograd.grad(loss_chunk, inputs, grad_outputs=scale, retain_graph=False)
                gi = 0
                if need_s:
                    grad_student[start:end] = grads[gi]
                    gi += 1
                if need_t:
                    grad_teacher[start:end] = grads[gi]

            # student_logits, teacher_logits, chunk_size, beta, num_valid
            return grad_student, grad_teacher, None, None, None

    _CHUNKED_JSD_FUNC = _ChunkedJSDFunc
    return _CHUNKED_JSD_FUNC


class GKDLoss(Loss):
    """Generalized Knowledge Distillation (GKD) loss based on Jensen-Shannon Divergence.

    Implements the on-policy distillation objective from:
        "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
        (https://arxiv.org/abs/2306.13649)

    The loss is a β-weighted mixture of two KL divergences:
        JSD_β(S || T) = β · KL(T || M) + (1 - β) · KL(S || M)
        where M = β · T + (1 - β) · S  (mixture distribution)

    Special cases:
        β = 0  →  forward KL(S || T)   (mean-seeking)
        β = 1  →  reverse KL(T || S)   (mode-seeking)
        β = 0.5 →  symmetric JSD

    Args:
        beta: Weight for teacher in the JSD mixture (default: 0.5).
        temperature: Softmax temperature applied to logits before divergence (default: 1.0).
        ignore_index: Token index to ignore in the loss mask (default: -100).
        chunk_size: Number of valid tokens processed per chunk to reduce peak memory (default: 512).
        sft_alpha: Weight of an auxiliary supervised cross-entropy term added on dataset-sourced
            rounds (default: 0.0, i.e. pure distillation). The caller enables it per round via the
            ``apply_sft_loss`` argument to :meth:`__call__`, so a run can mix on-policy student
            generations (distillation only) with labelled dataset rows (distillation + SFT).
    """

    require_logits = True

    def __init__(
        self,
        beta: float = 0.5,
        temperature: float = 1.0,
        ignore_index: int = -100,
        chunk_size: int = 512,
        sft_alpha: float = 0.0,
        **kwargs,
    ):
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f'beta must be in [0, 1], got {beta}')
        if temperature <= 0:
            raise ValueError(f'temperature must be > 0, got {temperature}')
        self.beta = beta
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.chunk_size = chunk_size
        self.sft_alpha = sft_alpha

    def micro_batch_scale(self, inputs, indices):
        return len(indices) / len(inputs)

    def __call__(
        self,
        inputs,
        outputs,
        *,
        teacher_logits: Optional['torch.Tensor'] = None,
        teacher_topk_logprobs: Optional['torch.Tensor'] = None,
        teacher_topk_indices: Optional['torch.Tensor'] = None,
        topk: Optional[int] = None,
        apply_sft_loss: bool = False,
        **kwargs,
    ) -> LossOutput:
        """Compute GKD / JSD distillation loss.

        Args:
            inputs: Dict containing 'labels' [batch, seq_len] with ignore_index for non-response tokens.
            outputs: Dict containing 'logits' [batch, seq_len, vocab_size] from the student model.
            teacher_logits: [batch, seq_len, vocab_size] full vocabulary logits from a local teacher.
                                Either teacher_logits or (teacher_topk_logprobs + teacher_topk_indices)
                                must be provided.
            teacher_topk_logprobs: [batch, seq_len, topk] log-probs from a remote teacher API.
                                Returned by a vLLM-compatible /v1/completions prompt_logprobs call.
            teacher_topk_indices: [batch, seq_len, topk] token indices corresponding to teacher_topk_logprobs.
            topk: If set together with teacher_logits, only the top-k teacher tokens are used to
                  reduce vocabulary size before computing the JSD (memory-efficient local teacher mode).

        Returns:
            LossOutput with scalar 'loss' averaged over valid (non-ignored) response tokens.
        """
        assert teacher_logits is not None or (teacher_topk_logprobs is not None and teacher_topk_indices is not None), (
            'Either logits or both topk_logprobs and topk_indices must be provided.')

        labels = inputs['labels']
        student_logits = outputs['logits']
        # Align seq dimension: some MLLMs return extra prefix logits
        if student_logits.shape[1] != labels.shape[1]:
            student_logits = student_logits[:, -labels.shape[1]:]
        if teacher_logits is not None and teacher_logits.shape[1] > student_logits.shape[1]:
            teacher_logits = teacher_logits[:, :student_logits.shape[1]]
        if teacher_topk_logprobs is not None and teacher_topk_logprobs.shape[1] > student_logits.shape[1]:
            teacher_topk_logprobs = teacher_topk_logprobs[:, :student_logits.shape[1]]
            teacher_topk_indices = teacher_topk_indices[:, :student_logits.shape[1]]

        loss = self._generalized_jsd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            labels=labels,
            beta=self.beta,
            temperature=self.temperature,
            ignore_index=self.ignore_index,
            chunk_size=self.chunk_size,
            topk=topk,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_topk_indices=teacher_topk_indices,
        )
        if apply_sft_loss and self.sft_alpha > 0:
            loss = loss + self.sft_alpha * self._supervised_ce(inputs['labels'], outputs['logits'])
        return LossOutput(loss=loss, num_tokens=0)

    def _supervised_ce(self, labels, logits):
        """Mean cross-entropy on the labelled rows, the auxiliary SFT term scaled by ``sft_alpha``."""
        import torch.nn.functional as F
        # Align the seq dimension the same way the JSD path does: some MLLMs return extra prefix logits.
        if logits.shape[1] != labels.shape[1]:
            logits = logits[:, -labels.shape[1]:]
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
            reduction='mean')

    @staticmethod
    def _generalized_jsd_loss(
        student_logits,
        teacher_logits=None,
        labels=None,
        beta: float = 0.5,
        temperature: float = 1.0,
        ignore_index: int = -100,
        chunk_size: int = 512,
        topk: Optional[int] = None,
        teacher_topk_logprobs=None,
        teacher_topk_indices=None,
    ):
        """Core chunked JSD loss computation.

        Supports three teacher modes:
        1. Full-vocabulary local teacher (teacher_logits, topk=None)
        2. Top-k local teacher (teacher_logits, topk=k)
        3. Remote API teacher (teacher_topk_logprobs + teacher_topk_indices)

        The function processes valid tokens in chunks to keep peak GPU memory bounded.

        Args:
            student_logits: [batch, seq_len, vocab_size] or [batch, seq_len, topk] after top-k reduction.
            teacher_logits: [batch, seq_len, vocab_size] full vocabulary logits from local teacher.
            labels: [batch, seq_len] shifted labels; positions where value == ignore_index are excluded.
            beta: JSD mixture weight (0=forward KL, 1=reverse KL, 0.5=symmetric JSD).
            temperature: Softmax temperature.
            chunk_size: Tokens per chunk.
            topk: If given, reduce local teacher to top-k tokens before computing JSD.
            teacher_topk_logprobs: [batch, seq_len, topk] from remote API.
            teacher_topk_indices: [batch, seq_len, topk] from remote API.

        Returns:
            Scalar loss tensor.
        """
        import torch
        import torch.nn.functional as F

        # ── Top-k reduction ──────────────────────────────────────────────────
        # Top-k mode: gather/topk first to get small [*, k] tensors, then scale in-place
        if teacher_topk_logprobs is not None and teacher_topk_indices is not None:
            # Remote API teacher: teacher already provides top-k log-probs (T=1).
            # Gather student logits at teacher's top-k indices, then scale in-place.
            teacher_topk_indices = teacher_topk_indices.to(student_logits.device)
            teacher_topk_logprobs = teacher_topk_logprobs.to(student_logits.device)
            student_logits = torch.gather(student_logits, dim=-1, index=teacher_topk_indices)
            student_logits.div_(temperature)
            teacher_logits = teacher_topk_logprobs / temperature
            temperature = 1.0
        elif topk is not None and teacher_logits is not None:
            # Local teacher: select top-k from teacher, gather corresponding student logits
            teacher_logits = teacher_logits.to(student_logits.device)
            teacher_logits, topk_idx = torch.topk(teacher_logits, k=topk, dim=-1)
            teacher_logits.div_(temperature)
            student_logits = torch.gather(student_logits, dim=-1, index=topk_idx)
            student_logits.div_(temperature)
            temperature = 1.0

        if teacher_logits is not None and teacher_logits.device != student_logits.device:
            teacher_logits = teacher_logits.to(student_logits.device)

        # ── Mask valid (response) tokens ──────────────────────────────────────
        if labels is not None:
            mask = labels != ignore_index
            # Vocab-size mismatch (e.g. Qwen2.5-VL-3B vs 7B): pad the smaller side
            # so both distributions are defined over the same token set.
            stu_dim = student_logits.shape[-1]
            tea_dim = teacher_logits.shape[-1]
            if stu_dim < tea_dim:
                student_logits = F.pad(student_logits, (0, tea_dim - stu_dim))
                student_logits[..., stu_dim:] = teacher_logits[..., stu_dim:]
            elif stu_dim > tea_dim:
                teacher_logits = F.pad(teacher_logits, (0, stu_dim - tea_dim))
                teacher_logits[..., tea_dim:] = student_logits[..., tea_dim:]
            student_logits = student_logits[mask]  # [num_valid, vocab/topk]
            teacher_logits = teacher_logits[mask]
            num_valid = mask.sum()
            # ``[mask]`` already created fresh storage, so in-place divide is safe
            # and avoids an extra [num_valid, V] allocation.
            student_logits.div_(temperature)
            teacher_logits.div_(temperature)
        else:
            # Keep logits, may be an infer scenario
            student_logits = student_logits.reshape(-1, student_logits.size(-1)) / temperature
            teacher_logits = teacher_logits.reshape(-1, teacher_logits.size(-1)) / temperature
            num_valid = student_logits.size(0)

        if num_valid == 0:
            # Keep the empty partition connected to the graph so its backward is a no-op, not an error.
            return student_logits.sum()

        # Chunked accumulation with per-chunk backward (see _ChunkedJSDFunc): a plain loop here
        # would retain every chunk's [chunk, V] intermediates in the autograd graph until backward,
        # which OOMs on long on-policy rounds (full vocab x thousands of response tokens).
        func = _get_chunked_jsd_func()
        return func.apply(student_logits, teacher_logits, chunk_size, beta, num_valid)
