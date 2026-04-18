# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import TYPE_CHECKING, Optional

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss

if TYPE_CHECKING:
    import torch


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
    """

    def __init__(
        self,
        beta: float = 0.5,
        temperature: float = 1.0,
        ignore_index: int = -100,
        chunk_size: int = 512,
        **kwargs,
    ):
        self.beta = beta
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.chunk_size = chunk_size

    def __call__(
        self,
        inputs,
        outputs,
        *,
        teacher_logits: Optional['torch.Tensor'] = None,
        teacher_topk_logprobs: Optional['torch.Tensor'] = None,
        teacher_topk_indices: Optional['torch.Tensor'] = None,
        topk: Optional[int] = None,
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
            chunk_size=self.chunk_size,
            topk=topk,
            teacher_topk_logprobs=teacher_topk_logprobs,
            teacher_topk_indices=teacher_topk_indices,
        )
        return LossOutput(loss=loss, num_tokens=0)

    @staticmethod
    def _generalized_jsd_loss(
        student_logits,
        teacher_logits=None,
        labels=None,
        beta: float = 0.5,
        temperature: float = 1.0,
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
            mask = labels != -100  # ignore_index is always -100 per convention
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
        else:
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            num_valid = student_logits.size(0)
        student_logits.div_(temperature)
        teacher_logits.div_(temperature)

        if num_valid == 0:
            return student_logits.new_zeros(())

        num_valid_int = int(num_valid) if isinstance(num_valid, int) else num_valid.item()
        total_loss = student_logits.new_zeros(())

        # Pre-compute log(beta) / log(1-beta) once for the mixture
        if beta not in (0, 1):
            beta_t = torch.tensor(beta, dtype=student_logits.dtype, device=student_logits.device)
            log_beta = torch.log(beta_t)
            log_1_minus_beta = torch.log1p(-beta_t)
        else:
            beta_t = log_beta = log_1_minus_beta = None

        # ── Chunked loss accumulation ─────────────────────────────────────────
        for start in range(0, num_valid_int, chunk_size):
            end = min(start + chunk_size, num_valid_int)
            s_chunk = student_logits[start:end]
            t_chunk = teacher_logits[start:end]

            s_log_probs = F.log_softmax(s_chunk, dim=-1)
            t_log_probs = F.log_softmax(t_chunk, dim=-1)
            del s_chunk, t_chunk

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
                del mixture_log_probs
                jsd_chunk = beta_t * kl_teacher + (1 - beta_t) * kl_student
                del kl_teacher, kl_student

            total_loss = total_loss + jsd_chunk.sum()
            del jsd_chunk, s_log_probs, t_log_probs

        return total_loss / num_valid
