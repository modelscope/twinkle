# Copyright (c) ModelScope Contributors. All rights reserved.
"""
CrossTokenLoss: Projection-Guided Cross-Tokenizer Knowledge Distillation Loss.

This module implements the X-Token approach with two loss modes:

1. **P-KL (Projection-KL)**: Projects student distribution to teacher vocabulary
   space via a sparse projection matrix W, then computes chunk-averaged KL
   divergence over a microbatch-global top-k teacher subset. This is the
   NeMo-style P-KL path.

2. **H-KL (Heuristic-KL)**: Partitions the student vocabulary into *common*
   (exact-token-mapped) and *uncommon* (no direct mapping) subsets. Computes
   KL divergence on common tokens, L1 loss on uncommon tokens, and ULD
   (Unmatched Logit Distillation) for student tokens with no projection
   mapping.

Reference:
    "X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation"
    (https://arxiv.org/pdf/2605.21699)

Key differences from CTKDLoss:
    1. P-KL mode adds chunk-averaged KL over top-k teacher subset (NeMo-style).
    2. H-KL mode adds common/uncommon partition + ULD loss component.
    3. Supports both full teacher logits (TransformersModel) and top-k
       logprobs (vLLM sampler).
    4. Multi-teacher aggregation with static weights.
"""
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import hashlib
import os
import pickle
import threading

import torch
import torch.nn.functional as F

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss


# ------------------------------------------------------------------
# Module-level helpers (replicating NeMo's chunk/log-prob utilities)
# ------------------------------------------------------------------

def _chunk_average_log_probs(
        log_probs: torch.Tensor,
        chunk_id: torch.Tensor,
        max_chunks: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average ``log_probs`` over chunks defined by ``chunk_id``.

    Builds a one-hot chunk mask from ``chunk_id`` (``-1`` = no chunk), then
    ``bmm``-aggregates and divides by chunk sizes.  Mirrors NeMo's
    ``chunk_average_log_probs`` (without CP all-reduce, since twinkle runs
    single-rank loss).

    Args:
        log_probs: ``[B, T, V]`` log-probabilities.
        chunk_id: ``[B, T]`` long tensor, values in ``[-1, max_chunks)``.
        max_chunks: number of chunk buckets.

    Returns:
        chunk_log_probs: ``[B, max_chunks, V]`` averaged log-probs.
        chunk_sizes: ``[B, max_chunks]`` float tensor of bucket sizes.
    """
    device = log_probs.device
    chunk_arange = torch.arange(max_chunks, device=device).view(1, 1, -1)
    chunk_mask = chunk_id.unsqueeze(-1) == chunk_arange  # [B, T, C]
    chunk_mask_f = chunk_mask.transpose(1, 2).to(log_probs.dtype)  # [B, C, T]
    chunk_sums = torch.bmm(chunk_mask_f, log_probs)  # [B, C, V]
    chunk_sizes = chunk_mask.sum(dim=1).float()  # [B, C]
    eps = 1e-10
    chunk_log_probs = chunk_sums / (chunk_sizes.unsqueeze(-1) + eps)
    return chunk_log_probs, chunk_sizes


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Global cache for projection matrices and exact token maps
_PROJECTION_MATRIX_CACHE = {}
_EXACT_TOKEN_MAP_CACHE = {}


class CrossTokenLoss(Loss):
    """Cross-Tokenizer Knowledge Distillation Loss using X-Token projection.

    Supports two loss modes:
    - ``'pkl'``: NeMo-style Projection-KL with top-k subset.
    - ``'hkl'``: Gold-loss with common/uncommon partition + ULD.

    Teacher data can be provided as:
    - ``teacher_logits_group``: list of [B, T_t, V_t] full logits
    - ``teacher_topk_logprobs_group`` + ``teacher_topk_indices_group``: top-k format
    - ``teacher_input_ids_group``: optional, for sequence alignment

    Args:
        student_tokenizer: Student model tokenizer.
        teacher_tokenizer_group: List of teacher tokenizers.
        teacher_weights: Per-teacher static weights (default: equal).
        max_length: Maximum span length L for multi-token matching.
        beta: Base weight β for projection.
        gamma: Decay rate γ for multi-token weights.
        loss_type: 'pkl' for projection-KL or 'hkl' for heuristic-KL.
        temperature: Softmax temperature for KL divergence.
        gamma_kl: Weight for common-KL term in H-KL mode.
        gamma_uld: Weight for ULD term in H-KL mode.
        vocab_topk: Top-k vocab size for teacher logits subset in P-KL.
        uncommon_topk: Top-k for uncommon L1 loss in H-KL mode.
        reverse_kl: If True, compute KL(student || teacher).
        exact_token_match_only: Only use exact token matches (no multi-token).
        kl_loss_weight: Scalar multiplier on distillation loss.
        ce_loss_weight: Scalar multiplier on CE auxiliary loss.
        dynamic_loss_scaling: Rescale KD term to match CE magnitude each step.
        device: Device for projection matrices.
    """

    require_logits = True

    def __init__(
            self,
            student_tokenizer: 'PreTrainedTokenizer',
            teacher_tokenizer_group: list,
            teacher_weights: Optional[list] = None,
            max_length: int = 4,
            beta: float = 0.95,
            gamma: float = 0.1,
            loss_type: str = 'pkl',
            temperature: float = 1.0,
            gamma_kl: float = 1.0,
            gamma_uld: float = 0.5,
            vocab_topk: int = 512,
            uncommon_topk: int = 8192,
            reverse_kl: bool = False,
            exact_token_match_only: bool = False,
            kl_loss_weight: float = 1.0,
            ce_loss_weight: float = 1.0,
            dynamic_loss_scaling: bool = False,
            projection_topk: int = 8,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer_group = teacher_tokenizer_group
        self.num_teachers = len(teacher_tokenizer_group)

        if teacher_weights is None:
            self.teacher_weights = [1.0 / self.num_teachers] * self.num_teachers
        else:
            if len(teacher_weights) != self.num_teachers:
                raise ValueError(
                    f"Number of weights ({len(teacher_weights)}) must match "
                    f"number of teachers ({self.num_teachers})"
                )
            weight_sum = sum(teacher_weights)
            self.teacher_weights = [w / weight_sum for w in teacher_weights]

        self.max_length = max_length
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.temperature = temperature
        self.gamma_kl = gamma_kl
        self.gamma_uld = gamma_uld
        self.vocab_topk = vocab_topk
        self.uncommon_topk = uncommon_topk
        self.reverse_kl = reverse_kl
        self.exact_token_match_only = exact_token_match_only
        self.kl_loss_weight = kl_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.dynamic_loss_scaling = dynamic_loss_scaling
        self.projection_topk = projection_topk

        if loss_type not in ('pkl', 'hkl'):
            raise ValueError(f"Unknown loss_type: {loss_type}. Use 'pkl' or 'hkl'.")

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif hasattr(torch, 'npu') and torch.npu.is_available():
                self.device = torch.device('npu:0')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device

        self.student_vocab_size = len(student_tokenizer)
        self.teacher_vocab_sizes = [len(tok) for tok in teacher_tokenizer_group]

        # Lazy initialization
        self._projection_matrices_built = False
        self.projection_student_indices_list: list = []
        self.projection_teacher_indices_list: list = []
        self.projection_values_list: list = []

        # H-KL: exact token maps (common/uncommon partition)
        self._exact_maps_built = False
        self._common_student_indices_list: list = []
        self._common_teacher_indices_list: list = []
        self._uncommon_student_indices_list: list = []
        self._uncommon_teacher_indices_list: list = []

        # 词表文本映射缓存:投影构建 phase1/phase2 时填充,exact maps 直接
        # 复用,避免对全词表二次 decode(原实现重复解码 ~41 万 token)。
        self._student_texts: list = []
        self._teacher_text_maps_list: list = [None] * self.num_teachers

    # ------------------------------------------------------------------
    # Projection matrix building (same as CTKDLoss)
    # ------------------------------------------------------------------

    def _ensure_projection_matrices_built(self):
        """Lazy-build projection matrices with caching."""
        if self._projection_matrices_built:
            return
        if not hasattr(self, '_build_lock'):
            self._build_lock = threading.Lock()
        with self._build_lock:
            if self._projection_matrices_built:
                return
            cache_key = self._generate_cache_key()
            self._cache_key = cache_key

            if cache_key in _PROJECTION_MATRIX_CACHE:
                cached = _PROJECTION_MATRIX_CACHE[cache_key]
                self.projection_student_indices_list = [
                    t.to(self.device) if self.device is not None else t.clone()
                    for t in cached['student_indices']
                ]
                self.projection_teacher_indices_list = [
                    t.to(self.device) if self.device is not None else t.clone()
                    for t in cached['teacher_indices']
                ]
                self.projection_values_list = [
                    t.to(self.device) if self.device is not None else t.clone()
                    for t in cached['values']
                ]
                # 精确匹配学生集合(归一化后权重不再是 1.0,统计改用集合)
                self._exact_matched_student_ids = [
                    set(ids) for ids in cached.get('exact_matched', [])
                ]
            else:
                self._exact_matched_student_ids = [
                    set() for _ in range(self.num_teachers)
                ]
                for i, teacher_tok in enumerate(self.teacher_tokenizer_group):
                    self._build_projection_matrix_for_teacher(teacher_tok, i)

                _PROJECTION_MATRIX_CACHE[cache_key] = {
                    'student_indices': self.projection_student_indices_list,
                    'teacher_indices': self.projection_teacher_indices_list,
                    'values': self.projection_values_list,
                    'exact_matched': [
                        list(s) for s in self._exact_matched_student_ids
                    ],
                }

            # Build exact token maps for H-KL
            if self.loss_type == 'hkl':
                self._build_exact_token_maps()

            self._projection_matrices_built = True

    def _ensure_student_texts(self) -> None:
        """Decode the full student vocab once into ``self._student_texts``.

        No-op when the list already exists (filled by projection phase2 in
        the H-KL path, or by an earlier call).  Only runs when the projection
        matrices were served from the module cache, where phase1/phase2 did
        not execute.
        """
        if self._student_texts:
            return
        self._student_texts = [
            self.student_tokenizer.decode(
                [sid], skip_special_tokens=False
            )
            for sid in range(self.student_vocab_size)
        ]

    def _ensure_teacher_text_maps(self, teacher_index: int) -> None:
        """Decode one teacher vocab once; cache exact/stripped text->id maps.

        No-op when the maps already exist (filled by projection phase1 in the
        H-KL path, or by an earlier call).  Only runs when the projection
        matrices were served from the module cache, where phase1 did not
        execute.
        """
        if self._teacher_text_maps_list[teacher_index] is not None:
            return
        teacher_tok = self.teacher_tokenizer_group[teacher_index]
        teacher_exact_text_to_id = {}
        teacher_stripped_text_to_id = {}
        for token_id in range(len(teacher_tok)):
            token_text = teacher_tok.decode(
                [token_id], skip_special_tokens=False
            )
            teacher_exact_text_to_id[token_text] = token_id
            teacher_stripped_text_to_id[token_text.strip()] = token_id
        self._teacher_text_maps_list[teacher_index] = (
            teacher_exact_text_to_id, teacher_stripped_text_to_id
        )


    def _build_exact_token_maps(self):
        """Build common/uncommon token partitions for H-KL mode.

        Common: tokens with a text match between student and teacher, matched
        exact-text first (raw decoded token text) with stripped-text fallback.
        Uncommon: all other tokens.

        不再对全词表二次 decode:学生 id→文本列表复用投影构建 phase2 的
        结果,教师文本→id 映射复用 phase1 的结果(懒构建兜底投影矩阵
        缓存命中路径)。这里只剩 dict 查找 + 张量转换,构建耗时从分钟级
        降到秒级,内存峰值也大幅下降。
        """


        # 进程内缓存:同一 tokenizer 组合的 common/uncommon 分区可跨
        # CrossTokenLoss 实例复用(与投影矩阵缓存同 key 同模式)。
        cache_key = getattr(self, '_cache_key', None) or self._generate_cache_key()
        if cache_key in _EXACT_TOKEN_MAP_CACHE:
            cached = _EXACT_TOKEN_MAP_CACHE[cache_key]
            (
                self._common_student_indices_list,
                self._common_teacher_indices_list,
                self._uncommon_student_indices_list,
                self._uncommon_teacher_indices_list,
            ) = [
                [t.to(self.device) if self.device is not None else t.clone()
                 for t in ts]
                for ts in cached
            ]
            return

        self._common_student_indices_list = []
        self._common_teacher_indices_list = []
        self._uncommon_student_indices_list = []
        self._uncommon_teacher_indices_list = []

        # 复用投影构建阶段(phase1/phase2)已解码的词表文本
        self._ensure_student_texts()
        student_texts = self._student_texts

        for i, teacher_tok in enumerate(self.teacher_tokenizer_group):
            self._ensure_teacher_text_maps(i)
            teacher_exact_text_to_id, teacher_stripped_text_to_id = (
                self._teacher_text_maps_list[i]
            )



            # Find common tokens
            common_s = []
            common_t = []
            uncommon_s = []
            # 投影行最大值(NeMo relaxed 规则):投影矩阵在本函数前段已构建,
            # 按学生行聚合 max 权重与 argmax 教师 token。H-KL 模式下 common
            # 集 = 文本精确匹配(优先) ∪ 投影 top 权重 ≥ 0.6(含多 token
            # 投影首目标 β=0.95,KL 信号面更宽,与 NeMo 一致)。
            proj_s = self.projection_student_indices_list[i].cpu()
            proj_t = self.projection_teacher_indices_list[i].cpu()
            proj_v = self.projection_values_list[i].cpu().float()
            proj_row_max = {}  # s_id -> (max_value, argmax_teacher)
            for s_, t_, v_ in zip(
                    proj_s.tolist(), proj_t.tolist(), proj_v.tolist()
            ):
                cur = proj_row_max.get(s_)
                if cur is None or v_ > cur[0]:
                    proj_row_max[s_] = (v_, t_)
                elif v_ == cur[0] and t_ < cur[1]:
                    proj_row_max[s_] = (v_, t_)  # 同权 → 最低教师 index
            for s_id in range(self.student_vocab_size):

                s_text = student_texts[s_id]
                teacher_id = teacher_exact_text_to_id.get(s_text)
                if teacher_id is None:
                    teacher_id = teacher_stripped_text_to_id.get(
                        s_text.strip()
                    )
                if teacher_id is not None:
                    common_s.append(s_id)
                    common_t.append(teacher_id)
                    continue
                # NeMo H-KL relaxed 规则:投影 top 权重 ≥ 0.6 计入 common
                row = proj_row_max.get(s_id)
                if row is not None and row[0] >= 0.6:
                    common_s.append(s_id)
                    common_t.append(row[1])
                else:
                    uncommon_s.append(s_id)

            # Uncommon teacher tokens (not mapped to any student)
            mapped_teacher_ids = set(common_t)
            uncommon_t = [t_id for t_id in range(len(teacher_tok))
                          if t_id not in mapped_teacher_ids]

            self._common_student_indices_list.append(
                torch.tensor(common_s, dtype=torch.long, device=self.device)
            )
            self._common_teacher_indices_list.append(
                torch.tensor(common_t, dtype=torch.long, device=self.device)
            )
            self._uncommon_student_indices_list.append(
                torch.tensor(uncommon_s, dtype=torch.long, device=self.device)
            )
            self._uncommon_teacher_indices_list.append(
                torch.tensor(uncommon_t, dtype=torch.long, device=self.device)
            )

            n_common = len(common_s)
            n_uncommon_s = len(uncommon_s)
            n_uncommon_t = len(uncommon_t)

        _EXACT_TOKEN_MAP_CACHE[cache_key] = (
            self._common_student_indices_list,
            self._common_teacher_indices_list,
            self._uncommon_student_indices_list,
            self._uncommon_teacher_indices_list,
        )

    def _generate_cache_key(self) -> str:
        """Generate cache key from tokenizer configurations."""
        config_data = {
            'student_vocab': self.student_tokenizer.get_vocab(),
            'teacher_vocabs': [t.get_vocab() for t in self.teacher_tokenizer_group],
            'max_length': self.max_length,
            'beta': self.beta,
            'gamma': self.gamma,
            'projection_topk': self.projection_topk,
        }
        config_bytes = pickle.dumps(config_data)
        return hashlib.md5(config_bytes).hexdigest()

    def _build_projection_matrix_for_teacher(self, teacher_tokenizer, teacher_index):
        """Build sparse projection matrix W in COO format.

        Token matching priority: same-id first, then exact-text (raw decoded
        token text), then stripped-text fallback.  Stripped matching collapses
        whitespace/space-prefixed variants (e.g. '的' vs ' 的') into a single
        target, so the projected student mass lands on a token the teacher
        gives ~0 probability — inflating the KL even for identical
        student/teacher models.  Exact-first keeps such variants distinct
        when both exist in the teacher vocabulary.  Same-id-first additionally
        fixes "same-text-multi-id" collapse (byte-fallback tokens: many ids
        decode to U+FFFD), where two models agreeing on token X would
        otherwise be compared against a different id with the same text.

        未精确匹配的学生 token 用教师 tokenizer 重编码展开多 token 映射
        (权重 β·γ^i,不归一化)。

        说明(已回退 NeMo 风格 reverse pass + 行归一化):chunk 对齐修复
        (BOS + 预测目标跨度)后,reverse 补充的映射把学生质量摊到教师
        q≈0 的目标上,重新引入 KL 触底(诊断中 p_after≈0.23 的标点行),
        故恢复原始单向投影——对齐正确时精确匹配权重 1.0 直接命中教师
        的预测目标。
        """
        teacher_vocab_size = len(teacher_tokenizer)

        student_indices = []
        teacher_indices = []
        values = []
        matched_student_ids = set()
        if not hasattr(self, '_exact_matched_student_ids'):
            self._exact_matched_student_ids = [
                set() for _ in range(self.num_teachers)
            ]

        # ── phase1: 教师 token 解码文本 ─────────────────────────────────
        # (H-KL 时存到 self,供 _build_exact_token_maps 复用,避免二次
        # 解码全词表;旧实现此处还维护了只写不读的 teacher_texts,已删除)
        teacher_exact_text_to_id = {}
        teacher_stripped_text_to_id = {}
        for token_id in range(teacher_vocab_size):
            token_text = teacher_tokenizer.decode(
                [token_id], skip_special_tokens=False
            )
            teacher_exact_text_to_id[token_text] = token_id
            teacher_stripped_text_to_id[token_text.strip()] = token_id
        if self.loss_type == 'hkl' and self._teacher_text_maps_list[teacher_index] is None:
            self._teacher_text_maps_list[teacher_index] = (
                teacher_exact_text_to_id, teacher_stripped_text_to_id
            )

        # ── phase2: 学生精确匹配(同 id / 精确文本 / 去空白文本),权重 1.0 ──
        student_texts = []
        for student_id in range(self.student_vocab_size):
            student_token_text = self.student_tokenizer.decode(
                [student_id], skip_special_tokens=False
            )
            student_texts.append(student_token_text)
            # 同 id 优先:学生 id 在教师词表内且解码文本相同 → 恒等映射
            # (修复同文本多 id 坍缩,如字节回退 token 多个 id 均解码为 U+FFFD)
            teacher_id = None
            if student_id < teacher_vocab_size:
                if teacher_tokenizer.decode(
                        [student_id], skip_special_tokens=False
                ) == student_token_text:
                    teacher_id = student_id
            if teacher_id is None:
                teacher_id = teacher_exact_text_to_id.get(student_token_text)
            if teacher_id is None:
                teacher_id = teacher_stripped_text_to_id.get(
                    student_token_text.strip()
                )
            if teacher_id is not None:
                student_indices.append(student_id)
                teacher_indices.append(teacher_id)
                values.append(1.0)
                matched_student_ids.add(student_id)

        # H-KL 时存下学生 id→文本列表,供 _build_exact_token_maps 复用
        if self.loss_type == 'hkl' and not self._student_texts:
            self._student_texts = student_texts

        # ── phase3: forward 多 token(未精确匹配的学生 token) ─────────────
        # 解码文本含 U+FFFD 的 token 跳过(字节回退 token 无法文本匹配)
        for student_id, text in enumerate(student_texts):
            if student_id in matched_student_ids:
                continue
            if not text or not text.strip() or '\ufffd' in text:
                continue
            teacher_token_ids = teacher_tokenizer.encode(text, add_special_tokens=False)
            seq_length = len(teacher_token_ids)
            if 0 < seq_length < self.max_length:
                for i, t_id in enumerate(teacher_token_ids):
                    weight = self.beta * (self.gamma ** i)
                    student_indices.append(student_id)
                    teacher_indices.append(t_id)
                    values.append(weight)

        # 精确匹配学生集合(统计/日志判定用)
        self._exact_matched_student_ids[teacher_index] = set(matched_student_ids)

        student_tensor = torch.tensor(student_indices, dtype=torch.long, device=self.device)
        teacher_tensor = torch.tensor(teacher_indices, dtype=torch.long, device=self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)

        self.projection_student_indices_list.append(student_tensor)
        self.projection_teacher_indices_list.append(teacher_tensor)
        self.projection_values_list.append(values_tensor)

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def __call__(self, inputs, outputs, **kwargs) -> LossOutput:
        """Compute CrossToken distillation loss.

        Args:
            inputs: Dict with 'labels' [B, S] (with -100 for padding).
            outputs: Dict with 'logits' [B, S, V_s] from student model.
            **kwargs: Teacher data.

        Teacher data kwargs:
            teacher_logits_group: list of [B, T, V_t] full logits
            teacher_topk_logprobs_group: list of [B, T, topk] logprobs
            teacher_topk_indices_group: list of [B, T, topk] indices
            teacher_labels: list of [B, S] labels for loss mask

        Returns:
            LossOutput with the computed loss and number of tokens.
        """
        self._ensure_projection_matrices_built()

        # Get teacher data
        teacher_logits_group = kwargs.get('teacher_logits_group')
        if teacher_logits_group is None:
            teacher_logits_group = outputs.get('teacher_logits_group')

        teacher_topk_logprobs_group = kwargs.get('teacher_topk_logprobs_group')
        teacher_topk_indices_group = kwargs.get('teacher_topk_indices_group')

        # Convert top-k format to full logits-like if needed
        if teacher_logits_group is None and teacher_topk_logprobs_group is not None \
                and teacher_topk_indices_group is not None:
            teacher_logits_group = self._topk_to_full(
                teacher_topk_logprobs_group, teacher_topk_indices_group
            )

        if teacher_logits_group is None:
            raise ValueError(
                "No teacher logits provided. Pass teacher_logits_group or "
                "(teacher_topk_logprobs_group + teacher_topk_indices_group)."
            )
        if len(teacher_logits_group) != self.num_teachers:
            raise ValueError(
                f"Got {len(teacher_logits_group)} teacher logits, expected {self.num_teachers}."
            )

        # Get labels for loss mask
        student_labels = inputs.get('labels')
        if student_labels is None:
            raise ValueError("labels not found in inputs")

        teacher_labels_group = kwargs.get('teacher_labels')
        if teacher_labels_group is not None and len(teacher_labels_group) > 0:
            labels = teacher_labels_group[0].to(self.device)
        else:
            labels = student_labels

        student_logits = outputs.get('logits')
        if student_logits is None:
            raise ValueError("logits not found in outputs")

        # ── Chunk-alignment input_ids (optional) ───────────────────────
        # Per-teacher input_ids for character-span chunk alignment.
        teacher_input_ids_group = kwargs.get('teacher_input_ids_group')
        student_ids_for_align = inputs.get('input_ids')  # list or tensor

        # Compute loss per teacher
        total_kd = torch.tensor(0.0, device=student_logits.device)
        ce_loss = self._compute_ce(student_logits, labels)
        teacher_metrics = []

        for i in range(self.num_teachers):
            t_logits = teacher_logits_group[i]
            weight = self.teacher_weights[i]

            if self.loss_type == 'pkl':
                # Resolve per-teacher chunk-alignment input_ids.
                s_ids = self._resolve_alignment_ids(
                    student_ids_for_align, 'student'
                )
                t_ids = self._resolve_alignment_ids(
                    teacher_input_ids_group, i
                    if teacher_input_ids_group is not None
                    else None,
                )
                kd, metrics = self._compute_pkl(
                    student_logits, t_logits, labels, i,
                    student_input_ids=s_ids,
                    teacher_input_ids=t_ids,
                )
            elif self.loss_type == 'hkl':
                # 与 pkl 相同的 chunk 对齐输入(common-KL 用字符跨度对齐)
                s_ids = self._resolve_alignment_ids(
                    student_ids_for_align, 'student'
                )
                t_ids = self._resolve_alignment_ids(
                    teacher_input_ids_group, i
                    if teacher_input_ids_group is not None
                    else None,
                )
                kd, metrics = self._compute_hkl(
                    student_logits, t_logits, labels, i,
                    student_input_ids=s_ids,
                    teacher_input_ids=t_ids,
                )
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            total_kd = total_kd + weight * kd
            teacher_metrics.append({'index': i, 'weight': weight, **metrics})

        # Combine KD + CE
        if self.dynamic_loss_scaling:
            kd_detached = total_kd.detach().abs()
            ce_detached = ce_loss.detach().abs()
            kl_scale = torch.where(
                kd_detached > 0, ce_detached / kd_detached,
                torch.ones_like(kd_detached),
            )
            loss = kl_scale * total_kd + ce_loss
        else:
            loss = self.kl_loss_weight * total_kd + self.ce_loss_weight * ce_loss

        # Compute number of valid tokens. Return as a tensor (not an int) to
        # match the LossOutput convention: calculate_loss divides num_tokens by
        # dp_world_size and calls .item() on it (see CrossEntropyLoss).
        loss_mask = (labels != -100).float()
        num_tokens = loss_mask.sum().clamp(min=1)

        return LossOutput(loss=loss, num_tokens=num_tokens)

    def _topk_to_full(self, topk_logprobs_group, topk_indices_group):
        """Convert vLLM top-k logprobs to full-vocab probability tensors."""
        full_group = []
        for i, (topk_lp, topk_idx) in enumerate(
                zip(topk_logprobs_group, topk_indices_group)
        ):
            vocab_size = self.teacher_vocab_sizes[i]
            batch_size, seq_len, topk = topk_lp.shape
            probs_full = torch.zeros(
                batch_size, seq_len, vocab_size,
                dtype=topk_lp.dtype, device=self.device,
            )
            topk_probs = torch.exp(topk_lp.to(self.device))
            topk_idx_dev = topk_idx.to(self.device)
            probs_full.scatter_(dim=2, index=topk_idx_dev, src=topk_probs)
            full_group.append(probs_full)
        return full_group

    def _compute_ce(self, student_logits, labels):
        """Compute auxiliary next-token cross-entropy loss."""
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous().to(student_logits.device)
        loss_mask = (shift_labels != -100).float()
        if loss_mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device)
        ce_per_token = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction='none',
        ).reshape(shift_labels.shape)
        return (ce_per_token * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)

    # ------------------------------------------------------------------
    # P-KL mode
    # ------------------------------------------------------------------

    def _compute_pkl(
            self,
            student_logits: torch.Tensor,
            teacher_probs: torch.Tensor,
            labels: torch.Tensor,
            teacher_index: int,
            *,
            student_input_ids: Optional[torch.Tensor] = None,
            teacher_input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """NeMo-style projection-KL loss with chunk alignment.

        When both ``student_input_ids`` and ``teacher_input_ids`` are provided,
        character-span alignment is used to assign each teacher token to the
        best-matching student position (chunk), then teacher log-probs are
        chunk-averaged before KL computation — matching NeMo's
        ``_compute_p_kl`` behaviour.  Otherwise falls back to per-position KL
        with simple ``min(seq_len)`` truncation.

        Diagnostics printed per step:
          - ``proj_mass_min/mean`` — projected probability mass per position
            (catches projection coverage gaps)
          - ``teacher_topk_cov`` — fraction of teacher probability mass
            captured by the top-k vocab subset
          - ``KL ANOMALIES`` — top-N positions with the largest per-position
            KL (env ``XTOKEN_DEBUG_KL_TOPN``, default 20) plus a count of
            positions exceeding ``XTOKEN_DEBUG_KL_THRESHOLD`` (default 10.0),
            each row carrying the student/teacher token mapping info and the
            top-k renormalized student/teacher probabilities of the projection
            target (``p_k``/``q_k``) plus the teacher-position count of the
            chunk (``t_n``) for verification
        """
        T = self.temperature
        eps = 1e-10

        # ── Next-token shift ─────────────────────────────────────────────
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_probs[..., :-1, :].contiguous().to(
            shift_student_logits.device
        )
        shift_labels = labels[..., 1:].contiguous().to(shift_student_logits.device)

        # ── Tokenizer-level vocab sizing (HF lm_head may be wider) ───────
        tkr_vocab_size = self.teacher_vocab_sizes[teacher_index]
        model_vocab_size = shift_teacher.shape[-1]
        if model_vocab_size > tkr_vocab_size:
            shift_teacher = shift_teacher[..., :tkr_vocab_size]

        # ── Chunk alignment (character-span based) ───────────────────────
        use_chunk_alignment = (
                student_input_ids is not None
                and teacher_input_ids is not None
                and self.num_teachers == 1  # multi-teacher alignment is complex
        )
        if use_chunk_alignment:
            (
                student_chunk_id,  # [B, Ts-1]  values in [-1, max_chunks)
                teacher_chunk_id,  # [B, Tt-1]  values in [-1, max_chunks)
                max_chunks,
            ) = self._build_chunk_ids(
                student_input_ids, teacher_input_ids, teacher_index,
            )
            # Trim seq dims to match the shifted logits
            s_clen = shift_student_logits.shape[1]
            t_clen = shift_teacher.shape[1]
            student_chunk_id = student_chunk_id[:, :s_clen]
            teacher_chunk_id = teacher_chunk_id[:, :t_clen]
            shift_labels = shift_labels[:, :s_clen]
        else:
            # Fallback: simple min(seq_len) truncation (original behaviour)
            s_clen = shift_student_logits.shape[1]
            t_clen = shift_teacher.shape[1]
            min_len = min(s_clen, t_clen)
            shift_student_logits = shift_student_logits[:, :min_len, :]
            shift_teacher = shift_teacher[:, :min_len, :]
            shift_labels = shift_labels[:, :min_len]

        # ── Build loss mask ──────────────────────────────────────────────
        loss_mask = (shift_labels != -100).float()  # [B, min_len]

        # ── Student log-probs + projection ───────────────────────────────
        student_log_probs = F.log_softmax(shift_student_logits / T, dim=-1)
        student_probs = student_log_probs.exp()

        projected = self._project_student_probs(
            student_probs, teacher_index, tkr_vocab_size
        )  # [B, S_s, V_t]

        # ── Diagnose projection coverage ─────────────────────────────────
        with torch.no_grad():
            proj_mass = projected.sum(dim=-1)  # [B, S]
            _masked_mass = proj_mass * loss_mask
            proj_mass_mean = (
                    _masked_mass.sum() / loss_mask.sum().clamp(min=1.0)
            ).item()
            proj_mass_min = proj_mass[loss_mask.bool()].min().item() if loss_mask.sum() > 0 else 1.0

        # ── Teacher logits → log-probs ───────────────────────────────────
        if shift_teacher.max() <= 1.0 and shift_teacher.sum(dim=-1).min() > 0.5:
            teacher_logits_approx = torch.log(shift_teacher.clamp(min=eps))
        else:
            teacher_logits_approx = shift_teacher

        # ── Microbatch-global top-k ──────────────────────────────────────
        k = min(self.vocab_topk, tkr_vocab_size, projected.shape[-1])
        teacher_flat = teacher_logits_approx.reshape(-1, tkr_vocab_size)
        importance = teacher_flat.max(dim=0).values
        _, topk_idx = torch.topk(importance, k=k)
        topk_idx = topk_idx.sort().values  # [k]

        # ── Teacher top-k 子集 log-probs(NeMo 式:子集内 log_softmax) ────
        # 只对 top-k 列做 log_softmax,避免全词表 [B, T_t, V_t] 张量
        # (~2.1GB,N_SAMPLES=4 时 batch=8 叠加 OOM)。子集归一化与 KL 里的
        # 重归一化语义一致。全词表 log-probs 仅在 debug(KL ANOMALIES)
        # 需要显示教师自身 token 概率时延迟计算。
        teacher_topk_logits = teacher_logits_approx[..., topk_idx]  # [B, T_t, k]
        teacher_topk_log_probs = torch.log_softmax(
            teacher_topk_logits / T, dim=-1
        )  # [B, T_t, k]

        # ── Chunk-average (if alignment available) ───────────────────────
        if use_chunk_alignment:
            # Student side: chunk-average the projected probs
            projected_chunks, _ = _chunk_average_log_probs(
                projected, student_chunk_id, max_chunks
            )  # [B, C, V_t]
            # Teacher side: chunk-average the top-k subset log-probs
            teacher_chunks, _ = _chunk_average_log_probs(
                teacher_topk_log_probs, teacher_chunk_id, max_chunks
            )  # [B, C, k]
            # Valid chunk mask: chunks that have both student and teacher
            # contributions (nonzero student chunk size + nonzero teacher).
            s_sizes = (student_chunk_id.unsqueeze(-1)
                       == torch.arange(max_chunks, device=student_chunk_id.device).view(1, 1, -1)).sum(dim=1).float()
            t_sizes = (teacher_chunk_id.unsqueeze(-1)
                       == torch.arange(max_chunks, device=teacher_chunk_id.device).view(1, 1, -1)).sum(dim=1).float()
            # 有效 chunk 还需学生标签有效:教师侧 padding token(id 0)的字符
            # 跨度会与学生的 padding chunk 重叠,把填充位置计入 KD 均值,
            # 抬高 masked_kl(现象:valid 数 > 非 -100 标签数)
            chunk_valid = ((s_sizes > 0) & (t_sizes > 0)
                           & (shift_labels != -100))  # [B, C]
            # Use student chunk as the KL axis
            proj_C = projected_chunks
            tgt_C = teacher_chunks
            valid_mask = chunk_valid  # [B, C]
        else:
            # Per-position (no chunk averaging)
            proj_C = projected
            tgt_C = teacher_topk_log_probs
            valid_mask = loss_mask > 0  # [B, S]

        # ── Slice to top-k(已切片) ───────────────────────────────────────
        projected_k = proj_C[..., topk_idx]  # [B, C|S, k]
        teacher_log_probs_k = tgt_C  # [B, C|S, k] 已是 top-k 子集

        # Renormalize projected student within top-k
        projected_k = projected_k / (projected_k.sum(dim=-1, keepdim=True) + eps)
        log_projected_k = (projected_k + eps).log()

        # Teacher within top-k: renormalize to ensure the K-subset sums to
        # 1 under exp (chunk-average can break exact normalization).
        teacher_log_probs_k = teacher_log_probs_k - torch.logsumexp(
            teacher_log_probs_k, dim=-1, keepdim=True
        )
        log_teacher_k = teacher_log_probs_k

        # ── KL divergence ────────────────────────────────────────────────
        if self.reverse_kl:
            per_pos_kl = F.kl_div(
                log_teacher_k, log_projected_k,
                reduction='none', log_target=True,
            ).sum(dim=-1)
        else:
            per_pos_kl = F.kl_div(
                log_projected_k, log_teacher_k,
                reduction='none', log_target=True,
            ).sum(dim=-1)

        valid_f = valid_mask.float()
        masked_kl = (per_pos_kl * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        # 数值防御:KL 理论 ≥0——短生成步投影质量≈0 时重归一化分布全为
        # eps 噪声,per_pos_kl 可能微负(-1e-3 量级),归零避免负 loss
        masked_kl = masked_kl.clamp(min=0.0)
        kd_loss = masked_kl * T * T

        # ── DIAGNOSTIC: 学生投影质量点 vs 教师 top-k 覆盖 ─────────────────
        # 回答“KL 为什么大”:学生投影 argmax 是否落在教师 top-k 内、教师
        # 是否认可该 token(q>1e-4)。三类结果:
        #   in_topk & q>1e-4  → 正常(教师认可学生主要质量点)
        #   in_topk & q<=1e-4 → 切分伪影(教师 top-k 含该 token 但给 ~0,
        #                        异架构 token 边界不同)
        #   out_topk           → 切片丢弃(学生质量点不在教师 top-k 内,
        #                        重归一化后分布被扭曲)
        # 跨 tokenizer 时预计 in_topk&q>1e-4 占比极低,说明 KL 触底是伪影。
        # 配置:XTOKEN_DEBUG_DIAG=1 开启(默认关闭,正式训练不打)。
        if os.environ.get('XTOKEN_DEBUG_DIAG', '0') == '1':
            with torch.no_grad():
                diag_proj = proj_C  # [B, C|S, V_t] 学生投影质量
                diag_q = tgt_C.exp().clamp(min=0.0)  # 教师概率(几何平均近似)
                proj_argmax = diag_proj.argmax(dim=-1)  # [B, C|S] 全词表索引
                proj_argmass = diag_proj.max(dim=-1).values  # [B, C|S]
                # 全词表索引 → top-k 子集内位置;不在子集内的映射为 -1
                # (该质量点被切片丢弃,教师概率按 0 计)
                topk_positions = torch.full(
                    (tkr_vocab_size,), -1, dtype=torch.long,
                    device=self.device,
                )
                topk_positions[topk_idx] = torch.arange(
                    k, device=self.device
                )
                pos_argmax = topk_positions[proj_argmax]  # [B, C|S]
                in_topk = pos_argmax >= 0  # [B, C|S]
                q_argmax = torch.where(
                    in_topk,
                    diag_q.gather(
                        dim=-1,
                        index=pos_argmax.clamp(min=0).unsqueeze(-1),
                    ).squeeze(-1),
                    torch.zeros_like(pos_argmax, dtype=diag_q.dtype),
                )  # [B, C|S]
                q_ok = q_argmax > 1e-4
                diag_valid = valid_mask.bool()
                n_diag = int(diag_valid.sum().item())
                if n_diag > 0:
                    n_in_topk = int((in_topk & diag_valid).sum().item())
                    n_q_ok = int((in_topk & q_ok & diag_valid).sum().item())
                    n_q_low = int((in_topk & ~q_ok & diag_valid).sum().item())
                    n_out = n_diag - n_in_topk
                    mean_mass = float(
                        (proj_argmass * diag_valid).sum().item() / n_diag
                    )

        # ── Diagnostic: teacher top-k coverage ───────────────────────────
        # 子集覆盖占比 = Σ exp(subset_logits - 全词表 logsumexp),
        # 用 logsumexp 避免分配全词表 softmax 张量(~2.1GB,与分块投影
        # 同一内存优化目标)
        with torch.no_grad():
            lse_full = torch.logsumexp(
                teacher_logits_approx / T, dim=-1
            )  # [B, T_t]
            teacher_topk_cov = (
                (teacher_topk_logits / T - lse_full.unsqueeze(-1))
                .exp().sum(dim=-1).mean().item()
            )

        # ── Accuracy ─────────────────────────────────────────────────────
        with torch.no_grad():
            teacher_k = torch.softmax(teacher_logits_approx[..., topk_idx] / T, dim=-1)
            if use_chunk_alignment:
                teacher_k, _ = _chunk_average_log_probs(
                    teacher_k.log(), teacher_chunk_id, max_chunks
                )
                teacher_k = teacher_k.exp()
            proj_top1 = projected_k.argmax(dim=-1)
            teach_top1 = teacher_k.argmax(dim=-1)
            matches = (proj_top1 == teach_top1) & valid_mask
            proj_acc = matches.sum().float() / valid_f.sum().clamp(min=1.0)

        metrics = {
            'kd_loss': kd_loss.item(),
            'proj_accuracy': proj_acc.item(),
            'proj_mass_min': proj_mass_min,
            'proj_mass_mean': proj_mass_mean,
            'teacher_topk_cov': teacher_topk_cov,
        }
        return kd_loss, metrics

    # ------------------------------------------------------------------
    # Chunk alignment helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_alignment_ids(
            ids_source, index_or_name
    ) -> Optional[torch.Tensor]:
        """Resolve input_ids from various sources to a [B, S] tensor.

        ``ids_source`` may be:
        - ``None`` → return ``None`` (no alignment)
        - ``list`` of ``[B, S]`` tensors per teacher → index by ``index_or_name``
        - ``torch.Tensor`` [B, S] (e.g. student input_ids from inputs dict)
        """
        if ids_source is None:
            return None
        if isinstance(ids_source, list):
            if isinstance(index_or_name, int) and 0 <= index_or_name < len(ids_source):
                t = ids_source[index_or_name]
                return t if isinstance(t, torch.Tensor) else torch.tensor(t)
            return None
        if isinstance(ids_source, torch.Tensor):
            return ids_source
        return None

    @staticmethod
    def _character_spans(
            token_ids: torch.Tensor,
            tokenizer: "PreTrainedTokenizer",
    ) -> list:
        """Build ``(start_char, end_char)`` spans for each position.

        Returns a list of ``(start, end)`` tuples per batch item,
        each inner list has length ``seq_len``.
        """
        spans_batch: list = []
        for b in range(token_ids.shape[0]):
            ids = token_ids[b].tolist()
            spans: list = []
            for pos in range(len(ids)):
                prefix = tokenizer.decode(
                    ids[: pos + 1], skip_special_tokens=False
                )
                if pos == 0:
                    prev_len = 0
                else:
                    prev = tokenizer.decode(
                        ids[:pos], skip_special_tokens=False
                    )
                    prev_len = len(prev)
                end = len(prefix)
                spans.append((prev_len, end))
            spans_batch.append(spans)
        return spans_batch

    def _build_chunk_ids(
            self,
            student_input_ids: torch.Tensor,
            teacher_input_ids: torch.Tensor,
            teacher_index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Build chunk-id tensors via character-span overlap.

        Each student position is its own chunk (``max_chunks = seq_len_s - 1``
        after shift).  Each teacher position is assigned to the student chunk
        with which it shares the greatest character overlap.

        Returns:
            ``(student_chunk_id, teacher_chunk_id, max_chunks)``.
            Both tensors have shape ``[B, seq_len - 1]`` (next-token shifted)
            with values in ``[-1, max_chunks)``; ``-1`` = unassigned.
        """
        device = student_input_ids.device
        batch_size = student_input_ids.shape[0]
        # Use shifted lengths (position i predicts token i+1)
        s_len = student_input_ids.shape[1] - 1
        t_len = teacher_input_ids.shape[1] - 1
        max_chunks = s_len  # one chunk per student token position

        teacher_tok = self.teacher_tokenizer_group[teacher_index]

        # Build character spans for original (unshifted) sequences, then drop
        # the first position to align with next-token shift.
        s_spans_full = self._character_spans(student_input_ids, self.student_tokenizer)
        t_spans_full = self._character_spans(teacher_input_ids, teacher_tok)

        s_chunk_id = torch.full(
            (batch_size, s_len), -1, dtype=torch.long, device=device
        )
        t_chunk_id = torch.full(
            (batch_size, t_len), -1, dtype=torch.long, device=device
        )

        for b in range(batch_size):
            # 跨度语义:位置 i 的分布预测 token i+1——chunk 对齐必须用
            # "预测目标(token i+1)的跨度"而非"上下文(token i)的跨度"。
            # 跨 tokenizer 切分不同(学生'你是'=1 token,教师'你'+'是'=2 tokens)
            # 时,上下文对齐会让教师 chunk 混入"预测'是'/'工程'"的位置,
            # 而学生 chunk 预测'工程师'——预测目标错开 → KL 触底。
            # shift 一位后用预测目标跨度:教师"预测'你'的位置"的预测目标
            # '你'与学生"你是"chunk 重叠 → 正确进 chunk(有稀释,不触底)。
            s_spans = s_spans_full[b][1:s_len + 1]  # token i+1 的跨度
            t_spans = t_spans_full[b][1:t_len + 1]  # token t_pos+1 的跨度

            # Each student chunk = its own index
            for s_pos in range(s_len):
                s_chunk_id[b, s_pos] = s_pos

            # Assign each teacher position to best-overlap student chunk
            for t_pos in range(t_len):
                t_start, t_end = t_spans[t_pos]
                best_s = -1
                best_overlap = 0
                for s_pos in range(s_len):
                    s_start, s_end = s_spans[s_pos]
                    overlap = max(0, min(s_end, t_end) - max(s_start, t_start))
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_s = s_pos
                t_chunk_id[b, t_pos] = best_s

        return s_chunk_id, t_chunk_id, max_chunks

    def _project_student_probs(
            self,
            student_probs: torch.Tensor,
            teacher_index: int,
            teacher_vocab_size: int,
    ) -> torch.Tensor:
        """Project student probs [B, S, V_s] to teacher vocab [B, S, V_t] via sparse W."""
        batch_size, seq_len, _ = student_probs.shape

        s_idx = self.projection_student_indices_list[teacher_index].to(student_probs.device)
        t_idx = self.projection_teacher_indices_list[teacher_index].to(student_probs.device)
        values = self.projection_values_list[teacher_index].to(student_probs.device).float()

        # Filter indices to valid range
        valid_mask = t_idx < teacher_vocab_size
        s_idx = s_idx[valid_mask]
        t_idx = t_idx[valid_mask]
        values = values[valid_mask]

        if s_idx.numel() == 0:
            return torch.zeros(batch_size, seq_len, teacher_vocab_size,
                               device=student_probs.device, dtype=student_probs.dtype)

        # 按 batch 分块投影:中间张量 [B, S, N] 随 batch 线性增长
        # (N_SAMPLES=4 时 batch=8、N≈30万,expanded_idx int64 ≈9.4GB +
        # weighted ≈4.7GB → NPU OOM)。逐样本串行把峰值内存降到 1/B,
        # 数学等价(scatter_add 对每行独立,计算量不变)。
        projected = torch.zeros(
            batch_size, seq_len, teacher_vocab_size,
            device=student_probs.device, dtype=student_probs.dtype,
        )
        for b in range(batch_size):
            selected = student_probs[b:b + 1].index_select(
                dim=-1, index=s_idx
            )  # [1, S, N]
            weighted = selected * values.unsqueeze(0).unsqueeze(0)  # [1, S, N]
            expanded_idx = t_idx.unsqueeze(0).unsqueeze(0).expand(
                1, seq_len, -1
            )  # [1, S, N]
            projected[b:b + 1].scatter_add_(
                dim=2, index=expanded_idx, src=weighted
            )

        return projected

    # ------------------------------------------------------------------
    # H-KL mode (gold loss with common/uncommon + ULD)
    # ------------------------------------------------------------------

    def _compute_hkl(
            self,
            student_logits: torch.Tensor,
            teacher_probs: torch.Tensor,
            labels: torch.Tensor,
            teacher_index: int,
            *,
            student_input_ids: Optional[torch.Tensor] = None,
            teacher_input_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """H-KL loss: common-KL + uncommon-L1 + ULD.

        Steps:
        1. Shift for next-token.
        2. Compute logprobs for student and teacher.
        3. Common tokens → KL。有 input_ids 时走字符跨度 chunk 对齐
           (教师 token 按重叠分配进学生位置 chunk,教师侧 chunk 平均;
           学生位置即 chunk 轴)——P-KL 的教训:跨 tokenizer 逐位置配对
           是错位的,common-KL 因此虚高。
        4. Uncommon tokens → L1 on top-k(教师侧同样 chunk 平均)。
        5. Weighted sum: gamma_kl*common_KL + gamma_l1*L1(NeMo 对齐,
           无 ULD 熵项),再乘 T*T。
        """
        T = self.temperature
        eps = 1e-10

        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher = teacher_probs[..., :-1, :].contiguous().to(
            shift_student_logits.device
        )
        shift_labels = labels[..., 1:].contiguous().to(shift_student_logits.device)

        # Tokenizer vocab size (sizes the exact-token map).  The model's
        # lm_head may have a larger out_features (HF padding), so slice
        # the teacher tensor to the tokenizer width.
        tkr_vocab_size = self.teacher_vocab_sizes[teacher_index]
        model_vocab_size = shift_teacher.shape[-1]
        if model_vocab_size > tkr_vocab_size:
            shift_teacher = shift_teacher[..., :tkr_vocab_size]

        # ── Chunk alignment (与 P-KL 同一机制) ────────────────────────
        use_chunk_alignment = (
            student_input_ids is not None and teacher_input_ids is not None
        )
        if use_chunk_alignment:
            s_chunk_id, t_chunk_id, max_chunks = self._build_chunk_ids(
                student_input_ids, teacher_input_ids, teacher_index,
            )
            s_chunk_id = s_chunk_id.to(shift_student_logits.device)
            t_chunk_id = t_chunk_id.to(shift_student_logits.device)
            # 有效 chunk:该学生位置有 ≥1 个教师 token 且学生标签有效
            t_sizes = (
                t_chunk_id.unsqueeze(-1)
                == torch.arange(max_chunks, device=t_chunk_id.device)
                .view(1, 1, -1)
            ).sum(dim=1).float()  # [B, C]
            valid = (t_sizes > 0) & (shift_labels != -100)  # [B, C]
        else:
            valid = (shift_labels != -100).float()  # [B, S-1]

        # Student log-probs on full vocab
        student_log_probs = F.log_softmax(shift_student_logits / T, dim=-1)
        student_probs = student_log_probs.exp()

        # Teacher log-probs (teacher_probs may be probs or raw logits)
        if shift_teacher.max() > 1.0 or shift_teacher.sum(dim=-1).min() < 0.5:
            teacher_log_probs = F.log_softmax(shift_teacher / T, dim=-1)
        else:
            teacher_log_probs = (shift_teacher + eps).log()

        # Get exact token map
        common_s = self._common_student_indices_list[teacher_index]
        common_t = self._common_teacher_indices_list[teacher_index]
        uncommon_s = self._uncommon_student_indices_list[teacher_index]
        uncommon_t = self._uncommon_teacher_indices_list[teacher_index]

        # 无对齐回退:截断到 min_seq_len 逐位置配对(旧行为)
        if not use_chunk_alignment:
            min_seq_len = min(
                student_log_probs.shape[1], teacher_log_probs.shape[1]
            )
            student_log_probs = student_log_probs[:, :min_seq_len]
            student_probs = student_probs[:, :min_seq_len]
            teacher_log_probs = teacher_log_probs[:, :min_seq_len]
            valid = valid[:, :min_seq_len]

        # --- Common KL (teacher side chunk-averaged when aligned) ---
        common_kl = torch.tensor(
            0.0, device=student_logits.device, requires_grad=True
        )
        if common_s.numel() > 0:
            s_common = student_log_probs[
                ..., common_s.to(student_logits.device)
            ]  # [B, C, N_c]
            t_common = teacher_log_probs[
                ..., common_t.to(student_logits.device)
            ]  # [B, T-1, N_c]
            if use_chunk_alignment:
                t_common, _ = _chunk_average_log_probs(
                    t_common, t_chunk_id, max_chunks
                )  # [B, C, N_c]

            if self.reverse_kl:
                kl_elem = F.kl_div(
                    t_common, s_common, reduction='none', log_target=True
                )
            else:
                kl_elem = F.kl_div(
                    s_common, t_common, reduction='none', log_target=True
                )

            kl_per_pos = kl_elem.sum(dim=-1)  # [B, C]
            common_kl = (
                (kl_per_pos * valid).sum() / valid.sum().clamp(min=1.0)
            )

        # --- Uncommon L1 (teacher side chunk-averaged when aligned) ---
        l1_uncommon = torch.tensor(
            0.0, device=student_logits.device, requires_grad=True
        )
        uncommon_topk = self.uncommon_topk

        if (uncommon_s.numel() > 0 or uncommon_t.numel() > 0) \
                and valid.sum() > 0:
            s_unc = student_probs[
                ..., uncommon_s.to(student_logits.device)
            ]  # [B, C, N_us]
            t_unc = teacher_log_probs.exp()[
                ..., uncommon_t.to(student_logits.device)
            ]  # [B, T-1, N_ut]
            if use_chunk_alignment:
                t_unc, _ = _chunk_average_log_probs(
                    t_unc, t_chunk_id, max_chunks
                )  # [B, C, N_ut]

            max_unc = min(
                s_unc.shape[-1], t_unc.shape[-1], uncommon_topk
            )
            if max_unc > 0:
                # Sort descending and truncate
                if s_unc.shape[-1] > max_unc:
                    s_unc = torch.topk(
                        s_unc, k=max_unc, dim=-1, largest=True
                    ).values
                if t_unc.shape[-1] > max_unc:
                    t_unc = torch.topk(
                        t_unc, k=max_unc, dim=-1, largest=True
                    ).values

                min_len = min(s_unc.shape[-1], t_unc.shape[-1])
                s_unc = s_unc[..., :min_len]
                t_unc = t_unc[..., :min_len]

                l1_per_pos = F.l1_loss(
                    s_unc, t_unc, reduction='none'
                ).sum(dim=-1)  # [B, C]
                l1_uncommon = (
                    (l1_per_pos * valid).sum() / valid.sum().clamp(min=1.0)
                )

        # Weighted combination
        # NeMo 对齐:无 ULD 项(H-KL = common-KL + uncommon-L1),
        # gamma_l1 = 1 - gamma_kl - gamma_uld(默认 0.5/0.0 → L1=0.5)
        gamma_l1 = max(0.0, 1.0 - self.gamma_kl - self.gamma_uld)
        kd_loss = (
                          self.gamma_kl * common_kl +
                          gamma_l1 * l1_uncommon
                  ) * T * T

        # Top-1 accuracy on common subset
        with torch.no_grad():
            if common_s.numel() > 0:
                s_c = student_log_probs[
                    ..., common_s.to(student_logits.device)
                ]
                t_c = teacher_log_probs[
                    ..., common_t.to(student_logits.device)
                ]
                if use_chunk_alignment:
                    t_c, _ = _chunk_average_log_probs(
                        t_c, t_chunk_id, max_chunks
                    )
                s_top1 = s_c.argmax(dim=-1)
                t_top1 = t_c.argmax(dim=-1)
                matches = (s_top1 == t_top1) & valid.bool()
                common_acc = (
                    matches.sum().float() / valid.sum().clamp(min=1.0)
                )
            else:
                common_acc = torch.tensor(0.0)

        metrics = {
            'kd_loss': kd_loss.item(),
            'kl_common': common_kl.item(),
            'l1_uncommon': l1_uncommon.item(),
            'proj_accuracy': common_acc.item(),
        }
        return kd_loss, metrics

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_mapping_statistics(self, teacher_index: int = 0) -> Dict:
        """Return statistics about the projection matrix."""
        self._ensure_projection_matrices_built()
        if teacher_index >= len(self.projection_student_indices_list):
            raise ValueError(f"No projection matrix for teacher {teacher_index}")

        student_indices = self.projection_student_indices_list[teacher_index]
        self.projection_values_list[teacher_index].float()

        nnz = student_indices.numel()
        total_elements = self.student_vocab_size * self.teacher_vocab_sizes[teacher_index]
        sparsity = 1.0 - (nnz / total_elements) if total_elements > 0 else 0.0

        # 精确匹配按构建时记录的集合判定(行归一化后权重不再是 1.0)
        exact_student_ids = (
            self._exact_matched_student_ids[teacher_index]
            if hasattr(self, '_exact_matched_student_ids')
               and len(self._exact_matched_student_ids) > teacher_index
            else set()
        )
        exact_matched = len(exact_student_ids)

        proj_student_ids = set(student_indices.tolist())
        multi_token_matched = len(proj_student_ids - exact_student_ids)

        unmatched = self.student_vocab_size - exact_matched - multi_token_matched

        return {
            'total_student_tokens': self.student_vocab_size,
            'exact_matched': exact_matched,
            'multi_token_matched': multi_token_matched,
            'unmatched': unmatched,
            'sparsity': sparsity,
        }