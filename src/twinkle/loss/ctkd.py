# Copyright (c) ModelScope Contributors. All rights reserved.
# https://arxiv.org/pdf/2605.21699
# X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation - 2605.21699
"""
CTKD (Cross-Tokenizer Knowledge Distillation) Loss Implementation.

This module implements the X-Token approach for knowledge distillation across
models with different tokenizers using a sparse projection matrix.

Reference:
    "X-Token: Projection-Guided Cross-Tokenizer Knowledge Distillation"
    (https://arxiv.org/pdf/2605.21699)
"""
from typing import TYPE_CHECKING, Dict, Optional, Tuple
import hashlib
import pickle

import torch
import torch.nn.functional as F

from twinkle.data_format import LossOutput
from twinkle.loss.base import Loss

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

# Global cache for projection matrices to avoid recomputation
_PROJECTION_MATRIX_CACHE = {}


class CTKDLoss(Loss):
    """
    Args:
        student_tokenizer: Tokenizer for the student model.
        teacher_tokenizer_group: List of tokenizers for the teacher models.
        teacher_weights: Optional list of weights for each teacher (default: equal weights).
        max_length: Maximum span length L for multi-token matching (default: 4).
        beta: Base weight β for projection (default: 0.9).
        gamma: Decay rate γ for multi-token weights (default: 0.1).
        loss_type: Type of KL loss to use - 'pkl' for P-KL or 'hkl' for H-KL (default: 'pkl').
        temperature: Temperature for softmax in KL divergence (default: 1.0).
        device: Device to place the projection matrices on (default: None).

    Example:
        >>> from transformers import AutoTokenizer
        >>> student_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
        >>> teacher_tokenizer1 = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> teacher_tokenizer2 = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
        >>> loss_fn = CTKDLoss(student_tokenizer, [teacher_tokenizer1, teacher_tokenizer2])
        >>> # Projection matrices are built automatically during initialization
        >>> print(len(loss_fn.projection_matrices))  # 2 (number of teachers)
    """
    def __init__(
            self,
            student_tokenizer: 'PreTrainedTokenizer',
            teacher_tokenizer_group: list,  # List of teacher tokenizers
            teacher_weights: Optional[list] = None,  # Optional weights for each teacher
            max_length: int = 4,
            beta: float = 0.9,
            gamma: float = 0.1,
            loss_type: str = None,  # Auto-select based on vocabulary coverage
            temperature: float = 1.0,
            device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer_group = teacher_tokenizer_group
        self.num_teachers = len(teacher_tokenizer_group)

        # Set teacher weights (default to equal weights)
        if teacher_weights is None:
            self.teacher_weights = [1.0 / self.num_teachers] * self.num_teachers
        else:
            if len(teacher_weights) != self.num_teachers:
                raise ValueError(f"Number of weights ({len(teacher_weights)}) must match number of teachers ({self.num_teachers})")
            # Normalize weights to sum to 1
            weight_sum = sum(teacher_weights)
            self.teacher_weights = [w / weight_sum for w in teacher_weights]

        self.max_length = max_length
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.device = device

        # Auto-select loss_type based on vocabulary coverage
        if loss_type is None:
            self.loss_type = self._auto_select_loss_type()
        else:
            self.loss_type = loss_type

        # Vocabulary sizes
        self.student_vocab_size = len(student_tokenizer)
        self.teacher_vocab_sizes = [len(tokenizer) for tokenizer in teacher_tokenizer_group]

        # Lazy initialization flags
        self._projection_matrices_built = False
        self.projection_matrices: list = []
        self.projection_student_indices_list: list = []
        self.projection_teacher_indices_list: list = []
        self._best_teacher_mappings: list = []

    def _auto_select_loss_type(self) -> str:
        """
        Automatically select loss_type based on vocabulary coverage.

        Coverage is defined as the intersection of student and teacher vocabularies
        divided by the union of vocabularies.

        When coverage is high (>= 0.7), use 'hkl' for better efficiency.
        When coverage is low (< 0.7), use 'pkl' for better accuracy.

        Returns:
            'pkl' or 'hkl'
        """
        # Calculate average coverage across all teachers
        coverages = []
        for teacher_tokenizer in self.teacher_tokenizer_group:
            coverage = self._calculate_vocab_coverage(self.student_tokenizer, teacher_tokenizer)
            coverages.append(coverage)

        avg_coverage = sum(coverages) / len(coverages)

        # Select loss_type based on coverage threshold
        if avg_coverage >= 0.7:  # High coverage: use H-KL
            return 'hkl'
        else:  # Low coverage: use P-KL
            return 'pkl'
    def _calculate_vocab_coverage(self, student_tokenizer, teacher_tokenizer) -> float:
        """
        Calculate vocabulary coverage between student and teacher tokenizers.

        Coverage = |intersection| / |union|

        Args:
            student_tokenizer: Student model tokenizer
            teacher_tokenizer: Teacher model tokenizer

        Returns:
            Coverage ratio between 0 and 1
        """
        # Get vocabulary sets
        student_vocab = set(student_tokenizer.get_vocab().keys())
        teacher_vocab = set(teacher_tokenizer.get_vocab().keys())

        # Calculate intersection and union
        intersection = student_vocab & teacher_vocab
        union = student_vocab | teacher_vocab

        # Avoid division by zero
        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)

    def _ensure_projection_matrices_built(self):
        """Ensure projection matrices are built (lazy initialization with caching)."""
        if self._projection_matrices_built:
            return


        # Check again inside the lock to avoid race condition
        if self._projection_matrices_built:
            return




        # Generate cache key based on tokenizer configurations
        cache_key = self._generate_cache_key()

        # Check if projection matrices are already cached
        if cache_key in _PROJECTION_MATRIX_CACHE:
            # Load from cache
            cached_data = _PROJECTION_MATRIX_CACHE[cache_key]
            self.projection_matrices = cached_data['projection_matrices']
            self.projection_student_indices_list = [t.to(self.device) if self.device is not None else t.clone() for t in cached_data['projection_student_indices_list']]
            self.projection_teacher_indices_list = [t.to(self.device) if self.device is not None else t.clone() for t in cached_data['projection_teacher_indices_list']]
            self.projection_values_list = [t.to(self.device) if self.device is not None else t.clone() for t in cached_data['projection_values_list']]
        else:
            # Build projection matrices
            self.projection_matrices = []
            self.projection_student_indices_list = []
            self.projection_teacher_indices_list = []
            self.projection_values_list = []
            for i, teacher_tokenizer in enumerate(self.teacher_tokenizer_group):
                self._build_projection_matrix_for_teacher(teacher_tokenizer, i)

            # Cache the built matrices
            _PROJECTION_MATRIX_CACHE[cache_key] = {
                'projection_matrices': self.projection_matrices,
                'projection_student_indices_list': self.projection_student_indices_list,
                'projection_teacher_indices_list': self.projection_teacher_indices_list,
                'projection_values_list': self.projection_values_list,
            }

        # For H-KL: precompute the best teacher token for each student token for each teacher
        if self.loss_type == 'hkl':
            self._best_teacher_mappings = []
            for i in range(self.num_teachers):
                self._build_best_teacher_mapping_for_teacher(i)

        self._projection_matrices_built = True

    def _generate_cache_key(self) -> str:
        """Generate a unique cache key based on tokenizer configurations."""
        # Create a hashable representation of the tokenizer configurations
        config_data = {
            'student_vocab': self.student_tokenizer.get_vocab(),
            'teacher_vocabs': [tokenizer.get_vocab() for tokenizer in self.teacher_tokenizer_group],
            'max_length': self.max_length,
            'beta': self.beta,
            'gamma': self.gamma,
        }

        # Use hash of the configuration data as cache key
        config_bytes = pickle.dumps(config_data)
        return hashlib.md5(config_bytes).hexdigest()

    def __call__(
        self,
        inputs,
        outputs,
        **kwargs,
    ) -> LossOutput:
        """Compute CTKD loss between student and multiple teacher models.

        Args:
            inputs: Dict containing 'input_ids' and 'labels' for student model.
            outputs: Dict containing 'logits' from student model.
            teacher_logits_group: List of teacher model logits for each teacher.
            teacher_topk_logprobs_group: List of teacher topk logprobs for each teacher.
            teacher_topk_indices_group: List of teacher topk indices for each teacher.
            **kwargs: Additional arguments.

        Returns:
            LossOutput with the computed loss and number of tokens.
        """
        # Ensure projection matrices are built (lazy initialization)
        try:
            self._ensure_projection_matrices_built()
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise
        # Extract student logits and labels
        student_logits = outputs.get('logits')
        if student_logits is None:
            raise ValueError("Student logits not found in outputs")

        student_labels = inputs.get('labels')
        if student_labels is None:
            raise ValueError("Student labels not found in inputs")

        # Extract teacher logits group
        teacher_logits_group = kwargs.get('teacher_logits_group')
        if teacher_logits_group is None:
            teacher_logits_group = outputs.get('teacher_logits_group')

        # Support topk format from vLLM for multiple teachers
        teacher_topk_logprobs_group = kwargs.get('teacher_topk_logprobs_group')
        teacher_topk_indices_group = kwargs.get('teacher_topk_indices_group')
        # If we have topk format but not full logits, convert topk to logits for each teacher
        if teacher_logits_group is None and teacher_topk_logprobs_group is not None and teacher_topk_indices_group is not None:
            if len(teacher_topk_logprobs_group) != self.num_teachers or len(teacher_topk_indices_group) != self.num_teachers:
                raise ValueError(f"Number of teachers in topk format ({len(teacher_topk_logprobs_group)}) must match number of teachers ({self.num_teachers})")

            teacher_logits_group = []
            for i in range(self.num_teachers):
                teacher_topk_logprobs = teacher_topk_logprobs_group[i]
                teacher_topk_indices = teacher_topk_indices_group[i]

                # Get vocabulary size for this teacher
                vocab_size = self.teacher_vocab_sizes[i]
                batch_size, seq_len, topk = teacher_topk_logprobs.shape

                # Create full logits tensor initialized with very negative values
                teacher_logits = torch.full(
                    (batch_size, seq_len, vocab_size),
                    -1e10,  # Very negative value for log-space
                    dtype=teacher_topk_logprobs.dtype,
                    device=student_logits.device if student_logits is not None else teacher_topk_logprobs.device
                )

                # Scatter the topk logprobs into the full logits tensor
                teacher_logits.scatter_(
                    dim=2,
                    index=teacher_topk_indices.to(teacher_logits.device),
                    src=teacher_topk_logprobs.to(teacher_logits.device)
                )
                teacher_logits_group.append(teacher_logits)

        if teacher_logits_group is None:
            raise ValueError("Teacher logits group not found in kwargs or outputs. "
                           "Provide either teacher_logits_group or (teacher_topk_logprobs_group + teacher_topk_indices_group)")

        if len(teacher_logits_group) != self.num_teachers:
            raise ValueError(f"Number of teacher logits ({len(teacher_logits_group)}) must match number of teachers ({self.num_teachers})")

        # Get labels
        labels = inputs.get('labels')
        if labels is None:
            raise ValueError("labels not found in inputs")
        # Compute loss for each teacher and apply weighted average
        total_loss = 0.0
        teacher_losses = []
        for i in range(self.num_teachers):
            teacher_logits = teacher_logits_group[i]
            weight = self.teacher_weights[i]

            if self.loss_type == 'pkl':
                teacher_loss = self._compute_pkl_loss(student_logits, teacher_logits, labels, teacher_index=i)
            elif self.loss_type == 'hkl':
                teacher_loss = self._compute_hkl_loss(student_logits, teacher_logits, labels, teacher_index=i)
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}. Use 'pkl' or 'hkl'")

            weighted_loss = weight * teacher_loss
            total_loss += weighted_loss
            teacher_losses.append({
                'teacher_index': i,
                'loss_type': self.loss_type,
                'raw_loss': teacher_loss.item(),
                'weight': weight,
                'weighted_loss': weighted_loss.item()
            })

        # Print detailed loss information
        print(f"\n=== CTKDLoss Detailed Breakdown ===")
        print(f"Total Teachers: {self.num_teachers}")
        print(f"Loss Type: {self.loss_type}")
        print("-" * 50)
        for loss_info in teacher_losses:
            print(f"Teacher {loss_info['teacher_index']}:")
            print(f"  Loss Type: {loss_info['loss_type']}")
            print(f"  Raw Loss: {loss_info['raw_loss']:.6f}")
            print(f"  Weight: {loss_info['weight']:.4f}")
            print(f"  Weighted Loss: {loss_info['weighted_loss']:.6f}")
        print("-" * 50)
        print(f"Total Loss: {total_loss.item():.6f}")
        print("=" * 50)

        num_tokens = labels.ne(-100).sum().item() if labels is not None else 0
        return LossOutput(loss=total_loss, num_tokens=num_tokens)


    def _build_projection_matrix_for_teacher(self, teacher_tokenizer, teacher_index):
        """
        Build the sparse projection matrix W for a specific teacher tokenizer.

        The construction follows the X-Token paper algorithm:

        Step 1: Initialize W[s,t] = 0 for all s in V_S, t in V_T

        Step 2: Exact match - For each student token s:
            - Decode s to text
            - If text matches a teacher token t's decoded text:
                W[s, t] = 1

        Step 3: Multi-token decoding match - For unmatched student tokens s:
            - Decode s to text
            - Encode text with teacher tokenizer -> (t[0], ..., t[ℓ-1])
            - If ℓ < L (max_length):
                For i in [0, ℓ-1]:
                    W[s, t[i]] = β * γ^i

        The resulting matrix maps student token probabilities to teacher token space,
        enabling KL divergence computation across different vocabularies.

        Note: This implementation directly builds sparse COO format to avoid
        memory issues with large vocabulary sizes.
        """
        teacher_vocab_size = len(teacher_tokenizer)

        # Check memory requirements for dense matrix (for reference only)
        matrix_size_gb = (self.student_vocab_size * teacher_vocab_size * 4) / (1024**3)

        # Use lists to store sparse matrix entries (COO format)
        # This avoids creating a huge dense matrix
        student_indices = []
        teacher_indices = []
        values = []

        # Get vocabulary mappings
        student_vocab = self.student_tokenizer.get_vocab()  # {token_str: token_id}
        teacher_vocab = teacher_tokenizer.get_vocab()

        # Track which student tokens have been matched
        matched_student_ids = set()

        # Step 2: Exact match - find tokens with identical text representation
        # Build a mapping from token text to teacher token id for efficient lookup
        teacher_token_text_to_id = {}
        for token_id in range(teacher_vocab_size):

            # Decode each teacher token to its text representation
            # skip_special_tokens=False to preserve special tokens
            token_text = teacher_tokenizer.decode(
                [token_id],
                skip_special_tokens=False
            ).strip()
            teacher_token_text_to_id[token_text] = token_id

        # Match student tokens to teacher tokens by text
        for student_id in range(self.student_vocab_size):

            # Decode student token to text
            student_token_text = self.student_tokenizer.decode(
                [student_id],
                skip_special_tokens=False
            ).strip()

            # Check if this text exists in teacher vocabulary
            if student_token_text in teacher_token_text_to_id:
                teacher_id = teacher_token_text_to_id[student_token_text]
                # Store in sparse format directly
                student_indices.append(student_id)
                teacher_indices.append(teacher_id)
                values.append(1.0)
                matched_student_ids.add(student_id)

        # Step 3: Multi-token decoding match for unmatched student tokens
        # For each unmatched student token, decode to text and re-encode with teacher tokenizer
        unmatched_count = 0
        for student_id in range(self.student_vocab_size):
            if student_id in matched_student_ids:
                continue  # Skip already matched tokens

            # Decode student token to raw text
            text = self.student_tokenizer.decode(
                [student_id],
                skip_special_tokens=False
            )

            # Skip empty text
            if not text or not text.strip():
                continue

            # Encode with teacher tokenizer
            teacher_token_ids = teacher_tokenizer.encode(
                text,
                add_special_tokens=False
            )

            # Get the length of encoded sequence
            seq_length = len(teacher_token_ids)

            # Only assign weights if sequence length < max_length (L)
            if seq_length > 0 and seq_length < self.max_length:
                for i, teacher_token_id in enumerate(teacher_token_ids):
                    # Weight follows exponential decay: β * γ^i
                    # Earlier tokens get higher weights
                    weight = self.beta * (self.gamma ** i)
                    # Store in sparse format directly
                    student_indices.append(student_id)
                    teacher_indices.append(teacher_token_id)
                    values.append(weight)
                    unmatched_count += 1

        # Convert to tensors
        student_indices_tensor = torch.tensor(student_indices, dtype=torch.long)
        teacher_indices_tensor = torch.tensor(teacher_indices, dtype=torch.long)
        values_tensor = torch.tensor(values, dtype=torch.float32).half()  # Use half precision

        # Store as separate tensors (COO sparse format)
        self.projection_student_indices_list.append(student_indices_tensor)
        self.projection_teacher_indices_list.append(teacher_indices_tensor)
        self.projection_values_list.append(values_tensor)

        # Store None for dense matrix to save memory
        self.projection_matrices.append(None)

        # Calculate actual memory usage
        sparse_memory_mb = (student_indices_tensor.numel() * 8 +
                           teacher_indices_tensor.numel() * 8 +
                           values_tensor.numel() * 2) / (1024**2)

        # Move to specified device if provided
        if self.device is not None:
            self.projection_student_indices_list[teacher_index] = self.projection_student_indices_list[teacher_index].to(self.device)
            self.projection_teacher_indices_list[teacher_index] = self.projection_teacher_indices_list[teacher_index].to(self.device)
            self.projection_values_list[teacher_index] = self.projection_values_list[teacher_index].to(self.device)

    def _build_best_teacher_mapping_for_teacher(self, teacher_index):
        """
        Build the best teacher token mapping for H-KL loss for a specific teacher.

        For each student token, find the teacher token with the highest projection weight:
            t* = argmax_{t' in V_T} W[s, t']
            constraint: W[s, t*] > 0

        This creates a one-to-one mapping for heuristic KL divergence computation.

        Note: This implementation uses sparse COO format to avoid memory issues.
        """
        if teacher_index >= len(self.projection_student_indices_list):
            raise ValueError(f"Projection matrix for teacher {teacher_index} must be built first")

        # Use sparse COO format data
        student_indices = self.projection_student_indices_list[teacher_index]
        teacher_indices = self.projection_teacher_indices_list[teacher_index]
        values = self.projection_values_list[teacher_index].float()

        # Initialize mapping with -1 (no mapping)
        best_teacher_mapping = torch.full(
            (self.student_vocab_size,),
            -1,
            dtype=torch.long
        )
        max_weights = torch.zeros(self.student_vocab_size, dtype=torch.float32)

        # Find the best teacher token for each student token
        # Process in chunks to avoid memory issues
        chunk_size = 100000
        for i in range(0, len(student_indices), chunk_size):
            chunk_student = student_indices[i:i+chunk_size]
            chunk_teacher = teacher_indices[i:i+chunk_size]
            chunk_values = values[i:i+chunk_size]

            # For each entry, check if it's the best weight for that student token
            for j in range(len(chunk_student)):
                s_id = chunk_student[j].item()
                t_id = chunk_teacher[j].item()
                w = chunk_values[j].item()

                if w > max_weights[s_id]:
                    max_weights[s_id] = w
                    best_teacher_mapping[s_id] = t_id

        self._best_teacher_mappings.append(best_teacher_mapping)


    def _compute_pkl_loss(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            labels: torch.Tensor,
            teacher_index: int = 0,
    ) -> torch.Tensor:
        """
        Compute P-KL (Partition-free KL) loss.

        P-KL projects the student distribution to teacher vocabulary space using
        the full projection matrix W, then computes KL divergence:

            p̃_S[t] = Σ_{s in V_S} W[s,t] * p_S[s]
            L_P = KL(p_T || p̃_S)

        This preserves the full distribution information and is suitable when
        critical tokens (e.g., numbers in math tasks) don't have exact matches.

        Args:
            student_logits: [batch, seq_len, student_vocab_size]
            teacher_logits: [batch, seq_len, teacher_vocab_size]
            labels: [batch, seq_len]

        Returns:
            Scalar loss value.
        """
        # Shift logits and labels for next-token prediction
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous().to(student_logits.device)
        shift_labels = labels[..., 1:].contiguous().to(student_logits.device)

        # Create loss mask
        loss_mask = (shift_labels != -100).float()

        # Compute probabilities with temperature
        student_probs = F.softmax(shift_student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)

        # Align sequence lengths by taking the minimum
        student_seq_len = student_probs.shape[1]
        teacher_seq_len = teacher_probs.shape[1]
        min_seq_len = min(student_seq_len, teacher_seq_len)

        # Use the same sequence length for both student and teacher probabilities
        student_probs = student_probs[:, :min_seq_len, :]
        teacher_probs = teacher_probs[:, :min_seq_len, :]
        loss_mask = loss_mask[:, :min_seq_len]

        # Project student probabilities to teacher vocabulary space
        # p̃_S[t] = Σ_s W[s,t] * p_S[s]
        # Instead of using sparse matrix multiplication (not supported on NPU),
        # we use scatter_add with stored indices and values

        batch_size, seq_len, student_vocab = student_probs.shape
        # Use actual teacher vocabulary size from teacher_probs tensor
        _, _, teacher_vocab_size = teacher_probs.shape

        # Move projection indices and values to device for the specific teacher
        student_indices = self.projection_student_indices_list[teacher_index].to(student_probs.device)
        teacher_indices = self.projection_teacher_indices_list[teacher_index].to(student_probs.device)
        proj_values = self.projection_values_list[teacher_index].to(student_probs.device).float()

        # Filter teacher indices to ensure they are within the valid range
        # This handles cases where the actual teacher vocab size differs from expected
        valid_mask = teacher_indices < teacher_vocab_size
        if not valid_mask.all():
            # Some teacher indices are out of bounds, filter them
            student_indices = student_indices[valid_mask]
            teacher_indices = teacher_indices[valid_mask]
            proj_values = proj_values[valid_mask]

        # Get student probabilities for the non-zero projection entries
        # student_probs: [batch, seq_len, student_vocab]
        # student_indices: [num_non_zero]
        # We need to gather: student_probs[:, :, student_indices]
        selected_student_probs = student_probs.index_select(
            dim=-1,
            index=student_indices
        )  # [batch, seq_len, num_non_zero]

        # Multiply by projection weights
        weighted_probs = selected_student_probs * proj_values.unsqueeze(0).unsqueeze(0)
        # [batch, seq_len, num_non_zero]

        # Scatter add to teacher vocabulary space
        projected_student_probs = torch.zeros(
            batch_size, seq_len, teacher_vocab_size,
            device=student_probs.device,
            dtype=student_probs.dtype
        )

        # Expand teacher_indices for scatter_add
        # teacher_indices: [num_non_zero] -> [batch, seq_len, num_non_zero]
        expanded_teacher_indices = teacher_indices.unsqueeze(0).unsqueeze(0).expand(
            batch_size, seq_len, -1
        )

        # Scatter add: accumulate weighted probabilities to teacher tokens
        # Ensure indices and source are on the same device as target
        projected_student_probs.scatter_add_(
            dim=2,
            index=expanded_teacher_indices.to(projected_student_probs.device),
            src=weighted_probs.to(projected_student_probs.device)
        )

        # Normalize projected probabilities
        projected_student_probs = projected_student_probs / (
                projected_student_probs.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Debug: Check dimensions before KL divergence
        if projected_student_probs.size(-1) != teacher_probs.size(-1):
            raise ValueError(
                f"Vocabulary dimension mismatch: projected_student_probs has size {projected_student_probs.size(-1)}, "
                f"teacher_probs has size {teacher_probs.size(-1)}. "
                f"Projection matrix was built for teacher vocab size {self.teacher_vocab_size}, "
                f"but actual teacher model has vocab size {teacher_probs.size(-1)}. "
                f"This suggests the teacher model used at runtime has a different vocabulary than the teacher tokenizer used for projection matrix construction."
            )

        # Compute KL divergence: KL(p_T || p̃_S)
        # KL(P||Q) = Σ P(x) * log(P(x)/Q(x))
        # Using F.kl_div: input=log(Q), target=P, reduction='none'
        log_projected_student = torch.log(projected_student_probs + 1e-8)
        kl_div = F.kl_div(
            log_projected_student,
            teacher_probs,
            reduction='none'
        ).sum(dim=-1)  # Sum over vocabulary dimension

        # Apply mask and average
        masked_kl = kl_div * loss_mask
        loss = masked_kl.sum() / (loss_mask.sum() + 1e-8)

        return loss


    def _compute_hkl_loss(
            self,
            student_logits: torch.Tensor,
            teacher_logits: torch.Tensor,
            labels: torch.Tensor,
            teacher_index: int = 0,
    ) -> torch.Tensor:
        """
        Compute H-KL (Heuristic KL) loss.

        H-KL uses the best teacher token mapping for each student token:
            t* = argmax_{t'} W[s, t']

        This creates a one-to-one mapping and is suitable when most critical
        tokens have exact matches in both vocabularies.

        Args:
            student_logits: [batch, seq_len, student_vocab_size]
            teacher_logits: [batch, seq_len, teacher_vocab_size]
            labels: [batch, seq_len]

        Returns:
            Scalar loss value.
        """
        # Shift logits and labels for next-token prediction
        shift_student_logits = student_logits[..., :-1, :].contiguous()
        shift_teacher_logits = teacher_logits[..., :-1, :].contiguous().to(student_logits.device)
        shift_labels = labels[..., 1:].contiguous().to(student_logits.device)

        # Create loss mask
        loss_mask = (shift_labels != -100).float()

        # Compute probabilities with temperature
        student_probs = F.softmax(shift_student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(shift_teacher_logits / self.temperature, dim=-1)

        # Align sequence lengths by taking the minimum
        student_seq_len = student_probs.shape[1]
        teacher_seq_len = teacher_probs.shape[1]
        min_seq_len = min(student_seq_len, teacher_seq_len)

        # Use the same sequence length for both student and teacher probabilities
        student_probs = student_probs[:, :min_seq_len, :]
        teacher_probs = teacher_probs[:, :min_seq_len, :]
        loss_mask = loss_mask[:, :min_seq_len]

        # Get best teacher token for each student token for the specific teacher
        best_mapping = self._best_teacher_mappings[teacher_index].to(student_probs.device)

        # Select student probabilities for mapped teacher tokens
        # student_probs: [batch, seq_len, student_vocab]
        # best_mapping: [student_vocab] -> teacher token id for each student token
        # We need to gather the mapped probabilities
        batch_size, seq_len, student_vocab = student_probs.shape
        teacher_vocab = teacher_probs.shape[-1]

        # Create output tensor for mapped student probabilities
        mapped_student_probs = torch.zeros(
            batch_size, seq_len, teacher_vocab,
            device=student_probs.device,
            dtype=student_probs.dtype
        )

        # For each student token with a valid mapping, add its probability to the mapped teacher token
        valid_mask = best_mapping >= 0  # [student_vocab]
        valid_student_ids = torch.where(valid_mask)[0]
        valid_teacher_ids = best_mapping[valid_student_ids]

        # Process in chunks to avoid memory issues with large vocabularies
        chunk_size = 10000  # Process 10k tokens at a time
        for i in range(0, len(valid_student_ids), chunk_size):
            chunk_student_ids = valid_student_ids[i:i+chunk_size]
            chunk_teacher_ids = valid_teacher_ids[i:i+chunk_size]

            # Scatter student probabilities to teacher vocabulary positions
            # Ensure indices and source are on the same device as target
            mapped_student_probs.scatter_add_(
                dim=2,
                index=chunk_teacher_ids.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).to(mapped_student_probs.device),
                src=student_probs[:, :, chunk_student_ids].to(mapped_student_probs.device)
            )

        # Normalize
        mapped_student_probs = mapped_student_probs / (
                mapped_student_probs.sum(dim=-1, keepdim=True) + 1e-8
        )

        # Compute KL divergence: KL(p_T || mapped_p_S)
        log_mapped_student = torch.log(mapped_student_probs + 1e-8)
        kl_div = F.kl_div(
            log_mapped_student,
            teacher_probs,
            reduction='none'
        ).sum(dim=-1)

        # Apply mask and average
        masked_kl = kl_div * loss_mask
        loss = masked_kl.sum() / (loss_mask.sum() + 1e-8)

        return loss


    def get_projection_matrix(self, teacher_index: int = 0) -> Optional[torch.Tensor]:
        """Return the projection matrix W for a specific teacher.

        Args:
            teacher_index: Index of the teacher model (default: 0).

        Returns:
            Tensor of shape [student_vocab_size, teacher_vocab_size] or None if using sparse format.
            Note: For large vocabularies, this returns None to save memory.
            Use get_sparse_projection_data() to get the sparse representation.
        """
        # Ensure projection matrices are built before accessing
        self._ensure_projection_matrices_built()

        if teacher_index >= len(self.projection_matrices):
            raise ValueError(f"Projection matrix for teacher {teacher_index} has not been built")
        return self.projection_matrices[teacher_index]

    def get_sparse_projection_data(self, teacher_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the sparse projection matrix data in COO format.

        Args:
            teacher_index: Index of the teacher model (default: 0).

        Returns:
            Tuple of (student_indices, teacher_indices, values) representing the sparse matrix.
        """
        # Ensure projection matrices are built before accessing
        self._ensure_projection_matrices_built()

        if teacher_index >= len(self.projection_student_indices_list):
            raise ValueError(f"Projection matrix for teacher {teacher_index} has not been built")

        return (
            self.projection_student_indices_list[teacher_index],
            self.projection_teacher_indices_list[teacher_index],
            self.projection_values_list[teacher_index],
        )


    def get_mapping_statistics(self, teacher_index: int = 0) -> Dict:
        """Return statistics about the projection matrix for a specific teacher.

        Args:
            teacher_index: Index of the teacher model (default: 0).

        Returns:
            Dict containing:
                - total_student_tokens: Total number of student tokens
                - exact_matched: Number of tokens with exact match (W[s,t]=1)
                - multi_token_matched: Number of tokens with multi-token mapping
                - unmatched: Number of tokens without any mapping
                - sparsity: Fraction of zero elements in the matrix
        """
        # Ensure projection matrices are built before accessing statistics
        self._ensure_projection_matrices_built()

        if teacher_index >= len(self.projection_student_indices_list):
            raise ValueError(f"Projection matrix for teacher {teacher_index} has not been built")

        # Use the stored indices and values (COO format) for the specific teacher
        student_indices = self.projection_student_indices_list[teacher_index]
        values = self.projection_values_list[teacher_index].float()  # Convert from half to float for comparison

        # Total number of non-zero elements
        nnz = student_indices.numel()
        total_elements = self.student_vocab_size * self.teacher_vocab_sizes[teacher_index]

        # Sparsity = fraction of zero elements
        sparsity = 1.0 - (nnz / total_elements)

        # Count exact matches (weight == 1)
        exact_match_mask = (values == 1.0)
        exact_matched_students = student_indices[exact_match_mask].unique()
        exact_matched = exact_matched_students.numel()

        # Count multi-token matches (0 < weight < 1)
        multi_token_mask = (values > 0) & (values < 1.0)
        multi_token_students = student_indices[multi_token_mask].unique()
        # Exclude students that already have exact matches
        multi_token_students = multi_token_students[
            ~multi_token_students.unsqueeze(1).eq(exact_matched_students.unsqueeze(0)).any(dim=1)
        ]
        multi_token_matched = multi_token_students.numel()

        # Count unmatched: total - exact_matched - multi_token_matched
        unmatched = self.student_vocab_size - exact_matched - multi_token_matched

        return {
            'total_student_tokens': self.student_vocab_size,
            'exact_matched': exact_matched,
            'multi_token_matched': multi_token_matched,
            'unmatched': unmatched,
            'sparsity': sparsity
        }