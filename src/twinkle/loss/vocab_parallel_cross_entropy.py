# Copyright (c) twinkle authors. All rights reserved.
"""Vocabulary-parallel cross entropy loss for Megatron TP training.

When using Tensor Parallelism, the vocabulary dimension is sharded across TP ranks.
Standard CrossEntropyLoss will fail because labels may fall outside the local
vocab partition. This module provides vocab-parallel loss computation.
"""
from typing import Optional

import torch
import torch.nn.functional as F

from .base import Loss

try:
    from megatron.core import parallel_state as mpu
    from megatron.core import tensor_parallel
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False


class VocabParallelCrossEntropyLoss(Loss):
    """Cross entropy loss that handles vocabulary parallelism in Megatron.
    
    When using TP (Tensor Parallelism), the vocabulary is sharded across TP ranks.
    This loss uses Megatron's vocab_parallel_cross_entropy which correctly handles
    the distributed computation.
    
    NOTE: Labels are expected to be pre-shifted by the template (using np.roll).
    This loss does NOT perform additional shifting.
    
    Fallback: When Megatron is not available or TP=1, uses standard CrossEntropyLoss.
    """
    
    def __call__(self, inputs, outputs, **kwargs):
        logits = outputs['logits']
        labels = inputs['labels']
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, :-1].contiguous()
        
        if not MEGATRON_AVAILABLE:
            # Fallback to standard loss
            logits_2d = shift_logits.view(-1, shift_logits.shape[-1])
            labels_1d = shift_labels.view(-1)
            return F.cross_entropy(logits_2d, labels_1d, ignore_index=-100)
        
        tp_size = mpu.get_tensor_model_parallel_world_size()
        
        if tp_size == 1:
            # No TP, use standard cross entropy
            logits_2d = shift_logits.view(-1, shift_logits.shape[-1])
            labels_1d = shift_labels.view(-1)
            return F.cross_entropy(logits_2d, labels_1d, ignore_index=-100)
        
        # Use Megatron's vocab-parallel cross entropy
        # Megatron expects [seq, batch, vocab] format for logits
        # and [seq, batch] for labels
        
        # Transpose logits: [batch, seq-1, vocab] -> [seq-1, batch, vocab]
        logits_sbv = shift_logits.transpose(0, 1).contiguous()
        
        # Transpose labels: [batch, seq-1] -> [seq-1, batch]
        labels_sb = shift_labels.transpose(0, 1).contiguous()
        
        # Megatron's vocab_parallel_cross_entropy handles the TP sharding correctly
        # It returns per-token loss of shape [seq-1, batch]
        per_token_loss = tensor_parallel.vocab_parallel_cross_entropy(logits_sbv, labels_sb)
        
        # Transpose back: [seq-1, batch] -> [batch, seq-1]
        per_token_loss = per_token_loss.transpose(0, 1).contiguous()
        
        # Apply loss mask (ignore labels == -100)
        loss_mask = (shift_labels != -100).float()
        
        # Compute mean loss (only over non-masked positions)
        loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        
        return loss


class MegatronCrossEntropyLoss(VocabParallelCrossEntropyLoss):
    """Alias for VocabParallelCrossEntropyLoss.
    
    Use this when training with Megatron backend and TP > 1.
    """
    pass
