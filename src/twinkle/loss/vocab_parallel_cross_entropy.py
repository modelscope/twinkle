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
    
    Fallback: When Megatron is not available or TP=1, uses standard CrossEntropyLoss.
    """
    
    def __call__(self, inputs, outputs, **kwargs):
        logits = outputs['logits']
        labels = inputs['labels']
        
        # Get dimensions
        # logits: [batch, seq, vocab] or [batch, seq, partition_vocab]
        # labels: [batch, seq]
        
        if not MEGATRON_AVAILABLE:
            # Fallback to standard loss
            logits_2d = logits.view(-1, logits.shape[-1])
            labels_1d = labels.view(-1)
            return F.cross_entropy(logits_2d, labels_1d, ignore_index=-100)
        
        tp_size = mpu.get_tensor_model_parallel_world_size()
        
        if tp_size == 1:
            # No TP, use standard cross entropy
            logits_2d = logits.view(-1, logits.shape[-1])
            labels_1d = labels.view(-1)
            return F.cross_entropy(logits_2d, labels_1d, ignore_index=-100)
        
        # Use Megatron's vocab-parallel cross entropy
        # Megatron expects [seq, batch, vocab] format for logits
        # and [seq, batch] for labels
        
        # Transpose logits: [batch, seq, vocab] -> [seq, batch, vocab]
        logits_sbv = logits.transpose(0, 1).contiguous()
        
        # Transpose labels: [batch, seq] -> [seq, batch]
        # Must be contiguous for Megatron's view() operations
        labels_sb = labels.transpose(0, 1).contiguous()
        
        # Megatron's vocab_parallel_cross_entropy handles the TP sharding correctly
        # It returns per-token loss of shape [seq, batch]
        per_token_loss = tensor_parallel.vocab_parallel_cross_entropy(logits_sbv, labels_sb)
        
        # Transpose back: [seq, batch] -> [batch, seq]
        per_token_loss = per_token_loss.transpose(0, 1).contiguous()
        
        # Apply loss mask (ignore labels == -100)
        loss_mask = (labels != -100).float()
        
        # Compute mean loss (only over non-masked positions)
        loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        
        return loss


class MegatronCrossEntropyLoss(VocabParallelCrossEntropyLoss):
    """Alias for VocabParallelCrossEntropyLoss.
    
    Use this when training with Megatron backend and TP > 1.
    """
    pass
