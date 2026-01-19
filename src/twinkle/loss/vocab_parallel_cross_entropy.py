# Copyright (c) twinkle authors. All rights reserved.
"""Vocab-parallel cross entropy loss for Megatron backend with Tensor Parallelism."""
import torch

from .base import Loss


class VocabParallelCrossEntropyLoss(Loss):
    """Vocab-parallel cross entropy loss for Megatron training with TP > 1.
    
    This loss uses Megatron's tensor_parallel.vocab_parallel_cross_entropy to
    correctly compute cross entropy when vocabulary is sharded across TP ranks.
    
    NOTE: Labels are expected to be pre-shifted by the template (using np.roll).
    This loss does NOT perform additional shifting.
    
    Args:
        ignore_index: The label value to ignore when computing loss. Default: -100.
    """
    
    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
    
    def __call__(self, inputs, outputs, **kwargs):
        from megatron.core import tensor_parallel
        
        logits = outputs['logits']
        labels = inputs['labels']
        
        # Transpose: [batch, seq, vocab] -> [seq, batch, vocab]
        logits_sbv = logits.transpose(0, 1).contiguous()
        labels_sb = labels.transpose(0, 1).contiguous()
        
        # Compute vocab-parallel cross entropy
        per_token_loss = tensor_parallel.vocab_parallel_cross_entropy(logits_sbv, labels_sb)
        per_token_loss = per_token_loss.transpose(0, 1).contiguous()
        
        # Apply loss mask
        loss_mask = (labels != self.ignore_index).float()
        loss = (per_token_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        
        return loss


MegatronCrossEntropyLoss = VocabParallelCrossEntropyLoss
