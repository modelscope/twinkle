# Copyright (c) ModelScope Contributors. All rights reserved.
from .base import Loss


class MegatronCrossEntropyLoss(Loss):
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
        import torch
        from megatron.core import parallel_state as mpu
        cp_size = mpu.get_context_parallel_world_size()
        labels_for_mask = inputs['labels']
        output_tensor = outputs
        # output_tensor is per-token loss [batch, seq]
        # Create loss mask from labels (ignore -100)
        loss_mask = (labels_for_mask != -100).float()

        # Flatten and compute mean
        losses = output_tensor.float().view(-1)
        loss_mask_flat = loss_mask.view(-1)

        # Compute local sum and count
        local_loss_sum = torch.sum(losses * loss_mask_flat)
        local_count = loss_mask_flat.sum()

        # For CP > 1, aggregate loss across CP ranks
        if cp_size > 1:
            # All-reduce the count across CP ranks
            total_count = local_count.clone()
            torch.distributed.all_reduce(
                total_count,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_context_parallel_group()
            )

            # All-reduce the loss sum
            total_loss_sum = local_loss_sum.clone()
            torch.distributed.all_reduce(
                total_loss_sum,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_context_parallel_group()
            )

            # Return global mean, divided by cp_size to counteract Megatron's multiplication
            loss = (total_loss_sum / total_count.clamp(min=1)) / cp_size
        else:
            loss = local_loss_sum / local_count.clamp(min=1)

        return loss, {'loss': loss.detach()}
