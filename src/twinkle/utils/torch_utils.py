from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union

if TYPE_CHECKING:
    import torch


def to_device(data: Any, device: Union[str, 'torch.device', int], non_blocking: bool = False) -> Any:
    """Move inputs to a device"""
    import torch
    if isinstance(data, Mapping):
        return type(data)({k: to_device(v, device, non_blocking) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(to_device(v, device, non_blocking) for v in data)
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=non_blocking)
    else:
        return data


def pad_sequence_to_length(
    tensor: 'torch.Tensor',
    max_seq_len: int,
    pad_value: float = 0.0,
    left_pad: bool = False,
) -> 'torch.Tensor':
    """
    Pad a 2D tensor in the last dimension to max_seq_len.

    Args:
        tensor: Input tensor of shape [batch, seq_len]
        max_seq_len: Target sequence length
        pad_value: Value to use for padding
        left_pad: If True, pad on the left; otherwise pad on the right

    Returns:
        Padded tensor of shape [batch, max_seq_len]
    """
    import torch.nn.functional as F
    if tensor.shape[-1] >= max_seq_len:
        return tensor
    pad_len = max_seq_len - tensor.shape[-1]
    # F.pad uses (left, right) for last dim
    pad_tuple = (pad_len, 0) if left_pad else (0, pad_len)
    return F.pad(tensor, pad_tuple, mode='constant', value=pad_value)


def selective_log_softmax(logits, index) -> 'torch.Tensor':
    """
    refer: trl/trainer/utils

    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    import torch
    import torch.nn.functional as F

    try:
        from megatron.core import parallel_state as mpu
        if mpu.get_tensor_model_parallel_world_size() >= 1:
            try:
                return _vocab_parallel_selective_log_softmax(logits, index)
            except Exception:
                import traceback
                print(traceback.format_exc())
    except Exception:
        pass
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficient approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def _vocab_parallel_selective_log_softmax(
    logits: 'torch.Tensor',
    index: 'torch.Tensor',
) -> 'torch.Tensor':
    from megatron.core import mpu
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    tp_group = mpu.get_tensor_model_parallel_group()

    return -fused_vocab_parallel_cross_entropy(logits, index, tp_group)
