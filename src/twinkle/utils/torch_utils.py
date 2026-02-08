from typing import Any, Union, Mapping, TYPE_CHECKING, List, Optional
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
    return F.pad(tensor, pad_tuple, mode="constant", value=pad_value)

# TODO: check and remove
def pad_2d_list_to_tensor(
    data_list: List[List],
    max_length: Optional[int] = None,
    pad_value: float = 0.0,
    left_pad: bool = False,
    dtype: 'torch.dtype' = None,
    device: Optional[Union[str, 'torch.device']] = None,
) -> 'torch.Tensor':
    """
    Pad a 2D list (e.g., list of logprobs) to a 2D tensor.
    
    Args:
        data_list: List of lists, each inner list can have different lengths
        max_length: Target length. If None, uses the max length in data_list
        pad_value: Value to use for padding
        left_pad: If True, pad on the left (right-align data); otherwise pad on the right
        dtype: Output tensor dtype
        device: Output tensor device
        
    Returns:
        Padded tensor of shape [batch, max_length]
    """
    import torch
    if dtype is None:
        dtype = torch.float32
    if not data_list:
        return torch.tensor([], dtype=dtype, device=device)
    
    # Find max length
    lengths = [len(item) if item is not None else 0 for item in data_list]
    data_max_len = max(lengths) if lengths else 0
    target_len = max_length if max_length is not None and max_length > data_max_len else data_max_len
    
    if target_len == 0:
        return torch.full((len(data_list), 0), pad_value, dtype=dtype, device=device)
    
    batch_size = len(data_list)
    result = torch.full((batch_size, target_len), pad_value, dtype=dtype, device=device)
    
    for i, item in enumerate(data_list):
        if item is None or len(item) == 0:
            continue
        seq_len = min(len(item), target_len)
        if torch.is_tensor(item):
            values = item[-seq_len:] if left_pad else item[:seq_len]
        else:
            values = torch.tensor(item[-seq_len:] if left_pad else item[:seq_len], dtype=dtype, device=device)
        
        if left_pad:
            result[i, -seq_len:] = values
        else:
            result[i, :seq_len] = values
    
    return result

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
        if mpu.get_tensor_model_parallel_world_size() > 1:
            return _vocab_parallel_selective_log_softmax(logits, index)
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
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
    from megatron.core import mpu
    tp_group = mpu.get_tensor_model_parallel_group()

    return fused_vocab_parallel_cross_entropy(logits, index, tp_group)
