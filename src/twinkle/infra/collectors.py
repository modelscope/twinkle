from typing import List, Dict, Any
import torch


def collect_tensor_dict(outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not outputs:
        return {}

    if len(outputs) == 1:
        return outputs[0]

    all_keys = set()
    for d in outputs:
        all_keys.update(d.keys())

    result = {}
    for key in all_keys:
        values = [d[key] for d in outputs if key in d]

        if not values:
            continue

        first_value = values[0]

        if isinstance(first_value, list):
            merged = []
            for v in values:
                if isinstance(v, list):
                    merged.extend(v)
                else:
                    merged.append(v)
            result[key] = merged

        elif isinstance(first_value, torch.Tensor):
            result[key] = _pad_and_stack_tensors(values)

        elif isinstance(first_value, dict):
            result[key] = collect_tensor_dict(values)

        else:
            result[key] = values

    return result


def _pad_and_stack_tensors(tensors: List[torch.Tensor], pad_value: float = 0) -> torch.Tensor:
    if not tensors:
        raise ValueError("Empty tensor list")

    if len(tensors) == 1:
        return tensors[0].unsqueeze(0)

    max_ndim = max(t.ndim for t in tensors)
    expanded_tensors = []
    for t in tensors:
        while t.ndim < max_ndim:
            t = t.unsqueeze(0)
        expanded_tensors.append(t)

    max_shape = []
    for dim in range(max_ndim):
        max_shape.append(max(t.shape[dim] for t in expanded_tensors))

    padded_tensors = []
    for t in expanded_tensors:
        if list(t.shape) == max_shape:
            padded_tensors.append(t)
        else:
            pad_params = []
            for dim in range(max_ndim - 1, -1, -1):
                pad_params.extend([0, max_shape[dim] - t.shape[dim]])
            padded = torch.nn.functional.pad(t, pad_params, value=pad_value)
            padded_tensors.append(padded)

    return torch.stack(padded_tensors, dim=0)