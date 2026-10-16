import numpy as np
from typing import Any, Dict, List

from twinkle import DeviceMesh
from twinkle.utils import pad_and_stack_tensors


def collect_tensor_dict(outputs: List[Dict[str, Any]], device_mesh: DeviceMesh) -> Dict[str, Any]:
    import torch
    if not outputs:
        return {}

    if len(outputs) == 1:
        return outputs[0]

    all_keys = set()
    for d in outputs:
        all_keys.update(d.keys())

    outputs = [r for i, r in enumerate(outputs) if i in device_mesh.get_collect_ranks()]
    result = {}
    for key in all_keys:
        values = [d[key] for d in outputs if key in d]

        if not values or all([v is None for v in values]):
            continue

        first_value = values[0]

        if isinstance(first_value, list):
            merged = []
            for v in values:
                if isinstance(v, list):
                    merged.extend(v)
                else:
                    merged.append(v)
            if isinstance(merged[0], torch.Tensor):
                merged = pad_and_stack_tensors(merged)
            result[key] = merged

        elif isinstance(first_value, torch.Tensor):
            result[key] = pad_and_stack_tensors(values)

        elif isinstance(first_value, dict):
            result[key] = collect_tensor_dict(values, device_mesh)

        elif isinstance(first_value, np.ndarray) and first_value.size > 1:
            raise NotImplementedError('Numpy array not supported for now.')

        else:
            result[key] = values

    if 'loss' in result and len(result['loss']) > 1:
        result['loss'] = np.mean(result['loss'])
    return result


def collect_dp_and_flatten(outputs: List[List[Any]], device_mesh: DeviceMesh) -> List[Any]:
    """Collect a ``slice_dp`` call whose per-worker result is a list: dedup by data rank, then flatten.

    Each worker returns the list of results for the data shard it was handed. ``slice_dp`` splits by
    *data rank*, and within one data rank several workers (TP / PP / ulysses peers) hold the same shard
    and produce identical outputs, so ``get_collect_ranks()`` nominates a single representative per data
    rank -- the same dedup ``collect_tensor_dict`` applies to tensors. Keeping only those and flattening
    in worker order yields each input's result exactly once, in the order it was submitted. DP and FSDP
    need no dedup (each of their ranks is its own data rank holding a distinct shard), and this handles
    them unchanged because ``get_collect_ranks()`` then returns every rank.
    """
    if not outputs:
        return []
    if len(outputs) == 1:
        return list(outputs[0])
    selected = [r for i, r in enumerate(outputs) if i in device_mesh.get_collect_ranks()]
    return [item for sublist in selected for item in sublist]
