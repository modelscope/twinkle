import numpy as np
from typing import Any, Dict, List

from twinkle import DeviceMesh
from twinkle.utils import pad_and_stack_tensors


def _data_rank_order(outputs: List[Any], device_mesh: DeviceMesh) -> List[Any]:
    """Reassemble ``slice_dp`` outputs in DATA-RANK order -- the order dispatch assigned shards.

    ``get_collect_ranks()`` nominates one representative global rank per data rank (returned sorted
    by GLOBAL rank, so its order is not the data-rank order); an actor spans ``stride`` consecutive
    global ranks (gpus_per_worker), so ``r // stride`` is its worker index and every rank of the
    actor shares its data rank. Selecting the nominated actors and sorting them by data rank
    inverts the dispatch exactly. The former membership filter compared WORKER indices against
    GLOBAL ranks (keeping the wrong subset under stride>1) and preserved WORKER order even at
    stride 1 -- which permutes rows whenever the mesh's fsdp dim precedes dp: data_rank =
    dp_rank*fsdp_ws + fsdp_rank is then a transpose of the global rank (fsdp2dp2: worker->data_rank
    [0,2,1,3]), so shards dispatched by data rank came back concatenated by worker index,
    misaligning every per-row channel (response-framed logps) against the input trajectory order.
    """
    stride = max(1, device_mesh.world_size // len(outputs))
    actors = {r // stride for r in device_mesh.get_collect_ranks()}
    ordered = sorted(actors, key=lambda i: device_mesh.get_data_rank_from_global_rank(i * stride))
    return [outputs[i] for i in ordered if i < len(outputs)]


def tensor_dict(outputs: List[Dict[str, Any]], device_mesh: DeviceMesh) -> Dict[str, Any]:
    import torch
    if not outputs:
        return {}

    if len(outputs) == 1:
        # A single output still carries the per-micro-batch list-of-tensors shape the multi-output
        # path below flattens (pad_and_stack) for keys like ``logps``: one worker forwarding a batch
        # splits it into micro-batches, so a lone worker's frame must merge the same way, or the
        # first consumer of that key sees a list where every other deployment shape sees a tensor.
        single = outputs[0]
        return {key: (pad_and_stack_tensors(value) if isinstance(value, list) and value
                      and isinstance(value[0], torch.Tensor) else value)
                for key, value in single.items()}

    all_keys = set()
    for d in outputs:
        all_keys.update(d.keys())

    outputs = _data_rank_order(outputs, device_mesh)
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
            if merged and isinstance(merged[0], torch.Tensor):
                merged = pad_and_stack_tensors(merged)
            result[key] = merged

        elif isinstance(first_value, torch.Tensor):
            result[key] = pad_and_stack_tensors(values)

        elif isinstance(first_value, dict):
            result[key] = tensor_dict(values, device_mesh)

        elif isinstance(first_value, str):
            # Constant metadata (frame declarations, fingerprints): workers must agree, and the
            # merged value is the constant itself -- never a per-worker list of copies.
            if any(v != first_value for v in values):
                raise ValueError(f'output key {key!r} disagrees across workers: {sorted(set(values))}')
            result[key] = first_value

        elif isinstance(first_value, np.ndarray) and first_value.size > 1:
            raise NotImplementedError('Numpy array not supported for now.')

        else:
            # One surviving worker keeps the bare value -- the counterpart of stacking a single
            # tensor, which adds no batch row either; several keep the per-worker list, the
            # scalar counterpart of that batch dimension. No key is special-cased: which
            # quantity a scalar reports, and how a caller folds it, is the caller's business.
            result[key] = values[0] if len(values) == 1 else values

    return result


def dp_flatten(outputs: List[List[Any]], device_mesh: DeviceMesh) -> List[Any]:
    """Collect a ``slice_dp`` call whose per-worker result is a list: dedup by data rank, then flatten.

    Each worker returns the list of results for the data shard it was handed. ``slice_dp`` splits by
    *data rank*, and within one data rank several workers (TP / PP / ulysses peers) hold the same shard
    and produce identical outputs, so ``get_collect_ranks()`` nominates a single representative per data
    rank -- the same dedup ``tensor_dict`` applies to tensors. Keeping only those and flattening in
    DATA-RANK order yields each input's result exactly once, in the order it was submitted (worker
    order would permute them whenever data_rank differs from the worker index, e.g. fsdp2dp2). DP and
    FSDP need no dedup (each of their ranks is its own data rank holding a distinct shard), and this
    handles them unchanged because ``get_collect_ranks()`` then returns every rank.
    """
    if not outputs:
        return []
    if len(outputs) == 1:
        return list(outputs[0])
    selected = _data_rank_order(outputs, device_mesh)
    return [item for sublist in selected for item in sublist]
