import torch
from collections.abc import Mapping, Sequence

def move_to_cpu_detach(obj):
    breakpoint()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    elif isinstance(obj, Mapping):
        return {k: move_to_cpu_detach(v) for k, v in obj.items()}
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if hasattr(obj, "_fields"): # namedtuple
            return type(obj)(*(move_to_cpu_detach(v) for v in obj))
        return type(obj)(move_to_cpu_detach(v) for v in obj)
    return obj