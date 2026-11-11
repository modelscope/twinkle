# Copyright (c) ModelScope Contributors. All rights reserved.
"""Placement records and optimizer-state moves shared by colocate offload."""
from typing import Tuple

import torch

_CACHE_RELEASE_COUNT = 0


def record_offload(strategy, *, model: bool, optimizer: bool) -> None:
    """Remember which pieces this rank moved, so reload restores only those."""
    strategy._offloaded_model = bool(model)
    strategy._offloaded_optimizer = bool(optimizer)


def release_caching_allocator(min_free_fraction: float | None = None) -> None:
    """Return unused caching-allocator blocks to the driver.

    Colocation can leave the parameters and optimizer resident. The backward peak is still
    reserved by this process, and a separate vLLM process only observes it through
    ``cudaMemGetInfo`` when it maps the KV cache. ``gc.collect`` drops tensors that are
    unreachable but not yet collected, which is what makes the following release see them.
    """
    import gc

    from twinkle.utils.framework import Torch

    if min_free_fraction is None or not torch.cuda.is_available():
        gc.collect()
        Torch.empty_cache()
        return
    # A colocated caller supplies the actual rollout reservation plus headroom.
    # Collect young cycles every round, older cycles periodically or under pressure.
    global _CACHE_RELEASE_COUNT
    _CACHE_RELEASE_COUNT += 1
    gc.collect(0)
    Torch.empty_cache()
    free, total = torch.cuda.mem_get_info()
    if _CACHE_RELEASE_COUNT % 16 == 1 or free < total * min_free_fraction:
        gc.collect()
        Torch.empty_cache()


def offload_record(strategy) -> Tuple[bool, bool]:
    return (
        bool(getattr(strategy, '_offloaded_model', False)),
        bool(getattr(strategy, '_offloaded_optimizer', False)),
    )


def clear_offload_record(strategy) -> None:
    strategy._offloaded_model = False
    strategy._offloaded_optimizer = False


def move_optimizer_state(optimizer, device) -> None:
    """Move Adam moments and other tensor state onto ``device``.

    ``module.to`` never sees ``optimizer.state``, so leaving it behind would keep most of
    the training memory on the GPU during a colocated rollout.
    """
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def move_grads(module, device) -> None:
    """``module.to`` moves parameters, not ``param.grad``."""
    for param in module.parameters():
        grad = getattr(param, 'grad', None)
        if torch.is_tensor(grad):
            param.grad = grad.to(device)


def parameter_device(module):
    for param in module.parameters():
        return param.device
    return torch.device('cpu')
