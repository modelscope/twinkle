# Copyright (c) ModelScope Contributors. All rights reserved.
"""Placement records and optimizer-state moves shared by colocate offload."""
from typing import Tuple

import torch


def record_offload(strategy, *, model: bool, optimizer: bool) -> None:
    """Remember which pieces this rank moved, so reload restores only those."""
    strategy._offloaded_model = bool(model)
    strategy._offloaded_optimizer = bool(optimizer)


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
