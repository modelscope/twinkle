# Copyright (c) ModelScope Contributors. All rights reserved.
"""Backend-neutral communicators for standalone process groups.

Typical use::

    comm = create_communicator(Rendezvous(host, port), rank, world_size, device)
    comm.barrier()
    comm.broadcast(buf, src=0)
    comm.synchronize()
    comm.destroy()

The backend comes from ``Platform.communicator_backend()`` unless passed explicitly.
"""
from .base import DEFAULT_TIMEOUT, Communicator, ReduceOp, Rendezvous, Work
from .registry import (
                       available_backends,
                       create_communicator,
                       default_backend,
                       get_communicator_cls,
                       register_backend,
                       unregister_backend,
)

__all__ = [
    'DEFAULT_TIMEOUT',
    'Communicator',
    'ReduceOp',
    'Rendezvous',
    'Work',
    'available_backends',
    'create_communicator',
    'default_backend',
    'get_communicator_cls',
    'register_backend',
    'unregister_backend',
]
