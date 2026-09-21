# Copyright (c) ModelScope Contributors. All rights reserved.
"""Request checks that are decidable before a task is enqueued.

Two checks behind one entry point (:func:`assert_request_supported`), both running in
the synchronous request path:

- the endpoint exists on this deployment's backend (else 501);
- no declared parameter belongs to a different backend (else 422).

Their shared property is what makes them belong together: each is answerable from the
request body plus this deployment's configuration, so a rejection costs one HTTP round
trip and runs the backend method on **zero** data-parallel ranks. Discovering the same
problems from a backend exception instead means the failure surfaces inside the NCCL
critical section, where some ranks have already done work.

Both read *declared* metadata, so neither can reject a valid request. A third,
heuristic check over passthrough key spellings was implemented and removed for failing
that bar -- see :mod:`.backend_compat` for the case that killed it.
"""
from .backend_compat import BackendCapability, assert_request_supported, resolve_backend

__all__ = [
    'BackendCapability',
    'assert_request_supported',
    'resolve_backend',
]
