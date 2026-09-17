# Copyright (c) ModelScope Contributors. All rights reserved.
"""Session-bound resource lifecycle utilities (adapters / processors).

Named ``session_resource`` (not ``lifecycle``) to avoid colliding with
``twinkle.server.lifecycle``, which is the *request* lifecycle (submit / retrieve
/ envelope). This package is the *resource* lifecycle: registration, heartbeat and
session-driven expiration of session-bound resources.
"""

from .adapter import AdapterManagerMixin
from .base import SessionResourceMixin
from .processor import ProcessorManagerMixin

__all__ = ['AdapterManagerMixin', 'ProcessorManagerMixin', 'SessionResourceMixin']
