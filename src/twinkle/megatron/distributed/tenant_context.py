# Copyright (c) twinkle authors. All rights reserved.
"""
Tenant context management using ContextVar.

This module provides process-level tenant context that automatically
propagates through async calls and threads, eliminating the need to
manually pass tenant_id to every method.
"""

import contextvars
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, TypeVar

import torch.distributed as dist

# Global ContextVar for current tenant - process level
_current_tenant: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    'current_tenant', default=None
)


def get_current_tenant() -> Optional[str]:
    """Get the current tenant ID from context."""
    return _current_tenant.get()


def set_current_tenant(tenant_id: Optional[str]) -> contextvars.Token:
    """Set the current tenant ID in context."""
    return _current_tenant.set(tenant_id)


def require_tenant() -> str:
    """Get current tenant ID, raising error if not set."""
    tenant_id = _current_tenant.get()
    if tenant_id is None:
        raise RuntimeError(
            "No tenant context set. Use 'with tenant_scope(tenant_id):' or "
            "call 'initialize()' first."
        )
    return tenant_id


@contextmanager
def tenant_scope(tenant_id: str):
    """
    Context manager to set the current tenant for a block of code.
    
    Example:
        >>> with tenant_scope('user_a'):
        ...     model.forward(input)  # Uses user_a's LoRA
        ...     loss.backward()
        ...     ddp.finish_grad_sync()  # Only syncs user_a's gradients
    """
    token = _current_tenant.set(tenant_id)
    try:
        yield tenant_id
    finally:
        _current_tenant.reset(token)


def generate_tenant_id() -> str:
    """Generate a unique tenant ID."""
    return str(uuid.uuid4())[:8]


@dataclass
class TenantInfo:
    """
    Information about a registered tenant.
    
    This is a lightweight dataclass that stores tenant metadata,
    separate from DDP-specific state.
    """
    tenant_id: str
    adapter_name: str
    process_group: Optional[dist.ProcessGroup] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


F = TypeVar('F', bound=Callable)


def with_tenant_context(func: F) -> F:
    """
    Decorator that automatically uses the current tenant context.
    
    The decorated function should have an optional 'tenant_id' parameter.
    If not provided, it will use the current tenant from context.
    
    Example:
        >>> @with_tenant_context
        ... def finish_grad_sync(self, tenant_id: Optional[str] = None):
        ...     # tenant_id is automatically set from context if None
        ...     ...
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, tenant_id: Optional[str] = None, **kwargs):
        if tenant_id is None:
            tenant_id = require_tenant()
        return func(*args, tenant_id=tenant_id, **kwargs)
    
    return wrapper  # type: ignore
