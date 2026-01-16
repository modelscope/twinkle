# Copyright (c) twinkle authors. All rights reserved.
"""
Distributed training utilities for multi-tenant Megatron LoRA.

Core components:
- tenant_context: ContextVar-based tenant management
- tenant_manager: Tenant lifecycle (adapters, optimizers)
- multi_tenant_ddp: Per-tenant gradient buffers and sync
"""

from .multi_tenant_ddp import MultiTenantLoRADDP, TenantDDPState
from .tenant_context import (TenantInfo, generate_tenant_id,
                             get_current_tenant, require_tenant,
                             set_current_tenant, tenant_scope)
from .tenant_manager import TenantManager, TenantState

__all__ = [
    # Context
    'get_current_tenant',
    'set_current_tenant',
    'require_tenant',
    'tenant_scope',
    'generate_tenant_id',
    'TenantInfo',
    # Manager
    'TenantManager',
    'TenantState',
    # DDP
    'MultiTenantLoRADDP',
    'TenantDDPState',
]
