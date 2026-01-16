# Copyright (c) twinkle authors. All rights reserved.
"""
Tenant Manager for multi-tenant LoRA training.

This module provides tenant lifecycle management
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type

import torch
import torch.distributed as dist
import torch.nn as nn

from .tenant_context import (
    generate_tenant_id,
    get_current_tenant,
    require_tenant,
    set_current_tenant,
    tenant_scope,
)

logger = logging.getLogger(__name__)


from peft import LoraConfig, PeftModel

@dataclass
class TenantState:
    """
    State for a single tenant.
    
    Contains:
    - Identity: tenant_id, adapter_name
    - Training: optimizer, scheduler, params
    - Config: gradient accumulation, max grad norm
    """
    tenant_id: str
    adapter_name: str
    
    # Parameters
    params: List[nn.Parameter] = field(default_factory=list)
    param_names: Dict[nn.Parameter, str] = field(default_factory=dict)
    
    # Training components
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None
    
    # Training config
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Process group for this tenant
    process_group: Optional[dist.ProcessGroup] = None


class TenantManager:
    """
    Manages tenant lifecycle for multi-tenant training.
    
    Responsibilities:
    1. Tenant registration/deregistration
    2. LoRA adapter management
    3. Optimizer/scheduler creation
    4. Tenant context switching
    
    This class is decoupled from DDP - it only manages tenant metadata
    and training components, not gradient buffers or communication.
    
    Example:
        >>> manager = TenantManager(model)
        >>> 
        >>> # Initialize tenant
        >>> tenant_id = manager.initialize(
        ...     lora_config=LoraConfig(r=8),
        ...     optimizer_cls=AdamW,
        ... )
        >>> 
        >>> # Use tenant context
        >>> with manager.scope(tenant_id):
        ...     # All operations use this tenant
        ...     pass
        >>> 
        >>> # Cleanup
        >>> manager.finalize(tenant_id)
    """
    
    def __init__(
        self,
        model: nn.Module,
        default_process_group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize tenant manager.
        
        Args:
            model: Model with LoRA structure.
            default_process_group: Default process group for tenants.
        """
        self.model = model
        self.default_process_group = default_process_group
        self._tenants: Dict[str, TenantState] = {}
        
        # Callbacks for DDP integration
        self._on_add_callbacks: List[Callable[[TenantState], None]] = []
        self._on_remove_callbacks: List[Callable[[TenantState], None]] = []
    
    def register_add_callback(self, callback: Callable[[TenantState], None]):
        """Register callback to be called when tenant is added."""
        self._on_add_callbacks.append(callback)
    
    def register_remove_callback(self, callback: Callable[[TenantState], None]):
        """Register callback to be called when tenant is removed."""
        self._on_remove_callbacks.append(callback)
    
    def initialize(
        self,
        lora_config: Optional['LoraConfig'] = None,
        optimizer_cls: Type[torch.optim.Optimizer] = torch.optim.AdamW,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        scheduler_cls: Optional[Type] = None,
        scheduler_kwargs: Optional[Dict[str, Any]] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        process_group: Optional[dist.ProcessGroup] = None,
        adapter_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Initialize a new tenant.
        
        Args:
            lora_config: LoRA configuration.
            optimizer_cls: Optimizer class.
            optimizer_kwargs: Optimizer arguments.
            scheduler_cls: Scheduler class.
            scheduler_kwargs: Scheduler arguments.
            gradient_accumulation_steps: Steps to accumulate.
            max_grad_norm: Max gradient norm for clipping.
            process_group: Process group for gradient sync.
            adapter_name: Adapter name (defaults to tenant_id).
            tenant_id: Tenant ID (generated if not provided).
            
        Returns:
            The tenant ID.
        """
        tenant_id = tenant_id or generate_tenant_id()
        adapter_name = adapter_name or tenant_id
        process_group = process_group or self.default_process_group
        
        if tenant_id in self._tenants:
            raise ValueError(f"Tenant '{tenant_id}' already exists")
        
        # Add LoRA adapter
        if lora_config is not None and isinstance(self.model, PeftModel):
            lora_config.modules_to_save = None
            lora_config.bias = 'none'
            self.model.add_adapter(adapter_name, lora_config)
            logger.info(f"Added LoRA adapter '{adapter_name}'")
        
        # Find trainable params
        params = []
        param_names = {}
        
        for name, param in self.model.named_parameters():
            if f'.{adapter_name}.' in name and 'lora_' in name:
                param.requires_grad = True
                params.append(param)
                param_names[param] = name
        
        if not params:
            logger.warning(f"No trainable params found for tenant '{tenant_id}'")
        
        # Create optimizer
        optimizer_kwargs = optimizer_kwargs or {'lr': 1e-4}
        optimizer = optimizer_cls(params, **optimizer_kwargs) if params else None
        
        # Create scheduler
        scheduler = None
        if scheduler_cls and optimizer:
            scheduler_kwargs = scheduler_kwargs or {}
            scheduler = scheduler_cls(optimizer, **scheduler_kwargs)
        
        # Create state
        state = TenantState(
            tenant_id=tenant_id,
            adapter_name=adapter_name,
            params=params,
            param_names=param_names,
            optimizer=optimizer,
            scheduler=scheduler,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            process_group=process_group,
        )
        
        self._tenants[tenant_id] = state
        
        # Notify callbacks (for DDP integration)
        for callback in self._on_add_callbacks:
            callback(state)
        
        # Set as current tenant
        set_current_tenant(tenant_id)
        
        logger.info(
            f"Initialized tenant '{tenant_id}' with {len(params)} params "
            f"({sum(p.numel() for p in params):,} elements)"
        )
        
        return tenant_id
    
    def finalize(self, tenant_id: Optional[str] = None):
        """
        Finalize a tenant and cleanup resources.
        
        Args:
            tenant_id: Tenant to finalize. Uses current if None.
        """
        tenant_id = tenant_id or get_current_tenant()
        if not tenant_id or tenant_id not in self._tenants:
            return
        
        state = self._tenants.pop(tenant_id)
        
        # Notify callbacks (for DDP cleanup)
        for callback in self._on_remove_callbacks:
            callback(state)
        
        # Remove adapter
        if isinstance(self.model, PeftModel):
            try:
                self.model.delete_adapter(state.adapter_name)
            except Exception as e:
                logger.warning(f"Failed to delete adapter: {e}")
        
        # Clear context if current
        if get_current_tenant() == tenant_id:
            set_current_tenant(None)
        
        logger.info(f"Finalized tenant '{tenant_id}'")
    
    @contextmanager
    def scope(self, tenant_id: Optional[str] = None):
        """Context manager for tenant scope."""
        tenant_id = tenant_id or require_tenant()
        with tenant_scope(tenant_id):
            yield self.get(tenant_id)
    
    def get(self, tenant_id: Optional[str] = None) -> TenantState:
        """Get tenant state."""
        tenant_id = tenant_id or require_tenant()
        if tenant_id not in self._tenants:
            raise KeyError(f"Tenant '{tenant_id}' not found")
        return self._tenants[tenant_id]
    
    def has(self, tenant_id: str) -> bool:
        """Check if tenant exists."""
        return tenant_id in self._tenants
    
    def count(self) -> int:
        """Number of tenants (does not expose tenant IDs for privacy)."""
        return len(self._tenants)
    
    # Note: list() method intentionally not exposed to clients to prevent
    # information leakage. Only server-side code should enumerate tenants.
