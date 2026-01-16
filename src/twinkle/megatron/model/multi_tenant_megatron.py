# Copyright (c) twinkle authors. All rights reserved.
"""
Multi-Tenant Megatron Model for LoRA training.

This module integrates TenantManager and MultiTenantLoRADDP to provide
a complete multi-tenant training solution.
"""

import contextvars
import logging
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type

import torch
import torch.distributed as dist
import torch.nn as nn

from ..distributed.multi_tenant_ddp import MultiTenantLoRADDP
from ..distributed.tenant_context import (get_current_tenant, require_tenant,
                                          set_current_tenant, tenant_scope)
from ..distributed.tenant_manager import TenantManager, TenantState

logger = logging.getLogger(__name__)

try:
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False

try:
    from peft.tuners.lora import LoraLayer, LoraModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


class MegatronMultiAdapter:
    """
    Patches LoRA layers to use ContextVar-based adapter selection.

    This enables thread-safe multi-tenant training where each tenant's
    active adapter is determined by the current context.
    """

    _adapter_var: contextvars.ContextVar[
        Optional[str]] = contextvars.ContextVar('adapter_names', default=None)
    _patched: bool = False

    def __call__(self, module: nn.Module) -> nn.Module:
        """Patch LoRA layers."""
        if MegatronMultiAdapter._patched:
            return module

        self._patch_peft_lora()
        self._patch_twinkle_lora()

        module.set_current_adapter_name = MegatronMultiAdapter.set_current_adapter_name
        MegatronMultiAdapter._patched = True

        return module

    def _patch_peft_lora(self):
        """Patch PEFT's LoraLayer/LoraModel."""
        if not PEFT_AVAILABLE:
            return

        if getattr(LoraLayer, '_patched', False):
            return

        def get_active_adapter(*args):
            return MegatronMultiAdapter._adapter_var.get()

        def get_active_adapters(*args):
            adapter = MegatronMultiAdapter._adapter_var.get()
            return [adapter] if adapter else []

        LoraLayer.active_adapter = property(get_active_adapter)
        LoraLayer.active_adapters = property(get_active_adapters)
        LoraLayer.set_adapter = lambda self, x: None
        LoraLayer._patched = True

        LoraModel.active_adapter = property(get_active_adapter)
        LoraModel.active_adapters = property(get_active_adapters)
        LoraModel.set_adapter = lambda self, x: None
        LoraModel._patched = True

        logger.info('Patched PEFT LoraLayer/LoraModel')

    def _patch_twinkle_lora(self):
        """Patch Twinkle's LoraParallelLinear."""
        try:
            from twinkle.megatron.tuners.lora import LoraParallelLinear
            if hasattr(LoraParallelLinear, '_patched'):
                return

            def get_active_adapter(self):
                return MegatronMultiAdapter._adapter_var.get()

            def get_active_adapters(self):
                adapter = MegatronMultiAdapter._adapter_var.get()
                return [adapter] if adapter else []

            LoraParallelLinear.active_adapter = property(get_active_adapter)
            LoraParallelLinear.active_adapters = property(get_active_adapters)
            LoraParallelLinear._patched = True
            logger.info('Patched LoraParallelLinear')
        except ImportError:
            pass

    @staticmethod
    def set_current_adapter_name(name: Optional[str]):
        """Set current adapter."""
        MegatronMultiAdapter._adapter_var.set(name)

    @staticmethod
    def get_current_adapter_name() -> Optional[str]:
        """Get current adapter."""
        return MegatronMultiAdapter._adapter_var.get()


class MultiTenantMegatronModel(nn.Module):
    """
    Multi-tenant Megatron model wrapper.

    Combines:
    - TenantManager: Tenant lifecycle (adapters, optimizers)
    - MultiTenantLoRADDP: Per-tenant gradient sync
    - MegatronMultiAdapter: Context-based adapter selection

    Example:
        >>> model = MultiTenantMegatronModel(base_model, config)
        >>>
        >>> # Initialize tenant (creates adapter, buffers, optimizer)
        >>> tenant_id = model.initialize(lora_config=LoraConfig(r=8))
        >>>
        >>> # Training (uses current tenant automatically)
        >>> model.zero_grad()
        >>> output = model(input)
        >>> loss = compute_loss(output)
        >>> model.backward(loss)
        >>> model.finish_grad_sync()
        >>> model.step()
        >>>
        >>> # Cleanup
        >>> model.finalize()
    """
    def __init__(
        self,
        model: nn.Module,
        config: 'TransformerConfig',
        ddp_config: Optional['DistributedDataParallelConfig'] = None,
    ):
        """
        Initialize.

        Args:
            model: Base model with LoRA structure.
            config: Transformer config.
            ddp_config: DDP config.
        """
        super().__init__()

        if not MEGATRON_AVAILABLE:
            raise ImportError('Megatron-Core required')

        self.config = config
        self.ddp_config = ddp_config or DistributedDataParallelConfig(
            overlap_grad_reduce=True,
            use_distributed_optimizer=False,
        )

        # Patch LoRA layers for multi-tenant
        self._multi_adapter = MegatronMultiAdapter()
        self.model = self._multi_adapter(model)

        # Create DDP
        self._ddp = MultiTenantLoRADDP(
            config=self.config,
            ddp_config=self.ddp_config,
            module=self.model,
        )

        # Create tenant manager
        self._manager = TenantManager(
            model=self.model,
            default_process_group=mpu.get_data_parallel_group(
                with_context_parallel=True),
        )

        # Wire up callbacks
        self._manager.register_add_callback(self._on_tenant_added)
        self._manager.register_remove_callback(self._on_tenant_removed)

        logger.info('MultiTenantMegatronModel initialized')

    def _on_tenant_added(self, state: TenantState):
        """Called when tenant is added via manager."""
        self._ddp.add_tenant(
            tenant_id=state.tenant_id,
            params=state.params,
            process_group=state.process_group,
            param_names=state.param_names,
        )

    def _on_tenant_removed(self, state: TenantState):
        """Called when tenant is removed via manager."""
        if self._ddp.has_tenant(state.tenant_id):
            self._ddp.remove_tenant(state.tenant_id)

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self._ddp(*args, **kwargs)

    # ========== Tenant Lifecycle ==========

    def initialize(self, **kwargs) -> str:
        """
        Initialize a tenant.

        Args:
            **kwargs: Passed to TenantManager.initialize()

        Returns:
            Tenant ID.
        """
        return self._manager.initialize(**kwargs)

    def finalize(self, tenant_id: Optional[str] = None):
        """Finalize a tenant."""
        self._manager.finalize(tenant_id)

    @contextmanager
    def scope(self, tenant_id: Optional[str] = None):
        """Context manager for tenant scope."""
        with self._manager.scope(tenant_id) as state:
            # Also set adapter
            MegatronMultiAdapter.set_current_adapter_name(state.adapter_name)
            try:
                yield state
            finally:
                MegatronMultiAdapter.set_current_adapter_name(None)

    # ========== Training Operations ==========

    def zero_grad(self, tenant_id: Optional[str] = None):
        """Zero gradients."""
        tenant_id = tenant_id or require_tenant()
        state = self._manager.get(tenant_id)

        self._ddp.zero_grad_buffer(tenant_id)
        if state.optimizer:
            state.optimizer.zero_grad(set_to_none=True)

    def backward(self, loss: torch.Tensor, tenant_id: Optional[str] = None):
        """Backward pass."""
        tenant_id = tenant_id or require_tenant()
        state = self._manager.get(tenant_id)

        MegatronMultiAdapter.set_current_adapter_name(state.adapter_name)
        scaled_loss = loss / state.gradient_accumulation_steps
        scaled_loss.backward()

    @contextmanager
    def no_sync(self, tenant_id: Optional[str] = None):
        """Disable gradient sync."""
        with self._ddp.no_sync(tenant_id):
            yield

    def finish_grad_sync(self, tenant_id: Optional[str] = None):
        """Finish gradient sync."""
        self._ddp.finish_grad_sync(tenant_id)

    def clip_grad_norm(
        self,
        max_norm: Optional[float] = None,
        tenant_id: Optional[str] = None,
    ) -> torch.Tensor:
        """Clip gradients."""
        tenant_id = tenant_id or require_tenant()
        state = self._manager.get(tenant_id)
        max_norm = max_norm or state.max_grad_norm
        return torch.nn.utils.clip_grad_norm_(state.params, max_norm)

    def step(self, tenant_id: Optional[str] = None):
        """Optimizer step."""
        tenant_id = tenant_id or require_tenant()
        state = self._manager.get(tenant_id)
        if state.optimizer:
            state.optimizer.step()

    def lr_step(self, tenant_id: Optional[str] = None):
        """LR scheduler step."""
        tenant_id = tenant_id or require_tenant()
        state = self._manager.get(tenant_id)
        if state.scheduler:
            state.scheduler.step()

    def get_lr(self, tenant_id: Optional[str] = None) -> Optional[float]:
        """Get current LR."""
        tenant_id = tenant_id or require_tenant()
        state = self._manager.get(tenant_id)
        if state.optimizer:
            return state.optimizer.param_groups[0]['lr']
        return None

    # ========== Utilities ==========

    def tenant_count(self) -> int:
        """Get number of active tenants."""
        return self._manager.count()

    def has_tenant(self, tenant_id: str) -> bool:
        """Check if a specific tenant exists."""
        return self._manager.has(tenant_id)

    @property
    def ddp(self) -> MultiTenantLoRADDP:
        """Get DDP wrapper."""
        return self._ddp

    @property
    def manager(self) -> TenantManager:
        """Get tenant manager."""
        return self._manager

    @property
    def unwrapped_model(self) -> nn.Module:
        """Get unwrapped model."""
        return self.model
