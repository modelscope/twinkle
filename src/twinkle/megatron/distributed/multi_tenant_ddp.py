# Copyright (c) twinkle authors. All rights reserved.
"""
Multi-Tenant LoRA DDP for Megatron models.

This module provides a DDP implementation for multi-tenant LoRA training,
inheriting from Megatron's DistributedDataParallel.

Key Design:
1. Inherits from MegatronDDP for code reuse
2. Overrides buffer/bucket creation to be per-tenant
3. Uses ContextVar for automatic tenant resolution
4. Tenant lifecycle managed by TenantManager (separate concern)
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from .tenant_context import get_current_tenant, require_tenant, tenant_scope

logger = logging.getLogger(__name__)

try:
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
    from megatron.core.distributed.param_and_grad_buffer import (
        _ParamAndGradBuffer,
        partition_buckets,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig
    MEGATRON_AVAILABLE = True
except ImportError:
    MEGATRON_AVAILABLE = False

    # Fallback for type hints
    class MegatronDDP(nn.Module):
        pass


@dataclass
class TenantDDPState:
    """Per-tenant DDP state: buffers, bucket groups, hooks."""
    tenant_id: str
    params: List[nn.Parameter] = field(default_factory=list)
    buffers: List = field(default_factory=list)
    bucket_groups: List = field(default_factory=list)
    param_to_bucket_group: Dict[nn.Parameter,
                                object] = field(default_factory=dict)
    grad_accs: List = field(default_factory=list)
    process_group: Optional[dist.ProcessGroup] = None


class MultiTenantLoRADDP(MegatronDDP):
    """
    Multi-Tenant LoRA DDP inheriting from MegatronDDP.

    This class extends MegatronDDP to support per-tenant gradient buffers
    and communication. The key difference is that instead of creating
    buffers for all parameters at init, buffers are created dynamically
    for each tenant.

    Comparison with MegatronDDP:
    - MegatronDDP: Creates buffers for all requires_grad=True params at __init__
    - MultiTenantLoRADDP: Creates buffers per-tenant when add_tenant is called

    Usage:
        >>> # Create with frozen base model (no trainable params yet)
        >>> ddp = MultiTenantLoRADDP(config, ddp_config, model)
        >>>
        >>> # Add tenant (creates buffers for their LoRA params)
        >>> ddp.add_tenant('tenant_a', params_a, process_group_a)
        >>>
        >>> # Training uses current tenant context
        >>> with tenant_scope('tenant_a'):
        ...     ddp.zero_grad_buffer()  # Zeros tenant_a's buffers
        ...     output = ddp(input)
        ...     loss.backward()
        ...     ddp.finish_grad_sync()  # Syncs tenant_a's gradients
        >>>
        >>> # Remove tenant
        >>> ddp.remove_tenant('tenant_a')
    """
    def __init__(
        self,
        config: 'TransformerConfig',
        ddp_config: 'DistributedDataParallelConfig',
        module: nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional['ProcessGroupCollection'] = None,
    ):
        """
        Initialize MultiTenantLoRADDP.

        Unlike MegatronDDP, this does NOT create buffers at init.
        Buffers are created per-tenant via add_tenant().

        Args:
            config: Transformer config.
            ddp_config: DDP config.
            module: Model (base model should be frozen).
            disable_bucketing: Disable bucketing.
            pg_collection: Process group collection.
        """
        if not MEGATRON_AVAILABLE:
            raise ImportError('Megatron-Core is required')

        # Skip MegatronDDP's buffer creation by temporarily setting all params to not require grad
        original_requires_grad = {}
        for name, param in module.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = False

        # Call parent init (will create empty buffers since no params require grad)
        super().__init__(
            config=config,
            ddp_config=ddp_config,
            module=module,
            disable_bucketing=disable_bucketing,
            pg_collection=pg_collection,
        )

        # Restore requires_grad
        for name, param in module.named_parameters():
            param.requires_grad = original_requires_grad[name]

        # Per-tenant state
        self._tenant_states: Dict[str, TenantDDPState] = {}

        logger.info('MultiTenantLoRADDP initialized (no buffers yet)')

    def add_tenant(
        self,
        tenant_id: str,
        params: List[nn.Parameter],
        process_group: Optional[dist.ProcessGroup] = None,
        param_names: Optional[Dict[nn.Parameter, str]] = None,
    ):
        """
        Add a tenant with their gradient buffers.

        This creates per-tenant buffers and hooks, similar to what
        MegatronDDP.__init__ does but scoped to this tenant.

        Args:
            tenant_id: Unique tenant ID.
            params: Trainable parameters for this tenant.
            process_group: Process group for gradient sync.
            param_names: Param to name mapping for debugging.
        """
        if tenant_id in self._tenant_states:
            raise ValueError(f"Tenant '{tenant_id}' already exists")

        if not params:
            raise ValueError('No parameters provided')

        process_group = process_group or self.intra_dp_cp_group
        param_names = param_names or {}

        # Build param_names if not provided
        if not param_names:
            for name, param in self.module.named_parameters():
                if param in params:
                    param_names[param] = name

        # Create tenant state
        state = TenantDDPState(
            tenant_id=tenant_id,
            params=params,
            process_group=process_group,
        )

        # Initialize grad flags
        for param in params:
            param.grad_added_to_main_grad = False

        # Create buffers
        self._create_tenant_buffers(state, param_names)

        # Register hooks
        self._register_tenant_hooks(state)

        self._tenant_states[tenant_id] = state

        logger.info(f"Added tenant '{tenant_id}' with {len(params)} params, "
                    f'{len(state.bucket_groups)} bucket groups')

    def _create_tenant_buffers(
        self,
        state: TenantDDPState,
        param_names: Dict[nn.Parameter, str],
    ):
        """Create gradient buffers for a tenant."""
        # Group by dtype
        param_and_grad_dtype_to_params = {}
        param_and_grad_dtype_to_indices = {}

        for param in state.params:
            param_dtype = param.dtype
            grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

            key = (param_dtype, grad_dtype)
            if key not in param_and_grad_dtype_to_params:
                param_and_grad_dtype_to_params[key] = []
                param_and_grad_dtype_to_indices[key] = []

            param_and_grad_dtype_to_params[key].append(param)
            param_and_grad_dtype_to_indices[key].append(
                len(param_and_grad_dtype_to_params[key]) - 1)

        # Calculate gradient scaling
        if self.config.calculate_per_token_loss:
            gradient_scaling_factor = 1.0
        elif self.ddp_config.average_in_collective:
            gradient_scaling_factor = 1.0
        else:
            gradient_scaling_factor = 1.0 / state.process_group.size()

        # ProcessGroupCollection for buffer creation
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = self.tp_group
        pg_collection.dp_cp = state.process_group

        # Create buffers
        for (param_dtype,
             grad_dtype), params in param_and_grad_dtype_to_params.items():
            indices = param_and_grad_dtype_to_indices[(param_dtype,
                                                       grad_dtype)]

            buffer = _ParamAndGradBuffer(
                self.ddp_config,
                param_dtype,
                grad_dtype,
                params,
                state.process_group,
                self.bucket_size,
                param_names,
                gradient_scaling_factor,
                indices,
                getattr(self.ddp_config, 'nccl_ub', False),
                pg_collection,
            )
            state.buffers.append(buffer)

        # Create bucket groups
        state.bucket_groups = partition_buckets(
            state.buffers,
            force_single_bucket_group=(self.bucket_size is None),
        )

        # Build param to bucket group mapping
        for bucket_group in state.bucket_groups:
            for bucket in bucket_group.buckets:
                for param in bucket.params_list:
                    state.param_to_bucket_group[param] = bucket_group

    def _register_tenant_hooks(self, state: TenantDDPState):
        """Register backward hooks for a tenant."""
        for param in state.params:
            if param not in state.param_to_bucket_group:
                continue

            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(
                self._make_tenant_backward_hook(param, state))
            state.grad_accs.append(grad_acc)

    def _make_tenant_backward_hook(self, param: nn.Parameter,
                                   state: TenantDDPState):
        """Create backward hook for a tenant's parameter."""
        def hook(*unused):
            if param in state.param_to_bucket_group:
                if param.grad is not None and not param.grad_added_to_main_grad:
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    bucket_group = state.param_to_bucket_group[param]
                    if bucket_group.is_last_microbatch:
                        bucket_group.register_grad_ready(param)

        return hook

    def remove_tenant(self, tenant_id: str):
        """Remove a tenant and cleanup their resources."""
        if tenant_id not in self._tenant_states:
            raise KeyError(f"Tenant '{tenant_id}' not found")

        state = self._tenant_states.pop(tenant_id)

        # Clear hooks
        state.grad_accs.clear()

        # Clear buffers
        state.buffers.clear()
        state.bucket_groups.clear()
        state.param_to_bucket_group.clear()

        # Clear param attributes
        for param in state.params:
            if hasattr(param, 'main_grad'):
                delattr(param, 'main_grad')
            if hasattr(param, 'grad_added_to_main_grad'):
                delattr(param, 'grad_added_to_main_grad')

        logger.info(f"Removed tenant '{tenant_id}'")

    def _get_tenant_state(self,
                          tenant_id: Optional[str] = None) -> TenantDDPState:
        """Get state for tenant (uses context if not specified)."""
        tenant_id = tenant_id or require_tenant()
        if tenant_id not in self._tenant_states:
            raise KeyError(f"Tenant '{tenant_id}' not registered")
        return self._tenant_states[tenant_id]

    # ========== Override MegatronDDP methods to be tenant-aware ==========

    @contextmanager
    def no_sync(self, tenant_id: Optional[str] = None):
        """Disable gradient sync for a tenant."""
        state = self._get_tenant_state(tenant_id)
        for bucket_group in state.bucket_groups:
            bucket_group.is_last_microbatch = False
        try:
            yield
        finally:
            for bucket_group in state.bucket_groups:
                bucket_group.is_last_microbatch = True

    def start_grad_sync(self, tenant_id: Optional[str] = None):
        """Start gradient sync for a tenant."""
        state = self._get_tenant_state(tenant_id)
        for bucket_group in state.bucket_groups:
            bucket_group.start_grad_sync()

    def finish_grad_sync(self, tenant_id: Optional[str] = None):
        """Finish gradient sync for a tenant."""
        state = self._get_tenant_state(tenant_id)
        for bucket_group in state.bucket_groups:
            bucket_group.finish_grad_sync()

    def zero_grad_buffer(self, tenant_id: Optional[str] = None):
        """Zero gradient buffers for a tenant."""
        state = self._get_tenant_state(tenant_id)

        for param in state.params:
            param.grad_added_to_main_grad = False

        for buffer in state.buffers:
            buffer.reset()

        for bucket_group in state.bucket_groups:
            bucket_group.reset()

    def scale_gradients(self,
                        scaling_factor: float,
                        tenant_id: Optional[str] = None):
        """Scale gradients for a tenant."""
        state = self._get_tenant_state(tenant_id)
        for buffer in state.buffers:
            buffer.scale_gradients(scaling_factor)

    def broadcast_params(self, tenant_id: Optional[str] = None):
        """Broadcast parameters for a tenant."""
        state = self._get_tenant_state(tenant_id)
        for param in state.params:
            dist.broadcast(
                param.data,
                src=dist.get_global_rank(state.process_group, 0),
                group=state.process_group,
            )

    # ========== Utility ==========

    def has_tenant(self, tenant_id: str) -> bool:
        """Check if tenant exists."""
        return tenant_id in self._tenant_states

    def list_tenants(self) -> List[str]:
        """List all tenants."""
        return list(self._tenant_states.keys())

    def get_tenant_params(self,
                          tenant_id: Optional[str] = None
                          ) -> List[nn.Parameter]:
        """Get parameters for a tenant (requires valid tenant context)."""
        state = self._get_tenant_state(tenant_id)
        return state.params

    # Note: list_tenants() intentionally not exposed to prevent
    # information leakage between tenants. Use has_tenant() instead.
