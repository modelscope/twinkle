# Copyright (c) twinkle authors. All rights reserved.
"""Megatron training strategy for distributed model parallelism."""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.distributed as dist

from .base import TrainStrategy

try:
    from twinkle import DeviceMesh
except ImportError:
    DeviceMesh = None

try:
    import megatron.core
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from packaging import version
    MEGATRON_AVAILABLE = True
    mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
except ImportError:
    MEGATRON_AVAILABLE = False
    mcore_013 = False


def check_megatron_available():
    """Check if Megatron-Core is available."""
    if not MEGATRON_AVAILABLE:
        raise ImportError(
            "Megatron-Core is not installed. Please install it with: "
            "pip install megatron-core"
        )


class MegatronStrategy(TrainStrategy):
    """Strategy for Megatron-Core based distributed training.
    
    Supports Tensor Parallel (TP), Pipeline Parallel (PP), Context Parallel (CP),
    Expert Parallel (EP), and Data Parallel (DP).
    
    This strategy integrates with twinkle's DeviceMesh to provide a unified
    interface for distributed training configuration.
    """

    def __init__(
        self,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        expert_tensor_parallel_size: Optional[int] = None,
        virtual_pipeline_model_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        params_dtype: Optional[str] = None,
        device_mesh: Optional['DeviceMesh'] = None,
        megatron_args: Optional[Dict[str, Any]] = None,
    ):
        """Initialize MegatronStrategy.
        
        Args:
            tensor_model_parallel_size: Degree of tensor model parallelism.
            pipeline_model_parallel_size: Degree of pipeline model parallelism.
            context_parallel_size: Degree of context parallelism.
            expert_model_parallel_size: Degree of expert model parallelism for MoE.
            expert_tensor_parallel_size: Degree of expert tensor parallelism.
            virtual_pipeline_model_parallel_size: Virtual pipeline model parallel size.
            sequence_parallel: Enable sequence parallelism.
            use_distributed_optimizer: Use Megatron's distributed optimizer.
            mixed_precision: Mixed precision mode.
            params_dtype: Parameter dtype string (e.g., 'bf16', 'fp32').
            device_mesh: Twinkle DeviceMesh for distributed configuration.
            megatron_args: Additional Megatron arguments.
        """
        check_megatron_available()
        
        # If device_mesh is provided, extract parallel sizes from it
        if device_mesh is not None:
            tensor_model_parallel_size = self._get_dim_from_mesh(device_mesh, 'tp', tensor_model_parallel_size)
            pipeline_model_parallel_size = self._get_dim_from_mesh(device_mesh, 'pp', pipeline_model_parallel_size)
            context_parallel_size = self._get_dim_from_mesh(device_mesh, 'cp', context_parallel_size)
            expert_model_parallel_size = self._get_dim_from_mesh(device_mesh, 'ep', expert_model_parallel_size)
            
        self.tp_size = tensor_model_parallel_size
        self.pp_size = pipeline_model_parallel_size
        self.cp_size = context_parallel_size
        self.ep_size = expert_model_parallel_size
        self.etp_size = expert_tensor_parallel_size or tensor_model_parallel_size
        self.vp_size = virtual_pipeline_model_parallel_size
        self.sequence_parallel = sequence_parallel
        self.use_distributed_optimizer = use_distributed_optimizer
        self.mixed_precision = mixed_precision
        self.params_dtype = params_dtype
        self.device_mesh = device_mesh
        self.megatron_args = megatron_args or {}
        
        self._initialized = False
        self._parallel_state = None
        
    @staticmethod
    def _get_dim_from_mesh(device_mesh: 'DeviceMesh', dim_name: str, default: int) -> int:
        """Get dimension size from device mesh.
        
        Args:
            device_mesh: The device mesh.
            dim_name: Name of the dimension.
            default: Default value if dimension not found.
            
        Returns:
            Dimension size.
        """
        if device_mesh is None:
            return default
        if hasattr(device_mesh, 'has_dim') and device_mesh.has_dim(dim_name):
            return device_mesh.get_dim_size(dim_name)
        return default

    @classmethod
    def from_device_mesh(
        cls,
        device_mesh: 'DeviceMesh',
        sequence_parallel: bool = False,
        use_distributed_optimizer: bool = True,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        **kwargs,
    ) -> 'MegatronStrategy':
        """Create MegatronStrategy from twinkle DeviceMesh.
        
        Args:
            device_mesh: Twinkle DeviceMesh with dimension names like 'tp', 'pp', 'cp', 'ep', 'dp'.
            sequence_parallel: Enable sequence parallelism.
            use_distributed_optimizer: Use Megatron's distributed optimizer.
            mixed_precision: Mixed precision mode.
            **kwargs: Additional arguments.
            
        Returns:
            MegatronStrategy instance.
        """
        return cls(
            device_mesh=device_mesh,
            sequence_parallel=sequence_parallel,
            use_distributed_optimizer=use_distributed_optimizer,
            mixed_precision=mixed_precision,
            **kwargs,
        )

    def initialize(self, **kwargs) -> None:
        """Initialize Megatron parallel state.
        
        Should be called after distributed process group is initialized.
        This sets up all the parallel groups for TP/PP/CP/EP/DP.
        """
        if self._initialized:
            return
            
        if not dist.is_initialized():
            # Initialize torch distributed if not already done
            dist.init_process_group(backend='nccl')
        
        world_size = dist.get_world_size()
        
        # Validate parallel configuration
        total_model_parallel = self.tp_size * self.pp_size * self.cp_size
        if world_size % total_model_parallel != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by "
                f"tp_size * pp_size * cp_size ({total_model_parallel})"
            )
        
        # Initialize Megatron parallel state
        init_kwargs = {
            'tensor_model_parallel_size': self.tp_size,
            'pipeline_model_parallel_size': self.pp_size,
            'context_parallel_size': self.cp_size,
        }
        
        if self.vp_size is not None:
            init_kwargs['virtual_pipeline_model_parallel_size'] = self.vp_size
            
        # Handle MoE parallelism
        if self.ep_size > 1:
            init_kwargs['expert_model_parallel_size'] = self.ep_size
            if mcore_013:
                init_kwargs['expert_tensor_parallel_size'] = self.etp_size
        
        parallel_state.initialize_model_parallel(**init_kwargs)
        
        self._parallel_state = parallel_state
        self._initialized = True
        
        # Set CUDA device
        local_rank = dist.get_rank() % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)

    def destroy(self) -> None:
        """Destroy parallel state and clean up resources."""
        if self._initialized and self._parallel_state is not None:
            self._parallel_state.destroy_model_parallel()
            self._initialized = False

    @property
    def tp_rank(self) -> int:
        """Get tensor parallel rank."""
        if not self._initialized:
            return 0
        return self._parallel_state.get_tensor_model_parallel_rank()

    @property
    def pp_rank(self) -> int:
        """Get pipeline parallel rank."""
        if not self._initialized:
            return 0
        return self._parallel_state.get_pipeline_model_parallel_rank()

    @property
    def dp_rank(self) -> int:
        """Get data parallel rank."""
        if not self._initialized:
            return 0
        return self._parallel_state.get_data_parallel_rank()

    @property
    def cp_rank(self) -> int:
        """Get context parallel rank."""
        if not self._initialized:
            return 0
        return self._parallel_state.get_context_parallel_rank()

    @property
    def ep_rank(self) -> int:
        """Get expert parallel rank."""
        if not self._initialized:
            return 0
        return self._parallel_state.get_expert_model_parallel_rank()

    @property
    def dp_size(self) -> int:
        """Get data parallel size."""
        if not self._initialized:
            world_size = dist.get_world_size() if dist.is_initialized() else 1
            return world_size // (self.tp_size * self.pp_size * self.cp_size)
        return self._parallel_state.get_data_parallel_world_size()

    @property
    def tp_group(self):
        """Get tensor parallel process group."""
        if not self._initialized:
            return None
        return self._parallel_state.get_tensor_model_parallel_group()

    @property
    def dp_group(self):
        """Get data parallel process group."""
        if not self._initialized:
            return None
        return self._parallel_state.get_data_parallel_group()

    @property
    def pp_group(self):
        """Get pipeline parallel process group."""
        if not self._initialized:
            return None
        return self._parallel_state.get_pipeline_model_parallel_group()

    @property
    def cp_group(self):
        """Get context parallel process group."""
        if not self._initialized:
            return None
        return self._parallel_state.get_context_parallel_group()

    @property
    def ep_group(self):
        """Get expert parallel process group."""
        if not self._initialized:
            return None
        return self._parallel_state.get_expert_model_parallel_group()

    def is_pipeline_first_stage(self) -> bool:
        """Check if current rank is pipeline first stage."""
        if not self._initialized:
            return True
        return self._parallel_state.is_pipeline_first_stage()

    def is_pipeline_last_stage(self) -> bool:
        """Check if current rank is pipeline last stage."""
        if not self._initialized:
            return True
        return self._parallel_state.is_pipeline_last_stage()

    def is_data_parallel_main_rank(self) -> bool:
        """Check if current rank is the main rank in data parallel group."""
        if not self._initialized:
            return True
        return self.dp_rank == 0

    def get_params_dtype(self) -> torch.dtype:
        """Get parameter dtype based on configuration.
        
        Returns:
            PyTorch dtype for model parameters.
        """
        if self.params_dtype is not None:
            dtype_map = {
                'fp32': torch.float32,
                'fp16': torch.float16,
                'bf16': torch.bfloat16,
            }
            return dtype_map.get(self.params_dtype, torch.bfloat16)
        
        if self.mixed_precision == 'bf16':
            return torch.bfloat16
        elif self.mixed_precision == 'fp16':
            return torch.float16
        return torch.float32

    def wrap_model(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Tuple[nn.Module, Optional[torch.optim.Optimizer]]:
        """Wrap model with distributed wrapper for data parallelism.
        
        In Megatron, TP/PP/CP/EP parallelism is already handled during model creation
        (via TransformerConfig and parallel_state). This method only handles Data
        Parallel (DP) wrapping, which synchronizes gradients across DP ranks.
        
        For PEFT/LoRA models:
        - We skip DDP wrapping to avoid compatibility issues
        - Gradients are synchronized manually via all_reduce_gradients()
        - This is more flexible and works reliably with dynamically added LoRA modules
        
        For full model training (non-PEFT):
        - Consider using Megatron's native training.setup_model_and_optimizer()
        - Or use Megatron DDP with proper TransformerConfig
        
        Args:
            model: The Megatron model to wrap (already parallelized via TP/PP).
            optimizer: Optional optimizer (not wrapped here; use DistributedOptimizer separately if needed).
            
        Returns:
            Tuple of (wrapped_model, wrapped_optimizer).
            For PEFT models, wrapped_model is the original model (no DDP wrapper).
        """
        if not self._initialized:
            self.initialize()
        
        # Check if this is a PEFT/LoRA model
        is_peft_model = hasattr(model, 'peft_config') or hasattr(model, 'base_model')
        
        if is_peft_model:
            # For PEFT models, skip DDP wrapping entirely.
            # Reasons:
            # 1. PEFT models have dynamically added modules that may cause issues with DDP
            # 2. LoRA typically has very few trainable parameters, so manual gradient sync is efficient
            # 3. Megatron DDP requires TransformerConfig which may not be accessible after PEFT wrapping
            # 4. PyTorch DDP has device placement issues when model uses CPU initialization
            #
            # Instead, gradients should be synchronized manually using all_reduce_gradients()
            # after backward() and before optimizer.step().
            return model, optimizer
        
        # For non-PEFT models, we can use Megatron DDP or PyTorch DDP
        dp_group = self.dp_group
        if dp_group is None or dist.get_world_size(dp_group) <= 1:
            # No DP needed (single GPU or no DP group)
            return model, optimizer
        
        # Get model config for Megatron DDP
        config = getattr(model, 'config', None)
        
        # Check if model is on GPU (required for DDP)
        model_device = next(model.parameters()).device
        if model_device.type == 'cpu':
            # Model is on CPU, need to move to GPU first
            # This happens when use_cpu_initialization=True
            local_rank = dist.get_rank() % torch.cuda.device_count()
            model = model.to(f'cuda:{local_rank}')
        
        if config is not None and hasattr(config, 'tensor_model_parallel_size'):
            # Model has TransformerConfig, use Megatron DDP
            try:
                from megatron.core.distributed import DistributedDataParallelConfig
                ddp_config = DistributedDataParallelConfig(
                    grad_reduce_in_fp32=True,
                    overlap_grad_reduce=False,
                    use_distributed_optimizer=self.use_distributed_optimizer,
                    check_for_nan_in_grad=False,
                    bucket_size=None,  # No bucketing for simpler gradient sync
                )
                wrapped_model = MegatronDDP(
                    config=config,
                    ddp_config=ddp_config,
                    module=model,
                )
                return wrapped_model, optimizer
            except (ImportError, TypeError) as e:
                # Fallback to PyTorch DDP if Megatron DDP fails
                pass
        
        # Fallback: PyTorch DDP for models without TransformerConfig
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        wrapped_model = TorchDDP(
            model,
            process_group=dp_group,
            # Note: Don't use device_ids for multi-GPU models or when model spans devices
        )
        
        return wrapped_model, optimizer

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        """Unwrap the distributed model to get the base model.
        
        Args:
            model: The wrapped model.
            
        Returns:
            The unwrapped base model.
        """
        if isinstance(model, MegatronDDP):
            return model.module
            
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        if isinstance(model, TorchDDP):
            return model.module
            
        return model

    def get_model_config(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_layers: int,
        ffn_hidden_size: Optional[int] = None,
        num_query_groups: Optional[int] = None,
        vocab_size: int = 32000,
        max_position_embeddings: int = 4096,
        num_experts: Optional[int] = None,
        moe_router_topk: int = 2,
        **kwargs,
    ):
        """Create a Megatron TransformerConfig.
        
        Args:
            hidden_size: Hidden dimension size.
            num_attention_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            ffn_hidden_size: FFN hidden size (default: 4 * hidden_size).
            num_query_groups: Number of KV heads for GQA.
            vocab_size: Vocabulary size.
            max_position_embeddings: Maximum sequence length.
            num_experts: Number of MoE experts.
            moe_router_topk: Top-k for MoE routing.
            **kwargs: Additional config arguments.
            
        Returns:
            Megatron TransformerConfig.
        """
        from megatron.core.transformer import TransformerConfig
        
        config = TransformerConfig(
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups or num_attention_heads,
            ffn_hidden_size=ffn_hidden_size or 4 * hidden_size,
            use_cpu_initialization=True,
            params_dtype=self.get_params_dtype(),
            tensor_model_parallel_size=self.tp_size,
            pipeline_model_parallel_size=self.pp_size,
            context_parallel_size=self.cp_size,
            expert_model_parallel_size=self.ep_size,
            sequence_parallel=self.sequence_parallel,
            num_moe_experts=num_experts,
            moe_router_topk=moe_router_topk,
            **kwargs,
        )
        
        return config
        
    def sync_gradients(self, model: Optional[nn.Module] = None) -> None:
        """Synchronize gradients across data parallel group.
        
        For DDP-wrapped models, gradients are synchronized automatically.
        For non-DDP models (e.g., PEFT models), this performs manual all-reduce.
        
        Args:
            model: Optional model to sync gradients for. If None, only barrier.
        """
        if not self._initialized:
            return
            
        dp_group = self.dp_group
        if dp_group is None:
            return
        
        dp_size = dist.get_world_size(dp_group)
        if dp_size <= 1:
            return
        
        if model is not None:
            # Manual gradient synchronization for non-DDP models (e.g., PEFT)
            self.all_reduce_gradients(model)
        else:
            # Just barrier for DDP models
            dist.barrier(dp_group)
    
    def all_reduce_gradients(self, model: nn.Module) -> None:
        """All-reduce gradients of trainable parameters across data parallel group.
        
        This is used for PEFT/LoRA models that are not wrapped with DDP.
        Gradients are averaged across all DP ranks.
        
        Args:
            model: The model whose gradients to synchronize.
        """
        if not self._initialized:
            return
            
        dp_group = self.dp_group
        if dp_group is None:
            return
            
        dp_size = dist.get_world_size(dp_group)
        if dp_size <= 1:
            return
        
        # Collect gradients from trainable parameters
        grads = []
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                grads.append(param.grad.data)
        
        if not grads:
            return
        
        # Flatten all gradients into a single tensor for efficient communication
        # This reduces the number of all-reduce operations
        flat_grads = torch.cat([g.contiguous().view(-1) for g in grads])
        
        # All-reduce and average
        dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM, group=dp_group)
        flat_grads.div_(dp_size)
        
        # Unflatten back to original gradient tensors
        offset = 0
        for grad in grads:
            numel = grad.numel()
            grad.copy_(flat_grads[offset:offset + numel].view_as(grad))
            offset += numel
        
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """All-reduce tensor across specified group.
        
        Args:
            tensor: Input tensor.
            op: Reduce operation.
            group: Process group (defaults to data parallel group).
            
        Returns:
            Reduced tensor.
        """
        if not self._initialized:
            return tensor
            
        if group is None:
            group = self.dp_group
            
        if group is not None:
            dist.all_reduce(tensor, op=op, group=group)
            
        return tensor
        
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int = 0,
        group: Optional[dist.ProcessGroup] = None,
    ) -> torch.Tensor:
        """Broadcast tensor from source rank.
        
        Args:
            tensor: Input tensor.
            src: Source rank.
            group: Process group (defaults to data parallel group).
            
        Returns:
            Broadcasted tensor.
        """
        if not self._initialized:
            return tensor
            
        if group is None:
            group = self.dp_group
            
        if group is not None:
            dist.broadcast(tensor, src=src, group=group)
            
        return tensor

    def get_parallel_info(self) -> Dict[str, Any]:
        """Get parallelism configuration information.
        
        Returns:
            Dict with parallel configuration details.
        """
        return {
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
            'cp_size': self.cp_size,
            'ep_size': self.ep_size,
            'etp_size': self.etp_size,
            'vp_size': self.vp_size,
            'dp_size': self.dp_size,
            'sequence_parallel': self.sequence_parallel,
            'use_distributed_optimizer': self.use_distributed_optimizer,
            'mixed_precision': self.mixed_precision,
            'tp_rank': self.tp_rank,
            'pp_rank': self.pp_rank,
            'dp_rank': self.dp_rank,
            'cp_rank': self.cp_rank,
            'ep_rank': self.ep_rank,
        }
        
    def __repr__(self) -> str:
        return (
            f"MegatronStrategy(tp={self.tp_size}, pp={self.pp_size}, "
            f"cp={self.cp_size}, ep={self.ep_size}, dp={self.dp_size}, "
            f"sequence_parallel={self.sequence_parallel})"
        )
