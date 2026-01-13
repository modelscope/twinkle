# Copyright (c) twinkle authors. All rights reserved.
"""Megatron-Core model wrapper for twinkle training framework."""
import contextlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

import twinkle
from twinkle import remote_class, remote_function, template, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss, MegatronCrossEntropyLoss
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils.plugin import Plugin
from .base import TwinkleModel
from .strategy import MegatronStrategy

try:
    import megatron.core
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel as MegatronDDP
    from packaging import version
    MEGATRON_AVAILABLE = True
    mcore_013 = version.parse(megatron.core.__version__) >= version.parse('0.13.0rc0')
except ImportError:
    MEGATRON_AVAILABLE = False
    mcore_013 = False


@dataclass
class MegatronOptimizerGroup:
    """Optimizer group for Megatron training.
    
    Similar to OptimizerGroup but adapted for Megatron's distributed training.
    """
    adapter_name: str = None
    adapter_config: Any = None
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None
    inputs: Dict[str, Any] = None
    outputs: Dict[str, Any] = None
    loss_instance: Loss = None
    loss_value: Any = None
    template: Template = None
    processor: InputProcessor = None
    gradient_accumulation_steps: int = 1
    cur_step: int = 0
    dp_group = None

    def do_grad_sync(self, gradient_accumulation_steps: Optional[int] = None) -> bool:
        """Check if gradient synchronization should happen."""
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        return self.cur_step % gradient_accumulation_steps == 0 and self.cur_step > 0


_default_adapter_name = ''


def check_megatron_available():
    """Check if Megatron-Core is available."""
    if not MEGATRON_AVAILABLE:
        raise ImportError(
            "Megatron-Core is not installed. Please install it with: "
            "pip install megatron-core"
        )


@remote_class(execute='all')
class MegatronModel(TwinkleModel, nn.Module):
    """Megatron-Core model wrapper for twinkle training framework.
    
    Note: Uses execute='all' to create workers on all ranks, which is required
    for Megatron's TP/DP parallelism where all ranks must participate in
    collective operations like gradient all-reduce.
    
    This class provides a similar API to TransformersModel but uses Megatron-Core
    as the training backend, supporting TP/PP/CP/EP parallelism.
    
    Args:
        pretrained_model_name_or_path: HuggingFace model path or ID.
        device_mesh: Twinkle DeviceMesh for distributed training.
        tensor_model_parallel_size: Tensor parallel size.
        pipeline_model_parallel_size: Pipeline parallel size.
        context_parallel_size: Context parallel size.
        expert_model_parallel_size: Expert parallel size.
        sequence_parallel: Enable sequence parallelism.
        mixed_precision: Mixed precision mode.
        use_distributed_optimizer: Use Megatron's distributed optimizer.
        **kwargs: Additional arguments passed to model initialization.
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        device_mesh: Optional[DeviceMesh] = None,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        sequence_parallel: bool = False,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        use_distributed_optimizer: bool = True,
        load_weights: bool = True,
        use_megatron_bridge: bool = True,  # Use bridge-based initialization (recommended)
        recompute_granularity: Optional[str] = 'selective',  # Activation checkpointing
        recompute_modules: Optional[list] = None,  # Modules to recompute
        **kwargs,
    ):
        check_megatron_available()
        nn.Module.__init__(self)
        
        self.model_id = pretrained_model_name_or_path
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self.use_megatron_bridge = use_megatron_bridge
        self.recompute_granularity = recompute_granularity
        self.recompute_modules = recompute_modules
        
        # Load HuggingFace config first
        model_path = HubOperation.download_model(pretrained_model_name_or_path)
        self._load_hf_config(model_path)
        
        # Store model_path for later use
        self._model_path = model_path
        
        # Create Megatron strategy
        self.strategy = MegatronStrategy(
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            sequence_parallel=sequence_parallel,
            use_distributed_optimizer=use_distributed_optimizer,
            mixed_precision=mixed_precision,
        )
        
        # Initialize parallel state (skip if using bridge init, as it handles this)
        if not use_megatron_bridge:
            self.strategy.initialize()
        
        # Create Megatron model
        self.model = self._create_megatron_model(model_path, load_weights, **kwargs)
        
        self._model_wrapped = False
        # This correctly handles vocab sharding in Tensor Parallelism
        self.optimizer_group: Dict[str, MegatronOptimizerGroup] = {
            _default_adapter_name: MegatronOptimizerGroup(loss_instance=MegatronCrossEntropyLoss())
        }
        
    def _load_hf_config(self, model_path: str):
        """Load HuggingFace model config."""
        from transformers import AutoConfig
        self.hf_config = AutoConfig.from_pretrained(model_path)
        
    def _create_megatron_model(
        self,
        model_path: str,
        load_weights: bool = True,
        **kwargs,
    ) -> nn.Module:
        """Create Megatron model from HuggingFace checkpoint.
        
        Args:
            model_path: Path to HuggingFace model.
            load_weights: Whether to load weights.
            **kwargs: Additional arguments.
            
        Returns:
            Megatron model on GPU.
        """
        params_dtype = torch.bfloat16
        if self.mixed_precision == 'fp16':
            params_dtype = torch.float16
        elif self.mixed_precision == 'no':
            params_dtype = torch.float32
        
        if self.use_megatron_bridge:
            # Use bridge-based initialization (recommended)
            # This ensures all patches are applied and config is correctly generated
            return self._create_megatron_model_with_bridge(model_path, load_weights, params_dtype, **kwargs)
        else:
            # Use twinkle's native initialization
            return self._create_megatron_model_native(model_path, load_weights, params_dtype, **kwargs)
    
    def _create_megatron_model_with_bridge(
        self,
        model_path: str,
        load_weights: bool,
        params_dtype: torch.dtype,
        **kwargs,
    ) -> nn.Module:
        """Create Megatron model using bridge-based initialization flow.
        
        This approach uses TwinkleBridgeInitializer for independent initialization
        It includes:
        - Proper config conversion from HuggingFace to Megatron format
        - Correct Megatron initialization (initialize_megatron)
        - Correct model creation
        - Weight loading with TwinkleGPTBridge
        
        Args:
            model_path: Path to HuggingFace model.
            load_weights: Whether to load weights.
            params_dtype: Parameter dtype.
            **kwargs: Additional arguments.
            
        Returns:
            Megatron model on GPU.
        """
        from twinkle.megatron.model.bridge import TwinkleBridgeInitializer
        
        # Create bridge-based initializer
        self._bridge_initializer = TwinkleBridgeInitializer(
            tp_size=self.strategy.tp_size,
            pp_size=self.strategy.pp_size,
            cp_size=self.strategy.cp_size,
            ep_size=self.strategy.ep_size,
            params_dtype=params_dtype,
            use_cpu_initialization=False,
            attention_backend='flash',  # Use flash for training performance
            recompute_granularity=self.recompute_granularity,
            recompute_modules=self.recompute_modules,
            recompute_method=getattr(self, 'recompute_method', None),
            recompute_num_layers=getattr(self, 'recompute_num_layers', None),
        )
        
        # Create model (this calls initialize_megatron internally)
        model = self._bridge_initializer.create_model(model_path, load_weights=load_weights)
        
        # Update strategy state since bridge has initialized Megatron
        self.strategy._initialized = True
        self.strategy._parallel_state = mpu
        
        # Move to GPU
        model = self._move_model_to_gpu(model)
        
        return model
    
    def _create_megatron_model_native(
        self,
        model_path: str,
        load_weights: bool,
        params_dtype: torch.dtype,
        **kwargs,
    ) -> nn.Module:
        """Create Megatron model using twinkle's native initialization.
        
        This is the fallback method when bridge is not available.
        
        Args:
            model_path: Path to HuggingFace model.
            load_weights: Whether to load weights.
            params_dtype: Parameter dtype.
            **kwargs: Additional arguments.
            
        Returns:
            Megatron model on GPU.
        """
        from twinkle.megatron.model.initializer import MegatronModelInitializer
            
        initializer = MegatronModelInitializer(
            tp_size=self.strategy.tp_size,
            pp_size=self.strategy.pp_size,
            cp_size=self.strategy.cp_size,
            ep_size=self.strategy.ep_size,
            sequence_parallel=self.strategy.sequence_parallel,
            params_dtype=params_dtype,
        )
        
        # Create model
        model = initializer.create_gpt_model(self.hf_config, **kwargs)
        
        # Load weights
        if load_weights:
            initializer.load_from_hf(model, model_path, self.hf_config)
        
        model = self._move_model_to_gpu(model)
            
        return model
    
    def _move_model_to_gpu(self, model: nn.Module) -> nn.Module:
        """Move model to correct GPU device.
        
        This method handles moving parameters, buffers, and any cached tensors
        (like RoPE embeddings) to the correct device for distributed training.
        """
        # Determine the target device based on local rank
        local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
        device = torch.device(f'cuda:{local_rank}')
        
        # Set CUDA device explicitly
        torch.cuda.set_device(local_rank)
        
        # Move all parameters and buffers to GPU
        model = model.to(device)
        
        # Force synchronize to ensure all transfers complete
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        
        return model
        
    def _lazy_wrap_model(self):
        """Lazily wrap model with distributed wrapper.
        
        Note: This should only be called after prepare_training() has been
        executed on all workers. Direct calls from forward() may cause
        deadlocks if not all DP ranks are participating.
        """
        if not self._model_wrapped:
            # Find an optimizer from any adapter group (prefer default, then first available)
            optimizer = None
            optimizer_adapter = None
            
            if _default_adapter_name in self.optimizer_group:
                optimizer = self.optimizer_group[_default_adapter_name].optimizer
                optimizer_adapter = _default_adapter_name
            else:
                for name, group in self.optimizer_group.items():
                    if group.optimizer is not None:
                        optimizer = group.optimizer
                        optimizer_adapter = name
                        break
            
            if optimizer is not None:
                self.model, optimizer = self.strategy.wrap_model(self.model, optimizer)
                self.optimizer_group[optimizer_adapter].optimizer = optimizer
            self._model_wrapped = True
    
    @remote_function(dispatch='all')
    def prepare_training(self, **kwargs):
        """Prepare model for training.
        
        Note: In Ray-based Megatron training, we skip DDP wrapping to avoid
        deadlocks from collective operations. Each DP replica trains independently.
        This method still calls _lazy_wrap_model for any non-DDP setup needed.
        """
        self._lazy_wrap_model()

    @remote_function()
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        """Forward pass with Megatron model.
        
        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments including adapter_name.
            
        Returns:
            Model outputs.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        
        # Encode inputs if needed
        if isinstance(inputs, dict) and 'input_ids' not in inputs:
            if optimizer_config.template is not None:
                inputs = optimizer_config.template.encode(inputs)
        if isinstance(inputs, list) and 'input_ids' not in inputs[0]:
            if optimizer_config.template is not None:
                inputs = optimizer_config.template.batch_encode(inputs)
                
        # Process inputs
        processor: InputProcessor = optimizer_config.processor
        if processor is not None:
            inputs: Dict[str, Any] = processor(inputs)
            
        labels = inputs.get('labels', None)
        if 'labels' in inputs:
            try:
                del inputs['labels']
            except (TypeError, KeyError):
                pass  # Some dict-like types don't support deletion
        
        # Forward through model
        outputs = self._forward_step(inputs)
        
        inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
        return outputs
        
    def _forward_step(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute forward step with pipeline parallelism support.
        
        Args:
            inputs: Processed inputs.
            
        Returns:
            Model outputs.
        """
        # Handle pipeline parallelism
        if self.strategy.pp_size > 1:
            return self._forward_step_pipeline(inputs)
        else:
            return self._forward_step_simple(inputs)
            
    def _forward_step_simple(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Simple forward step without pipeline parallelism."""
        model = self.strategy.unwrap_model(self.model)
        
        # Prepare inputs for Megatron
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        position_ids = inputs.get('position_ids')
        
        # Create position_ids if not provided
        if position_ids is None and input_ids is not None:
            position_ids = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
            
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        
        return {'logits': outputs}
        
    def _forward_step_pipeline(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Forward step with pipeline parallelism.
        
        Note: For PP > 1, the forward pass is handled by Megatron's pipeline scheduler
        in forward_backward(). This method is for simple forward-only inference.
        For training, use forward_backward() which uses get_forward_backward_func().
        """
        from twinkle.megatron.utils import forward_step_helper
        
        model = self.strategy.unwrap_model(self.model)
        
        # Use pipeline forward helper
        output = forward_step_helper(
            model,
            inputs,
            model.config,
        )
        
        if output is not None:
            return {'logits': output}
        return {}

    @remote_function()
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        """Forward pass without gradient computation.
        
        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments.
            
        Returns:
            Model outputs.
        """
        with torch.no_grad():
            return self.forward(inputs=inputs, **kwargs)

    @remote_function(collect='avg')
    def calculate_loss(self, **kwargs):
        """Calculate loss from forward outputs.
        
        Args:
            **kwargs: Additional arguments including adapter_name.
            
        Returns:
            Loss value as numpy array.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        loss_instance: Loss = optimizer_config.loss_instance
        
        inputs = optimizer_config.inputs
        outputs = optimizer_config.outputs
        
        assert inputs is not None and outputs is not None, \
            'Cannot calculate loss of empty inputs and outputs'
            
        loss_value = loss_instance(inputs, outputs, **kwargs)
        optimizer_config.loss_value = loss_value
        return loss_value.detach().cpu().float().numpy()

    @remote_function()
    def backward(self, **kwargs):
        """Backward pass.
        
        Args:
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        loss_value = optimizer_config.loss_value
        
        assert loss_value is not None, 'Do forwarding and calculating loss before backward'
        
        _gas = optimizer_config.gradient_accumulation_steps
        if 'gradient_accumulation_steps' in kwargs:
            _gas = kwargs['gradient_accumulation_steps']
            
        loss_value = loss_value / _gas
        loss_value.backward()
        optimizer_config.cur_step += 1

    @remote_function(dispatch='all', collect='avg', sync=True)
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        """Combined forward and backward pass using Megatron's scheduler.
        
        Note: sync=True is required for Ray mode because Megatron's pipeline
        parallel uses NCCL P2P communication that requires all ranks to enter
        the function simultaneously.
        
        Always uses Megatron's get_forward_backward_func() which handles:
        - Pipeline scheduling (1F1B, interleaved, or no-pipeline)
        - Communication between stages (using proper process groups for multi-tenant isolation)
        - Gradient accumulation
        
        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments.
            
        Returns:
            Loss value.
        """
        from functools import partial
        from megatron.core.pipeline_parallel import get_forward_backward_func
        
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        
        # Encode inputs if needed
        if isinstance(inputs, dict) and 'input_ids' not in inputs:
            if optimizer_config.template is not None:
                inputs = optimizer_config.template.encode(inputs)
        if isinstance(inputs, list) and 'input_ids' not in inputs[0]:
            if optimizer_config.template is not None:
                inputs = optimizer_config.template.batch_encode(inputs)
                
        # Process inputs
        processor = optimizer_config.processor
        if processor is not None:
            inputs = processor(inputs)
        
        # Store labels before removing from inputs
        labels = inputs.get('labels', None)
        if 'labels' in inputs:
            try:
                del inputs['labels']
            except (TypeError, KeyError):
                pass  # Some dict-like types don't support deletion
        
        # Get CP size for sequence padding and splitting
        cp_size = self.strategy.cp_size
        cp_rank = mpu.get_context_parallel_rank() if cp_size > 1 else 0
        
        # Get sequence length and batch size
        # Note: Megatron's schedule internally divides seq_length by cp_size
        # So we pass the padded full sequence length here
        original_seq_length = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 1
        micro_batch_size = inputs['input_ids'].shape[0] if 'input_ids' in inputs else 1
        
        # For CP > 1, pad seq_length to be divisible by 2*cp_size
        if cp_size > 1:
            divisor = 2 * cp_size
            if original_seq_length % divisor != 0:
                seq_length = original_seq_length + (divisor - original_seq_length % divisor)
            else:
                seq_length = original_seq_length
        else:
            seq_length = original_seq_length
        
        # Move labels to GPU if needed
        if labels is not None and not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, device=torch.cuda.current_device())
        elif labels is not None:
            labels = labels.to(torch.cuda.current_device())
        
        def split_tensor_for_cp(tensor, dim=-1):
            """
            Split tensor along sequence dimension for Context Parallel.
            
            With causal masking, split into 2*CP chunks and assign alternating
            chunks to balance workload across CP ranks.
            For CP rank i: chunks [i, 2*CP-1-i]
            """
            if tensor is None or cp_size <= 1:
                return tensor
            
            if dim < 0:
                dim = (dim + tensor.ndim) % tensor.ndim
            
            seq_len = tensor.shape[dim]
            
            # Reshape to [batch, 2*cp_size, seq_per_chunk, ...]
            view_shape = list(tensor.shape)
            view_shape[dim:dim+1] = [2 * cp_size, seq_len // (2 * cp_size)]
            reshaped = tensor.view(*view_shape)
            
            # Select chunks [cp_rank, 2*cp_size-1-cp_rank]
            index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], 
                                device='cpu', pin_memory=True).cuda(non_blocking=True)
            selected = reshaped.index_select(dim, index)
            
            # Reshape back: [batch, 2*seq_per_chunk, ...]
            out_shape = list(tensor.shape)
            out_shape[dim] = seq_len // cp_size
            return selected.reshape(*out_shape)
        
        # Define forward step function for Megatron
        # forward_step_func(data_iterator, model) -> (output_tensor, partial(loss_func))
        def forward_step_func(data_iterator, model):
            batch = next(data_iterator)
            input_ids = batch.get('input_ids')
            position_ids = batch.get('position_ids')
            attention_mask = batch.get('attention_mask')
            batch_labels = batch.get('labels', labels)  # Use batch labels or passed labels
            
            # Pad sequence for Context Parallel compatibility
            # Megatron's RoPE requires seq_len % (2 * cp_size) == 0
            if cp_size > 1 and input_ids is not None:
                seq_len = input_ids.shape[1]
                divisor = 2 * cp_size
                if seq_len % divisor != 0:
                    pad_len = divisor - (seq_len % divisor)
                    # Pad input_ids
                    input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=0)
                    # Pad labels if present
                    if batch_labels is not None:
                        batch_labels = torch.nn.functional.pad(batch_labels, (0, pad_len), value=-100)
                    # Pad attention_mask if present
                    if attention_mask is not None:
                        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
                    # Pad position_ids if present
                    if position_ids is not None:
                        position_ids = torch.nn.functional.pad(position_ids, (0, pad_len), value=0)
            
            # Create position_ids if not provided
            if position_ids is None and input_ids is not None:
                position_ids = torch.arange(
                    input_ids.shape[1],
                    device=input_ids.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(input_ids.shape[0], -1)
            
            # Split tensors for Context Parallel
            # Each CP rank processes a portion of the sequence
            if cp_size > 1:
                input_ids = split_tensor_for_cp(input_ids, dim=-1)
                position_ids = split_tensor_for_cp(position_ids, dim=-1)
                attention_mask = split_tensor_for_cp(attention_mask, dim=-1)
                batch_labels = split_tensor_for_cp(batch_labels, dim=-1)
            
            # Forward pass with labels - Megatron will compute loss internally
            # This uses Megatron's compute_language_model_loss which properly handles
            # vocab parallel cross entropy
            output_tensor = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=batch_labels,  # Pass labels to let Megatron compute loss
            )
            
            # Megatron's compute_language_model_loss returns per-token loss [batch, seq]
            # We need to aggregate it with loss_mask
            def megatron_loss_func(labels_for_mask, cp_size, output_tensor):
                # output_tensor is per-token loss [batch, seq]
                # Create loss mask from labels (ignore -100)
                loss_mask = (labels_for_mask != -100).float()
                
                # Flatten and compute mean
                losses = output_tensor.float().view(-1)
                loss_mask_flat = loss_mask.view(-1)
                
                # Compute local sum and count
                local_loss_sum = torch.sum(losses * loss_mask_flat)
                local_count = loss_mask_flat.sum()
                
                # For CP > 1, aggregate loss across CP ranks
                # Note: Megatron's schedules.py will multiply loss by cp_group_size
                # for legacy 2-output loss_func. This assumes loss_func returns SUM/cp_size (MEAN).
                # So we should return local MEAN (not global MEAN) and let Megatron handle it.
                if cp_size > 1:
                    # All-reduce the count across CP ranks to get total token count
                    # This is needed for correct averaging
                    total_count = local_count.clone()
                    torch.distributed.all_reduce(
                        total_count,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mpu.get_context_parallel_group()
                    )
                    
                    # Return local_loss_sum / total_count
                    # Megatron will multiply by cp_size, so the final result is:
                    # (local_loss_sum / total_count) * cp_size
                    # = (local_loss_sum * cp_size) / total_count
                    # But we want: SUM(local_loss_sum) / total_count
                    # So we need to do all_reduce on loss_sum too
                    total_loss_sum = local_loss_sum.clone()
                    torch.distributed.all_reduce(
                        total_loss_sum,
                        op=torch.distributed.ReduceOp.SUM,
                        group=mpu.get_context_parallel_group()
                    )
                    
                    # Return global mean, but Megatron will multiply by cp_size
                    # So we divide by cp_size first to counteract that
                    loss = (total_loss_sum / total_count.clamp(min=1)) / cp_size
                else:
                    loss = local_loss_sum / local_count.clamp(min=1)
                
                return loss, {'loss': loss.detach()}
            
            return output_tensor, partial(megatron_loss_func, batch_labels, cp_size)
        
        # Get Megatron's forward-backward function
        # This automatically selects the right scheduler based on PP config:
        # - PP > 1: forward_backward_pipelining_without_interleaving (or with interleaving if VPP)
        # - PP = 1: forward_backward_no_pipelining
        forward_backward_func = get_forward_backward_func()
        
        # Create single-item iterator
        data_iter = iter([inputs])
        
        # Run forward-backward with Megatron's scheduler
        # Megatron handles all communication internally using proper process groups
        losses = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=[self.model],
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )
        
        # Extract loss from results (only last PP stage returns non-empty)
        loss = 0.0
        
        if losses:
            for loss_dict in losses:
                if isinstance(loss_dict, dict) and 'loss' in loss_dict:
                    loss = loss_dict['loss']
                    break
                elif isinstance(loss_dict, torch.Tensor):
                    loss = loss_dict
                    break
        
        # For PP > 1, broadcast loss from last PP stage to all ranks
        # Note: mpu is imported at module level, no need to reimport
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            if isinstance(loss, torch.Tensor):
                loss_tensor = loss.detach().clone()
            else:
                loss_tensor = torch.tensor(loss, dtype=torch.float32, device=torch.cuda.current_device())
            
            # Broadcast from last PP stage (rank with pipeline_model_parallel_rank == pp_size - 1)
            src_rank = mpu.get_pipeline_model_parallel_last_rank()
            pp_group = mpu.get_pipeline_model_parallel_group()
            
            torch.distributed.broadcast(
                loss_tensor, 
                src=src_rank, 
                group=pp_group
            )
            
            loss = loss_tensor.item()
        
        optimizer_config.cur_step += 1
        
        # Critical: Synchronize all DP replicas before returning
        # This ensures all DP replicas complete the same training step before
        # moving to the next batch, preventing P2P communication deadlocks
        dp_world_size = mpu.get_data_parallel_world_size()
        if dp_world_size > 1:
            # Use barrier on DP+CP group to synchronize all replicas
            dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
            dist.barrier(group=dp_cp_group)
        
        if isinstance(loss, torch.Tensor):
            return loss.detach().cpu().float().numpy()
        return float(loss)

    @remote_function(dispatch='all')
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type: int = 2, **kwargs):
        """Clip gradient norm.
        
        Args:
            max_grad_norm: Maximum gradient norm.
            norm_type: Type of norm to use.
            **kwargs: Additional arguments.
            
        Returns:
            Total norm of gradients.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        parameters = self._get_trainable_parameters(adapter_name).values()
        
        return torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm, norm_type=norm_type
        ).detach().cpu().numpy()

    @remote_function(dispatch='all')
    def step(self, **kwargs):
        """Optimizer step.
        
        For DDP-wrapped models:
        - Gradients are synchronized automatically during backward via DDP
        
        For non-DDP models (e.g., PEFT/LoRA):
        - Gradients are NOT synchronized across DP ranks
        - Each DP replica trains independently with different data
        - This is a common pattern for PEFT training where the overhead of
          gradient averaging is not worth the benefit
        
        Note: Uses dispatch='all' to ensure all workers execute this method.
        
        Args:
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return
        
        # For DDP-wrapped models, gradients are already synchronized during backward
        if self._is_model_ddp_wrapped():
            # For Megatron DDP, ensure gradient buffers are finalized
            if hasattr(self.model, 'finish_grad_sync'):
                self.model.finish_grad_sync()
        # For non-DDP models (e.g., PEFT), we skip gradient synchronization
        # Each DP replica trains independently, which is acceptable for PEFT
            
        optimizer = optimizer_config.optimizer
        assert optimizer is not None, 'Set optimizer correctly before stepping'
        
        optimizer.step(**kwargs)
    
    def _is_model_ddp_wrapped(self) -> bool:
        """Check if model is wrapped with DDP.
        
        Returns:
            True if model is wrapped with DDP (either Megatron DDP or PyTorch DDP).
        """
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        return isinstance(self.model, (MegatronDDP, TorchDDP))
    
    def _get_unwrapped_model(self) -> nn.Module:
        """Get the unwrapped model.
        
        Returns:
            The base model without DDP wrapper.
        """
        return self.strategy.unwrap_model(self.model)

    @remote_function(dispatch='all')
    def zero_grad(self, **kwargs):
        """Zero gradients.
        
        For DDP-wrapped models, also zeros the DDP gradient buffers.
        
        Args:
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return
            
        optimizer = optimizer_config.optimizer
        if optimizer is not None:
            optimizer.zero_grad(**kwargs)
        
        # For Megatron DDP, zero the gradient buffer
        if self._is_model_ddp_wrapped() and hasattr(self.model, 'zero_grad_buffer'):
            self.model.zero_grad_buffer()

    @remote_function()
    def lr_step(self, **kwargs):
        """Learning rate scheduler step.
        
        Args:
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return
            
        lr_scheduler = optimizer_config.lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(**kwargs)

    @remote_function(dispatch='all')
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        """Set loss function.
        
        NOTE: For MegatronModel, the loss is computed internally by Megatron's
        GPTModel when labels are passed. This method is kept for API compatibility
        but the provided loss_cls is NOT used during forward_backward.
        
        Megatron internally uses vocab_parallel_cross_entropy which correctly
        handles tensor parallelism. This design ensures Loss classes don't need
        to be aware of the training backend (Megatron vs Transformers).
        
        Args:
            loss_cls: Loss class or string name (not used for Megatron).
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if isinstance(loss_cls, str):
            if hasattr(twinkle.loss, loss_cls):
                loss_cls = getattr(twinkle.loss, loss_cls)
            else:
                loss_cls = Plugin.load_plugin(loss_cls, Loss)
        # Keep for API compatibility, but not used in forward_backward
        optimizer_config.loss_instance = loss_cls()

    @remote_function(dispatch='all')
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        """Set optimizer.
        
        Args:
            optimizer_cls: Optimizer class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if isinstance(optimizer_cls, str):
            if hasattr(torch.optim, optimizer_cls):
                optimizer_cls = getattr(torch.optim, optimizer_cls)
            else:
                optimizer_cls = Plugin.load_plugin(optimizer_cls, Optimizer)
                
        optimizer_config.optimizer = optimizer_cls(
            self._get_trainable_parameters(adapter_name).values(), **kwargs
        )

    def _get_trainable_parameters(self, adapter_name: str = _default_adapter_name) -> Dict[str, nn.Parameter]:
        """Get trainable parameters.
        
        Args:
            adapter_name: Name of adapter.
            
        Returns:
            Dict mapping parameter names to parameters.
        """
        is_default = adapter_name == _default_adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        
        params = {}
        model = self.strategy.unwrap_model(self.model)
        for name, param in model.named_parameters():
            if param.requires_grad and (pattern.search(name) or is_default):
                params[name] = param
        return params

    @remote_function(dispatch='all')
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str], **kwargs):
        """Set learning rate scheduler.
        
        Args:
            scheduler_cls: Scheduler class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if isinstance(scheduler_cls, str):
            if hasattr(torch.optim.lr_scheduler, scheduler_cls):
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cls)
            else:
                scheduler_cls = Plugin.load_plugin(scheduler_cls, LRScheduler)
                
        optimizer = optimizer_config.optimizer
        assert optimizer is not None, 'Set optimizer before setting lr_scheduler'
        optimizer_config.lr_scheduler = scheduler_cls(optimizer, **kwargs)

    @remote_function(dispatch='all', sync=True)
    def save(self, output_dir: str, **kwargs):
        """Save model checkpoint.
        
        Args:
            output_dir: Output directory.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        save_format = kwargs.pop('save_format', 'hf')  # 'hf' or 'megatron'
        
        if save_format == 'hf':
            self._save_hf_format(output_dir, adapter_name)
        else:
            self._save_megatron_format(output_dir, adapter_name)
            
        self._save_tokenizer(output_dir, adapter_name)
        
    def _save_hf_format(self, output_dir: str, adapter_name: str):
        """Save in HuggingFace format using bridge adapter.
        
        For distributed training:
        - All PP ranks participate in export (each has different layers)
        - Only DP rank 0 actually writes to disk
        - Uses barrier for synchronization
        
        For LoRA training:
        - Saves in PEFT format (adapter_model.safetensors + adapter_config.json)
        """
        from twinkle.megatron.model.bridge import TwinkleBridgeAdapter
        import os
        
        # Check if this is LoRA training (has adapter_name other than default)
        is_lora = adapter_name and adapter_name != ''
        is_peft_format = is_lora
        
        # Create output directory on rank 0 only
        try:
            from megatron.core import parallel_state as mpu
            dp_rank = mpu.get_data_parallel_rank() if mpu.is_initialized() else 0
        except (ImportError, AssertionError):
            dp_rank = 0
        
        if dp_rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        
        # Synchronize before saving
        if dist.is_initialized():
            dist.barrier()
        
        # Calculate padded vocab size
        padded_vocab_size = self._pad_vocab_size(self.hf_config.vocab_size) \
            if hasattr(self, '_pad_vocab_size') else None
        
        # Use TwinkleBridgeAdapter for weight conversion
        # All ranks participate - bridge handles which ranks write
        adapter = TwinkleBridgeAdapter(
            hf_config=self.hf_config,
            tp_size=self.strategy.tp_size,
            pp_size=self.strategy.pp_size,
            ep_size=self.strategy.ep_size,
            model_path=self._model_path if hasattr(self, '_model_path') else self.model_id,
            padded_vocab_size=padded_vocab_size,
        )
        
        # Get the model (unwrap if DDP wrapped)
        model = self.strategy.unwrap_model(self.model)
        
        # Use bridge to save weights
        adapter.save_weights([model], output_dir, is_peft_format=is_peft_format)
        
        # Save config on rank 0 only
        if dp_rank == 0:
            self.hf_config.save_pretrained(output_dir)
    
    def _pad_vocab_size(self, vocab_size: int) -> int:
        """Pad vocab size for tensor parallelism."""
        divisor = self.strategy.tp_size * 128
        return ((vocab_size + divisor - 1) // divisor) * divisor
        
    def _save_megatron_format(self, output_dir: str, adapter_name: str):
        """Save in Megatron checkpoint format."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        model = self.strategy.unwrap_model(self.model)
        state_dict = self._get_trainable_parameters(adapter_name)
        
        # Convert to CPU
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}
        
        # Save with rank info for distributed checkpointing
        rank = dist.get_rank() if dist.is_initialized() else 0
        checkpoint_path = os.path.join(output_dir, f'model_rank{rank}.pt')
        torch.save(cpu_state_dict, checkpoint_path)
        
    def _save_tokenizer(self, output_dir: str, adapter_name: str = _default_adapter_name):
        """Save tokenizer."""
        optimizer_config = self.optimizer_group.get(adapter_name)
        if optimizer_config and optimizer_config.template:
            optimizer_config.template.tokenizer.save_pretrained(output_dir)

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        """Get trainable state dict.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            State dict of trainable parameters.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        return self._get_trainable_parameters(adapter_name)

    _peft_patched = False
    
    @classmethod
    def _patch_peft_for_megatron(cls):
        """Patch PEFT's BaseTuner to handle Megatron's TransformerConfig.
        
        Megatron's TransformerConfig doesn't have a .get() method like HuggingFace
        configs. This patch handles the AttributeError that occurs when PEFT tries
        to check tie_word_embeddings.
        """
        if cls._peft_patched:
            return
        
        from typing import List
        import torch.nn as nn
        from peft.tuners.tuners_utils import BaseTuner
        
        _origin_get_tied_target_modules = BaseTuner._get_tied_target_modules
        
        def _get_tied_target_modules(self, model: nn.Module) -> List[str]:
            try:
                return _origin_get_tied_target_modules(self, model)
            except AttributeError:
                # Megatron's TransformerConfig doesn't have .get() method
                # Check share_embeddings_and_output_weights instead
                tied_target_modules = []
                if getattr(model, 'share_embeddings_and_output_weights', False):
                    for target_module in self.targeted_module_names:
                        module_name = target_module.split('.')[-1]
                        if module_name in ['output_layer', 'embedding', 'word_embeddings']:
                            tied_target_modules.append(target_module)
                return tied_target_modules
        
        BaseTuner._get_tied_target_modules = _get_tied_target_modules
        cls._peft_patched = True

    @remote_function(dispatch='all', sync=True)
    def add_adapter_to_model(
        self,
        adapter_name: str,
        config_or_dir: Union[Any, str],
        **kwargs,
    ):
        """Add LoRA adapter to model.
        
        Args:
            adapter_name: Name of the adapter.
            config_or_dir: LoRA config or path to saved adapter.
            **kwargs: Additional arguments.
        """
        from twinkle.megatron.utils import (
            prepare_lora_model, patch_deepcopy, get_target_modules, set_linear_is_expert
        )
        
        # Patch PEFT BaseTuner to handle Megatron's TransformerConfig
        # which doesn't have a .get() method like HuggingFace configs
        self._patch_peft_for_megatron()
        
        assert adapter_name, 'Use a non-empty adapter_name'
        
        model = self.strategy.unwrap_model(self.model)
        
        # Mark expert layers for MoE models
        set_linear_is_expert(model)
        
        if isinstance(config_or_dir, str):
            # Load from path
            config_or_dir = HubOperation.download_model(config_or_dir)
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                model, config_or_dir, adapter_name=adapter_name,
                is_trainable=kwargs.get('is_trainable', True)
            )
        else:
            # Create from config
            from peft import LoraConfig, get_peft_model
            
            if not isinstance(config_or_dir, LoraConfig):
                # Convert dict to LoraConfig
                config_or_dir = LoraConfig(**config_or_dir)
            
            # Expand target_modules (e.g., 'all-linear' -> actual module names)
            if config_or_dir.target_modules:
                if isinstance(config_or_dir.target_modules, str):
                    target_modules = [config_or_dir.target_modules]
                else:
                    target_modules = list(config_or_dir.target_modules)
                
                expanded_modules = get_target_modules(model, target_modules)
                config_or_dir.target_modules = expanded_modules
                
            with patch_deepcopy():
                model = get_peft_model(model, config_or_dir, adapter_name=adapter_name)
                
        # Update model reference
        if self._model_wrapped:
            if isinstance(self.model, MegatronDDP):
                self.model.module = model
        else:
            self.model = model
        
        # Add finish_grad_sync method for Megatron's finalize_model_grads compatibility
        # This is needed because Megatron's forward_backward_func calls finish_grad_sync
        # on model chunks, but PEFT models don't have this method by default
        if not hasattr(self.model, 'finish_grad_sync'):
            def finish_grad_sync():
                """Synchronize gradients across DP ranks for non-DDP models.
                
                This is a compatibility shim for Megatron's finalize_model_grads.
                For PEFT/LoRA models, we manually all-reduce gradients.
                """
                dp_world_size = mpu.get_data_parallel_world_size()
                if dp_world_size > 1:
                    dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
                    grads = []
                    for param in self.model.parameters():
                        if param.requires_grad and param.grad is not None:
                            grads.append(param.grad.data)
                    
                    if grads:
                        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
                        coalesced = _flatten_dense_tensors(grads)
                        dist.all_reduce(coalesced, op=dist.ReduceOp.AVG, group=dp_cp_group)
                        for grad, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                            grad.copy_(synced)
            
            self.model.finish_grad_sync = finish_grad_sync
            
        # Create optimizer group for adapter
        self.optimizer_group[adapter_name] = MegatronOptimizerGroup()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config_or_dir
        self.optimizer_group[adapter_name].gradient_accumulation_steps = kwargs.get(
            'gradient_accumulation_steps', 1
        )
        
        # Copy settings from default
        default_config = self.optimizer_group.get(_default_adapter_name)
        if default_config:
            if default_config.template:
                self.optimizer_group[adapter_name].template = default_config.template
            if default_config.processor:
                self.optimizer_group[adapter_name].processor = default_config.processor
            if default_config.loss_instance:
                self.optimizer_group[adapter_name].loss_instance = default_config.loss_instance

    @remote_function(dispatch='all')
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        """Set template for input encoding.
        
        Args:
            template_cls: Template class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if isinstance(template_cls, str):
            if hasattr(template, template_cls):
                template_cls = getattr(template, template_cls)
            else:
                template_cls = Plugin.load_plugin(template_cls, template.Template)
        optimizer_config.template = template_cls(self.model_id, **kwargs)

    @remote_function(dispatch='all')
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str], **kwargs):
        """Set input processor.
        
        Args:
            processor_cls: Processor class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if isinstance(processor_cls, str):
            if hasattr(twinkle.processor, processor_cls):
                processor_cls = getattr(twinkle.processor, processor_cls)
            else:
                processor_cls = Plugin.load_plugin(processor_cls, InputProcessor)
        optimizer_config.processor = processor_cls(device_mesh=self.device_mesh, **kwargs)

    @remote_function(execute='first')
    def get_train_configs(self, **kwargs):
        """Get training configuration summary.
        
        Args:
            **kwargs: Additional arguments.
            
        Returns:
            Configuration summary string.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        expr = f'Backend: Megatron-Core\n'
        expr += f'TP size: {self.strategy.tp_size}\n'
        expr += f'PP size: {self.strategy.pp_size}\n'
        expr += f'CP size: {self.strategy.cp_size}\n'
        expr += f'EP size: {self.strategy.ep_size}\n'
        expr += f'Sequence Parallel: {self.strategy.sequence_parallel}\n'
        
        if optimizer_config.adapter_config is not None:
            config = optimizer_config.adapter_config.__dict__
            config = {key: str(value) for key, value in config.items() if value is not None}
            expr += f'Adapter config:\n{json.dumps(config, indent=2, ensure_ascii=False)}\n'
            
        if optimizer_config.optimizer:
            expr += f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
            expr += f'Learning rate: {optimizer_config.optimizer.defaults.get("lr", "N/A")}\n'
        if optimizer_config.lr_scheduler:
            expr += f'LR scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
        expr += f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n'
        
        return expr
        
    def __repr__(self):
        return (
            f"MegatronModel(model_id='{self.model_id}', "
            f"tp={self.strategy.tp_size}, pp={self.strategy.pp_size}, "
            f"cp={self.strategy.cp_size}, ep={self.strategy.ep_size})"
        )

