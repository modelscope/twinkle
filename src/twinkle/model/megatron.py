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
from twinkle.loss import Loss, VocabParallelCrossEntropyLoss
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
        **kwargs,
    ):
        check_megatron_available()
        nn.Module.__init__(self)
        
        self.model_id = pretrained_model_name_or_path
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        
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
        
        # Initialize parallel state
        self.strategy.initialize()
        
        # Load HuggingFace config
        model_path = HubOperation.download_model(pretrained_model_name_or_path)
        self._load_hf_config(model_path)
        
        # Create Megatron model
        self.model = self._create_megatron_model(model_path, load_weights, **kwargs)
        
        self._model_wrapped = False
        # Use VocabParallelCrossEntropyLoss by default for Megatron
        # This correctly handles vocab sharding in Tensor Parallelism
        self.optimizer_group: Dict[str, MegatronOptimizerGroup] = {
            _default_adapter_name: MegatronOptimizerGroup(loss_instance=VocabParallelCrossEntropyLoss())
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
        from twinkle.megatron.model.initializer import MegatronModelInitializer
        
        params_dtype = torch.bfloat16
        if self.mixed_precision == 'fp16':
            params_dtype = torch.float16
        elif self.mixed_precision == 'no':
            params_dtype = torch.float32
            
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
        # Determine the target device based on local rank
        local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
        device = torch.device(f'cuda:{local_rank}')
        
        # Move all parameters and buffers to GPU
        model = model.to(device)
        
        return model
        
    def _lazy_wrap_model(self):
        """Lazily wrap model with distributed wrapper."""
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
            
        labels = inputs.pop('labels', None)
        
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
        """Forward step with pipeline parallelism."""
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

    @remote_function(collect='avg')
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        """Combined forward and backward pass.
        
        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments.
            
        Returns:
            Loss value.
        """
        self.forward(inputs=inputs, **kwargs)
        loss = self.calculate_loss(**kwargs)
        self.backward(**kwargs)
        return loss

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
        
        For PEFT models, gradients are NOT synchronized across DP ranks
        because each DP replica trains independently with different data.
        This is a common pattern for PEFT training where gradient averaging
        is not strictly necessary.
        
        Note: Uses dispatch='all' to ensure all workers execute this method,
        though gradient sync is disabled for PEFT models.
        
        Args:
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return
        
        # Note: For PEFT/LoRA models, we skip gradient synchronization across DP ranks.
        # Each DP replica trains independently. This avoids distributed communication
        # complexity and is acceptable for most PEFT training scenarios.
        # If gradient averaging is needed, use DDP-wrapped models instead.
            
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

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        """Set loss function.
        
        Args:
            loss_cls: Loss class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if isinstance(loss_cls, str):
            if hasattr(twinkle.loss, loss_cls):
                loss_cls = getattr(twinkle.loss, loss_cls)
            else:
                loss_cls = Plugin.load_plugin(loss_cls, Loss)
        optimizer_config.loss_instance = loss_cls()

    @remote_function()
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

    @remote_function()
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

    @remote_function()
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
        """Save in HuggingFace format using swift's GPTBridge."""
        from twinkle.megatron.model.bridge import TwinkleBridgeAdapter
        import os
        
        # Only save from last PP rank
        if not self.strategy.is_pipeline_last_stage():
            return
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Use TwinkleBridgeAdapter which wraps swift's GPTBridge
        adapter = TwinkleBridgeAdapter(
            hf_config=self.hf_config,
            tp_size=self.strategy.tp_size,
            pp_size=self.strategy.pp_size,
            ep_size=self.strategy.ep_size,
            model_path=self.pretrained_model_name_or_path,
        )
        
        # Use swift's bridge to save weights
        adapter.save_weights([self.model], output_dir, is_peft_format=False)
            
        # Save config
        self.hf_config.save_pretrained(output_dir)
        
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
        
        Reference: swift/swift/megatron/init.py::_patch_peft_BaseTuner
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

    @remote_function()
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

    @remote_function()
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

    @remote_function()
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

