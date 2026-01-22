# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from peft import PeftModel, PeftConfig
from peft import LoraConfig, get_peft_model
from transformers import PretrainedConfig
from transformers import AutoConfig
import twinkle
from twinkle import DeviceMesh, remote_class, remote_function, template, Platform
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss, MegatronCrossEntropyLoss
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle import requires
import twinkle.metric
from twinkle import torch_util
from twinkle.model.base import TwinkleModel
from .strategy import MegatronStrategy
from twinkle.metric import Metric, LossMetric, Accuracy
from twinkle.utils import construct_class


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
    _dp_group = None
    metrics: List[Metric] = field(default_factory=list)
    _device_mesh: DeviceMesh = None
    # Megatron optimizer specific fields
    is_megatron_optimizer: bool = False
    _last_grad_norm: float = 0.0
    _last_step_success: bool = True

    def do_grad_sync(self,
                     gradient_accumulation_steps: Optional[int] = None
                     ) -> bool:
        """Check if gradient synchronization should happen."""
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        return self.cur_step % gradient_accumulation_steps == 0 and self.cur_step > 0


_default_adapter_name = ''


@remote_class(execute='all')
class MegatronModel(TwinkleModel, nn.Module):

    def __init__(
        self,
        model_id: str,
        config: Optional[PretrainedConfig] = None,
        device_mesh: Optional[DeviceMesh] = None,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        load_weights: bool = True,
        recompute_granularity: Optional[str] = 'selective',  # Activation checkpointing
        recompute_modules: Optional[list] = None,  # Modules to recompute
        **kwargs,
    ):
        requires('megatron_core')
        nn.Module.__init__(self)
        from twinkle.patch.megatron_peft import MegatronPeft

        self.model_id = model_id
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self.recompute_granularity = recompute_granularity
        self.recompute_modules = recompute_modules
        model_path = HubOperation.download_model(model_id)
        if config is None:
            # Load HuggingFace config first
            self.hf_config = AutoConfig.from_pretrained(model_path)
        else:
            self.hf_config = config
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        # Store model_path for later use
        self._model_path = model_path

        self._seed = kwargs.pop('seed', None)
        if self._seed is None and os.environ.get('TWINKLE_SEED'):
            self._seed = int(os.environ.get('TWINKLE_SEED'))
        if self._seed is None:
            self._seed = 42
        self._default_tokenizer = None
        self.use_distributed_optimizer = kwargs.get('use_distributed_optimizer', True)
        self.variable_seq_lengths = kwargs.get('variable_seq_lengths', False)
        # Create Megatron strategy
        self.strategy = MegatronStrategy(self.device_mesh, mixed_precision=mixed_precision, **kwargs)

        self.model: List[nn.Module] = self._create_megatron_model(model_path, load_weights, **kwargs)

        self._model_wrapped = False
        # This correctly handles vocab sharding in Tensor Parallelism
        self.optimizer_group: Dict[str, MegatronOptimizerGroup] = self._construct_megatron_optimizer_group()
        MegatronPeft().patch()

    def _construct_default_optimizer_group(self):
        return MegatronOptimizerGroup(
            loss_instance=MegatronCrossEntropyLoss(),
            template=Template(self.tokenizer_id),
            processor=InputProcessor(self.device_mesh),
            _device_mesh=self.device_mesh,
        )

    def _create_megatron_model(
        self,
        model_path: str,
        load_weights: bool = True,
        **kwargs,
    ) -> List[nn.Module]:
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

        return self._create_megatron_model_with_bridge(
            model_path, load_weights, params_dtype, **kwargs)

    def _create_megatron_model_with_bridge(
        self,
        model_path: str,
        load_weights: bool,
        params_dtype: torch.dtype,
        **kwargs,
    ) -> List[nn.Module]:
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
        from twinkle.model.megatron.strategy.bridge import BridgeInitializer

        # Create bridge-based initializer
        self._bridge_initializer = BridgeInitializer(
            tp_size=self.device_mesh.tp_world_size,
            pp_size=self.device_mesh.pp_world_size,
            cp_size=self.device_mesh.cp_world_size,
            ep_size=self.device_mesh.ep_size,
            vpp_size=self.device_mesh.vpp_size,
            params_dtype=params_dtype,
            seed=self._seed,
            use_cpu_initialization=False,
            attention_backend='flash',  # Use flash for training performance
            sequence_parallel=self.strategy.sequence_parallel,
            recompute_granularity=self.recompute_granularity,
            recompute_modules=self.recompute_modules,
            recompute_method=getattr(self, 'recompute_method', None),
            recompute_num_layers=getattr(self, 'recompute_num_layers', None),
        )

        # Create model (this calls initialize_megatron internally)
        model = self._bridge_initializer.create_model(
            model_path, load_weights=load_weights)

        self._transformer_config = self._bridge_initializer.config
        _models = []
        for _model in model:
            _model = self._move_model_to_gpu(_model)
            _models.append(_model)
        return _models

    @staticmethod
    def _move_model_to_gpu(model: nn.Module) -> nn.Module:
        model_device = next(model.parameters()).device
        torch.cuda.set_device(Platform.get_local_rank())
        if model_device.type == 'cpu':
            model = model.to(Platform.get_local_device())
        torch_util.synchronize()
        return model

    def _lazy_wrap_model(self):
        if not self._model_wrapped:
            assert len(self.optimizer_group) == 1
            optimizer = self.optimizer_group[_default_adapter_name].optimizer
            if optimizer is None:
                optimizer = self._create_megatron_optimizer()
                self.optimizer_group[_default_adapter_name].optimizer = optimizer
            self.model, optimizer = self.strategy.wrap_model(self.model, optimizer)
            self.optimizer_group[_default_adapter_name].optimizer = optimizer
            self._model_wrapped = True

    @staticmethod
    def _not_encoded(inputs):
        assert isinstance(inputs, dict)
        return 'input_ids' not in inputs and 'input_embedding' not in inputs

    @staticmethod
    def _accumulate_metric(optimizer_config: MegatronOptimizerGroup):
        if len(optimizer_config.metrics) > 0 and optimizer_config.inputs is not None and optimizer_config.outputs is not None:
            for metric in optimizer_config.metrics:
                metric.accumulate(optimizer_config.inputs, optimizer_config.outputs)

    @remote_function()
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature],
                                       Trajectory, List[Trajectory]],
                **kwargs):
        raise NotImplementedError(f'Megatron only supports `forward_backward` and `forward_only`')

    @remote_function(collect='last_pp')
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature],
                                            List[Trajectory]],
                     micro_batch_size: Optional[int] = None,
                     **kwargs):
        """Forward pass without gradient computation.

        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments.

        Returns:
            Model outputs.
        """
        self._lazy_wrap_model()
        from functools import partial
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core import parallel_state as mpu

        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]

        vpp_size = self.device_mesh.vpp_size
        if vpp_size is None or vpp_size == 1:
            micro_batch_size = None
        else:
            micro_batch_size = 1

        if isinstance(inputs, dict) and self._not_encoded(inputs):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = optimizer_config.template.encode(inputs)  # noqa
        if isinstance(inputs, list) and self._not_encoded(inputs[0]):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        processor: InputProcessor = optimizer_config.processor
        assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
        inputs = processor(inputs, micro_batch_size=micro_batch_size, variable_seq_lengths=self.variable_seq_lengths)
        self._accumulate_metric(optimizer_config)

        # Get parallelism settings for sequence padding and splitting
        cp_size = self.device_mesh.cp_world_size
        # Check actual sequence_parallel setting from model config
        # Bridge may auto-enable sequence_parallel for MoE models
        if self.variable_seq_lengths:
            seq_length = None
        else:
            _example = inputs[0] if isinstance(inputs, list) else inputs
            original_seq_length = _example['input_ids'].shape[1] if 'input_ids' in _example else _example[
                'input_embedding']
            if cp_size > 1:
                divisor = 2 * cp_size
            elif self.sequence_parallel and self.device_mesh.tp_world_size > 1:
                divisor = self.device_mesh.tp_world_size
            else:
                divisor = 1

            if divisor > 1 and original_seq_length % divisor != 0:
                seq_length = original_seq_length + (divisor - original_seq_length % divisor)
            else:
                seq_length = original_seq_length

        # Define forward step function for Megatron
        # forward_step_func(data_iterator, model) -> (output_tensor, partial(loss_func))
        def forward_step_func(data_iterator, model):
            batch = next(data_iterator)
            batch = self.strategy.split_inputs_for_cp(batch)
            input_ids = batch.get('input_ids')
            position_ids = batch.get('position_ids')
            attention_mask = batch.get('attention_mask')
            batch_labels = batch.get('labels')

            # Forward pass with labels - Megatron will compute loss internally
            # This uses Megatron's compute_language_model_loss which properly handles
            # vocab parallel cross entropy
            output_tensor = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=batch_labels,  # Pass labels to let Megatron compute loss
            )
            return output_tensor, partial(optimizer_config.loss_instance.__call__, batch)

        # Get Megatron's forward-backward function
        # This automatically selects the right scheduler based on PP config:
        # - PP > 1: forward_backward_pipelining_without_interleaving (or with interleaving if VPP)
        # - PP = 1: forward_backward_no_pipelining
        forward_backward_func = get_forward_backward_func()
        vpp_size = self.device_mesh.vpp_size

        if vpp_size is None or vpp_size == 1:
            data_iter = [iter(inputs)]
        else:
            data_iter = [iter(inputs) for _ in range(0, vpp_size)]

        # Run forward-backward with Megatron's scheduler
        # Megatron handles all communication internally using proper process groups
        losses = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=1,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )

        # Extract loss from results (only last PP stage returns non-empty)
        logits = None

        if losses:
            for loss_dict in losses:
                if isinstance(loss_dict, torch.Tensor):
                    logits = loss_dict
                    break

        # Critical: Synchronize all DP replicas before returning
        # This ensures all DP replicas complete the same training step before
        # moving to the next batch, preventing P2P communication deadlocks
        dp_world_size = mpu.get_data_parallel_world_size()
        if dp_world_size > 1:
            # Use barrier on DP+CP group to synchronize all replicas
            dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
            dist.barrier(group=dp_cp_group)

        return logits

    @remote_function(collect='mean')
    def calculate_loss(self, **kwargs):
        raise NotImplementedError(f'Megatron only supports `forward_backward` and `forward_only`')

    @remote_function()
    def backward(self, **kwargs):
        raise NotImplementedError(f'Megatron only supports `forward_backward` and `forward_only`')

    @remote_function(dispatch='all', collect='mean', sync=True)
    def forward_backward(self,
                         *,
                         inputs: Union[InputFeature, List[InputFeature],
                                       Trajectory, List[Trajectory]],
                         micro_batch_size: Optional[int] = None,
                         **kwargs):
        """Combined forward and backward pass using Megatron's scheduler.

        Note: sync=True is required for Ray mode because Megatron's pipeline
        parallel uses NCCL P2P communication that requires all ranks to enter
        the function simultaneously.

        Always uses Megatron's get_forward_backward_func() which handles:
        - Pipeline scheduling (1F1B, interleaved, or no-pipeline)
        - Communication between stages (using proper process groups for multi-tenant isolation)
        - Gradient accumulation across microbatches

        Args:
            inputs: Model inputs. Can be:
                - A single batch dict (num_microbatches=1)
                - A list of batch dicts (num_microbatches=len(inputs))
                - An iterator yielding batch dicts
            **kwargs: Additional arguments.

        Returns:
            Average loss value across all microbatches.
        """
        self._lazy_wrap_model()
        from functools import partial
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core import parallel_state as mpu

        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]

        vpp_size = self.device_mesh.vpp_size
        if vpp_size is None or vpp_size == 1:
            micro_batch_size = None
        else:
            micro_batch_size = 1

        if isinstance(inputs, dict) and self._not_encoded(inputs):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = optimizer_config.template.encode(inputs) # noqa
        if isinstance(inputs, list) and self._not_encoded(inputs[0]):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = optimizer_config.template.batch_encode(inputs) # noqa
        processor: InputProcessor = optimizer_config.processor
        assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
        inputs = processor(inputs, micro_batch_size=micro_batch_size, variable_seq_lengths=self.variable_seq_lengths)
        self._accumulate_metric(optimizer_config)

        # Get parallelism settings for sequence padding and splitting
        cp_size = self.device_mesh.cp_world_size
        # Check actual sequence_parallel setting from model config
        # Bridge may auto-enable sequence_parallel for MoE models
        if self.variable_seq_lengths:
            seq_length = None
        else:
            _example = inputs[0] if isinstance(inputs, list) else inputs
            original_seq_length = _example['input_ids'].shape[1] if 'input_ids' in _example else _example['input_embedding']
            if cp_size > 1:
                divisor = 2 * cp_size
            elif self.sequence_parallel and self.device_mesh.tp_world_size > 1:
                divisor = self.device_mesh.tp_world_size
            else:
                divisor = 1

            if divisor > 1 and original_seq_length % divisor != 0:
                seq_length = original_seq_length + (divisor - original_seq_length % divisor)
            else:
                seq_length = original_seq_length

        # Define forward step function for Megatron
        # forward_step_func(data_iterator, model) -> (output_tensor, partial(loss_func))
        def forward_step_func(data_iterator, model):
            batch = next(data_iterator)
            batch = self.strategy.split_inputs_for_cp(batch)
            input_ids = batch.get('input_ids')
            position_ids = batch.get('position_ids')
            attention_mask = batch.get('attention_mask')
            batch_labels = batch.get('labels')

            # Forward pass with labels - Megatron will compute loss internally
            # This uses Megatron's compute_language_model_loss which properly handles
            # vocab parallel cross entropy
            output_tensor = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=batch_labels,  # Pass labels to let Megatron compute loss
            )
            return output_tensor, partial(optimizer_config.loss_instance.__call__, batch)

        # Get Megatron's forward-backward function
        # This automatically selects the right scheduler based on PP config:
        # - PP > 1: forward_backward_pipelining_without_interleaving (or with interleaving if VPP)
        # - PP = 1: forward_backward_no_pipelining
        forward_backward_func = get_forward_backward_func()
        vpp_size = self.device_mesh.vpp_size

        if vpp_size is None or vpp_size == 1:
            data_iter = [iter(inputs)]
        else:
            data_iter = [iter(inputs) for _ in range(0, vpp_size)]

        # Run forward-backward with Megatron's scheduler
        # Megatron handles all communication internally using proper process groups
        losses = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=self.model,
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
            loss_tensor = loss.detach().clone()
            # Broadcast from last PP stage (rank with pipeline_model_parallel_rank == pp_size - 1)
            src_rank = mpu.get_pipeline_model_parallel_last_rank()
            pp_group = mpu.get_pipeline_model_parallel_group()

            torch.distributed.broadcast(loss_tensor,
                                        src=src_rank,
                                        group=pp_group)

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
    def clip_grad_norm(self,
                       max_grad_norm: float = 1.0,
                       norm_type: int = 2,
                       **kwargs):
        """Clip gradient norm.

        Args:
            max_grad_norm: Maximum gradient norm.
            norm_type: Type of norm to use.
            **kwargs: Additional arguments.

        Returns:
            Total norm of gradients.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]

        if optimizer_config.is_megatron_optimizer:
            # Megatron optimizer handles gradient clipping in step()
            # Return the grad_norm from last step if available
            return getattr(optimizer_config, '_last_grad_norm', 0.0)

        parameters = self._get_trainable_parameters(adapter_name).values()

        return torch.nn.utils.clip_grad_norm_(
            parameters, max_grad_norm,
            norm_type=norm_type).detach().cpu().numpy()

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

        if not optimizer_config.do_grad_sync(
                kwargs.pop('gradient_accumulation_steps', None)):
            return

        optimizer = optimizer_config.optimizer
        assert optimizer is not None, 'Set optimizer correctly before stepping'

        if optimizer_config.is_megatron_optimizer:
            # Megatron optimizer step() returns (success, grad_norm, num_zeros)
            success, grad_norm, num_zeros = optimizer.step(**kwargs)
            # Store grad_norm for later retrieval
            optimizer_config._last_grad_norm = grad_norm if grad_norm is not None else 0.0
            optimizer_config._last_step_success = success
        else:
            optimizer.step(**kwargs)

    def _is_model_ddp_wrapped(self) -> bool:
        """Check if model is wrapped with DDP.

        Returns:
            True if model is wrapped with DDP (either Megatron DDP, LoRA DDP, or PyTorch DDP).
        """
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        return isinstance(self.model, (MegatronDDP, TorchDDP))

    @remote_function(dispatch='all')
    def zero_grad(self, **kwargs):
        """Zero gradients.

        For DDP-wrapped models, also zeros the DDP gradient buffers.

        Note: For DDP-wrapped models, zero_grad_buffer() is always called
        because it's essential for the next training iteration. The
        do_grad_sync check only affects the optimizer.zero_grad() call.

        Args:
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]

        # For DDP-wrapped models, ALWAYS zero the gradient buffer
        # This is essential because Megatron's forward_backward_func uses
        # the buffer's state to track gradient accumulation
        if self._is_model_ddp_wrapped() and hasattr(self.model,
                                                    'zero_grad_buffer'):
            self.model.zero_grad_buffer()

        if not optimizer_config.do_grad_sync(
                kwargs.pop('gradient_accumulation_steps', None)):
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

        if not optimizer_config.do_grad_sync(
                kwargs.pop('gradient_accumulation_steps', None)):
            return

        lr_scheduler = optimizer_config.lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(**kwargs)

    @remote_function(dispatch='all')
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str], **kwargs):
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
        optimizer_config.loss_instance = construct_class(loss_cls, Loss, twinkle.loss, **kwargs)

    def add_metric(self, metric_cls: Union[Metric, str], **kwargs):
        """Add an eval metric

        Args:
            metric_cls: A metric class type or id.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the metric_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['device_mesh'] = self.device_mesh
        kwargs['process_group'] = optimizer_config._dp_group
        metric = construct_class(metric_cls, Metric, twinkle.metric, **kwargs)
        optimizer_config.metrics.append(metric)

    @remote_function(dispatch='all')
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str],
                      **kwargs):
        """Set optimizer.

        Args:
            optimizer_cls: Optimizer class or string name.
                - Standard PyTorch optimizers: 'AdamW', 'Adam', 'SGD', etc.
                - 'MegatronDistributed': Use Megatron's distributed optimizer
            **kwargs: Additional arguments.
                - For standard optimizers: lr, weight_decay, etc.
                - For MegatronDistributed: use_distributed_optimizer, clip_grad, etc.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        use_megatron_optimizer = kwargs.pop('use_megatron_optimizer', False)

        # Check if requesting Megatron distributed optimizer
        if optimizer_cls == 'MegatronDistributed' or use_megatron_optimizer:
            optimizer_config.optimizer = self._create_megatron_optimizer(**kwargs)
            optimizer_config.is_megatron_optimizer = True
        else:
            optimizer_config.optimizer = construct_class(optimizer_cls, Optimizer, torch.optim, **kwargs)
            optimizer_config.is_megatron_optimizer = False

    def _create_megatron_optimizer(self, **kwargs):
        """Create Megatron distributed optimizer.

        This provides significant memory savings for large models by sharding
        optimizer states across DP replicas.

        Args:
            **kwargs: Optimizer configuration options.
                - lr: Learning rate (default: 1e-4)
                - weight_decay: Weight decay (default: 0.0)
                - use_distributed_optimizer: Shard optimizer states (default: True)
                - clip_grad: Gradient clipping threshold (default: 1.0)
                - bf16: Use bf16 training (default: True)
                - adam_beta1, adam_beta2, adam_eps: Adam parameters

        Returns:
            MegatronOptimizer instance.
        """
        from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig
        from megatron.core import parallel_state as mpu

        # Build optimizer config
        lr = kwargs.get('lr', 1e-4)
        use_distributed_optimizer: bool = kwargs.get('use_distributed_optimizer', self.use_distributed_optimizer)

        opt_config = OptimizerConfig(
            optimizer='adam',
            lr=lr,
            min_lr=kwargs.get('min_lr', 0.0),
            weight_decay=kwargs.get('weight_decay', 0.0),
            adam_beta1=kwargs.get('adam_beta1', 0.9),
            adam_beta2=kwargs.get('adam_beta2', 0.999),
            adam_eps=kwargs.get('adam_eps', 1e-8),
            clip_grad=kwargs.get('clip_grad', 1.0),
            bf16=kwargs.get('bf16', True),
            use_distributed_optimizer=use_distributed_optimizer,
            overlap_param_gather=kwargs.get('overlap_param_gather', False),
            log_num_zeros_in_grad=kwargs.get('log_num_zeros_in_grad', False),
        )

        # For PEFT models, we need to handle the case where model is not DDP-wrapped
        # We create a temporary wrapper to satisfy Megatron's optimizer requirements
        model_chunks = [self.model]

        # Check if model has ddp_config (required for distributed optimizer)
        if not hasattr(self.model, 'ddp_config') and use_distributed_optimizer:
            # For PEFT models without DDP, fall back to non-distributed optimizer
            # but still use Megatron's optimized implementation
            opt_config.use_distributed_optimizer = False

        optimizer = get_megatron_optimizer(
            config=opt_config,
            model_chunks=model_chunks,
        )
        return optimizer

    def _get_trainable_parameters(
            self,
            adapter_name: str = _default_adapter_name
    ) -> Dict[str, nn.Parameter]:
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
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str],
                         **kwargs):
        """Set learning rate scheduler.

        Args:
            scheduler_cls: Scheduler class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.lr_scheduler = construct_class(scheduler_cls, LRScheduler, torch.optim.lr_scheduler, **kwargs)

    @remote_function(dispatch='all')
    def clip_grad_and_step(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        self.clip_grad_norm(max_grad_norm, norm_type, **kwargs)
        self.step(**kwargs)
        self.zero_grad(**kwargs)
        self.lr_step(**kwargs)

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

        self._save_tokenizer(output_dir, adapter_name=adapter_name)

    def _save_hf_format(self, output_dir: str, adapter_name: str):
        """Save in HuggingFace format using bridge adapter.

        For distributed training:
        - All PP ranks participate in export (each has different layers)
        - Only DP rank 0 actually writes to disk
        - Uses barrier for synchronization

        For LoRA training:
        - Saves in PEFT format (adapter_model.safetensors + adapter_config.json)
        """
        from .strategy import BridgeAdapter
        # Check if this is LoRA training (has adapter_name other than default)
        is_lora = adapter_name and adapter_name != ''
        is_peft_format = is_lora

        # Create output directory on rank 0 only
        from megatron.core import parallel_state as mpu
        dp_rank = mpu.get_data_parallel_rank() if mpu.is_initialized() else 0

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
        adapter = BridgeAdapter(
            hf_config=self.hf_config,
            tp_size=self.device_mesh.tp_world_size,
            pp_size=self.device_mesh.pp_world_size,
            ep_size=self.device_mesh.ep_size,
            model_path=self._model_path
            if hasattr(self, '_model_path') else self.model_id,
            padded_vocab_size=padded_vocab_size,
        )

        # Get the model (unwrap if DDP wrapped)
        model = self.strategy.unwrap_model(self.model)

        # Use bridge to save weights
        adapter.save_weights(model,
                             output_dir,
                             is_peft_format=is_peft_format)

        # Save config on rank 0 only
        if dp_rank == 0:
            self.hf_config.save_pretrained(output_dir)

    def _pad_vocab_size(self, vocab_size: int) -> int:
        """Pad vocab size for tensor parallelism."""
        divisor = self.device_mesh.tp_world_size * 128
        return ((vocab_size + divisor - 1) // divisor) * divisor

    def _save_megatron_format(self, output_dir: str, adapter_name: str):
        """Save in Megatron checkpoint format."""
        os.makedirs(output_dir, exist_ok=True)

        state_dict = self._get_trainable_parameters(adapter_name)

        # Convert to CPU
        cpu_state_dict = {k: v.cpu() for k, v in state_dict.items()}

        # Save with rank info for distributed checkpointing
        rank = dist.get_rank() if dist.is_initialized() else 0
        checkpoint_path = os.path.join(output_dir, f'model_rank{rank}.pt')
        torch.save(cpu_state_dict, checkpoint_path)

    def _save_tokenizer(self,
                        output_dir: str,
                        **kwargs):
        """Save tokenizer."""
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        template_ins = optimizer_config.template
        if template_ins is not None:
            template_ins.tokenizer.save_pretrained(output_dir)
        else:
            self._default_tokenizer.save_pretrained(output_dir)

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

    def _patch_adapter(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], train_group: str, **kwargs):
        from .tuners.multi_lora import set_linear_is_expert, get_target_modules, patch_deepcopy
        assert adapter_name, 'Use a non-empty adapter_name'
        model = self.strategy.unwrap_model(self.model)
        if isinstance(config_or_dir, str):
            config_or_dir = HubOperation.download_model(config_or_dir)

        _models = []
        for _model in model:
            # Mark expert layers for MoE models
            set_linear_is_expert(_model)
            if isinstance(config_or_dir, str):
                _model = PeftModel.from_pretrained(_model,
                                                  config_or_dir,
                                                  adapter_name=adapter_name,
                                                  is_trainable=kwargs.get(
                                                      'is_trainable', True))
                config = _model.peft_config
            else:
                if not isinstance(config_or_dir, LoraConfig):
                    config_or_dir = LoraConfig(**config_or_dir)
                config = config_or_dir

                # Expand target_modules (e.g., 'all-linear' -> actual module names)
                if config.target_modules:
                    if isinstance(config.target_modules, str):
                        target_modules = [config.target_modules]
                    else:
                        target_modules = list(config.target_modules)

                    expanded_modules = get_target_modules(_model, target_modules)
                    config.target_modules = expanded_modules

                with patch_deepcopy():
                    _model = get_peft_model(_model,
                                           config,
                                           adapter_name=adapter_name)
            _models.append(_model)
        self.model = _models

        # Create optimizer group for adapter
        self.optimizer_group[train_group] = self._construct_default_optimizer_group()
        self.optimizer_group[train_group].adapter_name = adapter_name
        self.optimizer_group[train_group].adapter_config = config
        self.optimizer_group[
            train_group].gradient_accumulation_steps = kwargs.get(
            'gradient_accumulation_steps', 1)

        default_config = self.optimizer_group[_default_adapter_name]
        if default_config.template:
            self.optimizer_group[train_group].template = default_config.template
        if default_config.processor:
            self.optimizer_group[train_group].processor = default_config.processor
        if default_config.loss_instance:
            self.optimizer_group[train_group].loss_instance = default_config.loss_instance
        self._default_tokenizer = self.optimizer_group[train_group].template.tokenizer
        dp_group = self.optimizer_group[train_group]._dp_group
        self.optimizer_group[train_group].metrics = [
                LossMetric(self.device_mesh, dp_group),
                Accuracy(self.device_mesh, dp_group),
            ]

    @remote_function(dispatch='all', sync=True)
    def add_adapter_to_model(
        self,
        adapter_name: str,
        config_or_dir: Union[Dict[str, Any], LoraConfig, str],
        **kwargs,
    ):
        """Add LoRA adapter to model.

        Args:
            adapter_name: Name of the adapter.
            config_or_dir: LoRA config or path to saved adapter.
            **kwargs: Additional arguments.
        """
        self._patch_adapter(adapter_name, config_or_dir, _default_adapter_name, **kwargs)


    @remote_function(dispatch='all')
    def set_template(self, template_cls: Union[Type[template.Template], str],
                     **kwargs):
        """Set template for input encoding.

        Args:
            template_cls: Template class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.template = construct_class(template_cls, Template, twinkle.template, **kwargs)

    @remote_function(dispatch='all')
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str],
                      **kwargs):
        """Set input processor.

        Args:
            processor_cls: Processor class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)

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
        expr += f'DP size: {self.device_mesh.dp_world_size}\n'
        expr += f'TP size: {self.device_mesh.tp_world_size}\n'
        expr += f'  - VPP size: {self.device_mesh.vpp_size}\n'
        expr += f'PP size: {self.device_mesh.pp_world_size}\n'
        expr += f'CP size: {self.device_mesh.cp_world_size}\n'
        expr += f'EP size: {self.device_mesh.ep_size}\n'
        expr += f'Sequence Parallel: {self.strategy.sequence_parallel}\n'

        if optimizer_config.adapter_config is not None:
            config = optimizer_config.adapter_config.__dict__
            config = {
                key: str(value)
                for key, value in config.items() if value is not None
            }
            expr += f'Adapter config:\n{json.dumps(config, indent=2, ensure_ascii=False)}\n'

        if optimizer_config.optimizer:
            expr += f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
            expr += f'Learning rate: {optimizer_config.optimizer.defaults.get("lr", "N/A")}\n'
        if optimizer_config.lr_scheduler:
            expr += f'LR scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
        expr += f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n'

        return expr
