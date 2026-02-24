# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import inspect
import json
import logging
import numpy as np
import os
import random
import re
import threading
import torch
import torch.distributed as dist
import torch.nn as nn
from dataclasses import dataclass, field
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from peft.tuners.lora import Linear as LoraLinear
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoConfig, PretrainedConfig
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Tuple, Type, Union

import twinkle
import twinkle.metric
import twinkle.patch
from twinkle import DeviceMesh, Platform, remote_class, remote_function, requires, torch_util
from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin
from twinkle.data_format import InputFeature, ModelOutput, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss, VocabParallelCrossEntropyLoss
from twinkle.metric import LossMetric, Metric, TrainMetric
from twinkle.model.base import TwinkleModel
from twinkle.patch import Patch, apply_patch
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils import construct_class, exists
from .strategy import MegatronStrategy


@dataclass
class MegatronOptimizerGroup:
    """Optimizer group for Megatron training.

    Similar to OptimizerGroup but adapted for Megatron's distributed training.
    """
    adapter_name: str = None
    adapter_config: Any = None
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None
    inputs: List[InputFeature] = None
    outputs: ModelOutput = None
    loss_instance: Loss = None
    loss_value: Any = None
    template: Template = None
    processor: InputProcessor = None
    gradient_accumulation_steps: int = 1
    cur_step: int = 0
    _dp_group = None
    train_metrics: List[Metric] = field(default_factory=list)
    eval_metrics: List[Metric] = field(default_factory=list)
    _device_mesh: DeviceMesh = None
    # Megatron optimizer specific fields
    _last_grad_norm: float = 0.0
    _last_step_success: bool = True

    def do_grad_sync(self, gradient_accumulation_steps: Optional[int] = None) -> bool:
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        else:
            self.gradient_accumulation_steps = gradient_accumulation_steps
        return (self.cur_step - 1) % gradient_accumulation_steps == 0 and self.cur_step > 1

    def __post_init__(self):
        if self._device_mesh.data_world_size > 1:
            self._dp_group = self._device_mesh.create_process_group(['dp', 'fsdp'])
        self.train_metrics = [
            LossMetric(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]

        self.eval_metrics = [
            LossMetric(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]

    def _get_lr(self):
        _lrs = []
        _default_lr = self.optimizer.chained_optimizers[0].config.lr
        for param_group in self.optimizer.param_groups:
            _lrs.append(param_group.get('lr', _default_lr))
        return _lrs

    def accumulate_metrics(self, is_training):
        if is_training:
            metrics = self.train_metrics
        else:
            metrics = self.eval_metrics
        if len(metrics) > 0 and self.inputs is not None and self.outputs is not None:
            for metric in metrics:
                metric.accumulate(
                    self.inputs,
                    self.outputs,
                    lr=self._get_lr(),
                    step=self.cur_step - 1,
                    gradient_accumulation_steps=self.gradient_accumulation_steps,
                    grad_norm=self._last_grad_norm)

    def calculate_metrics(self, is_training):
        self.accumulate_metrics(is_training)
        if is_training:
            metrics = self.train_metrics
        else:
            metrics = self.eval_metrics
        results = {}
        for metric in metrics:
            results.update(metric.calculate())
        return results


_default_adapter_name = ''

_BASE_LAYER_SUFFIXES = [
    '.q_proj.weight',
    '.q_proj.bias',
    '.k_proj.weight',
    '.k_proj.bias',
    '.v_proj.weight',
    '.v_proj.bias',
    '.o_proj.weight',
    '.o_proj.bias',
    '.gate_proj.weight',
    '.up_proj.weight',
    '.down_proj.weight',
    '.mlp.gate.weight',
    '.mlp.gate.bias',
    '.mlp.gate.e_score_correction_bias',
]


def _add_base_layer_suffix(params):
    """Insert ``.base_layer.`` before the final attribute for LoRA-target modules.

    Converts plain HF names exported by the Megatron bridge into the format
    expected by vLLM when ``enable_lora=True``::

        model.layers.0.self_attn.q_proj.weight
        -> model.layers.0.self_attn.q_proj.base_layer.weight

    Non-matching names are yielded unchanged.

    Args:
        params: Iterable of ``(name, tensor)`` pairs.

    Yields:
        ``(name, tensor)`` with ``.base_layer.`` inserted where needed.
    """
    for name, param in params:
        for suffix in _BASE_LAYER_SUFFIXES:
            if name.endswith(suffix):
                attr = suffix.rsplit('.', 1)[-1]  # 'weight' or 'bias'
                name = f'{name[:-len(attr)]}base_layer.{attr}'
                break
        yield name, param


@remote_class(execute='all')
class MegatronModel(TwinkleModel, nn.Module, CheckpointEngineMixin):

    def __init__(
        self,
        model_id: str,
        config: Optional[PretrainedConfig] = None,
        device_mesh: Optional[DeviceMesh] = None,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        load_weights: bool = True,
        recompute_granularity: Optional[str] = 'full',  # Activation checkpointing
        recompute_method: Optional[str] = 'uniform',
        recompute_num_layers: Optional[int] = 1,
        recompute_modules: Optional[list] = None,  # Modules to recompute
        **kwargs,
    ):
        requires('megatron_core')
        from .args import TwinkleMegatronArgs, get_args, set_args
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        nn.Module.__init__(self)
        from twinkle.patch.megatron_peft import MegatronPeft

        self.model_id = model_id
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision

        self._model_path = HubOperation.download_model(model_id)
        self.hf_config = config or AutoConfig.from_pretrained(self._model_path)
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)

        self._seed = kwargs.pop('seed', None) or int(os.environ.get('TWINKLE_SEED', 42))
        self._default_tokenizer = None
        self.use_distributed_optimizer = kwargs.get('use_distributed_optimizer', True)
        self.variable_seq_lengths = kwargs.get('variable_seq_lengths', True)
        torch_util.set_device()

        self.strategy = MegatronStrategy(self.device_mesh, mixed_precision=mixed_precision, **kwargs)

        # Determine params_dtype and activation checkpointing kwargs
        params_dtype = torch.bfloat16
        if self.mixed_precision == 'fp16':
            params_dtype = torch.float16
        elif self.mixed_precision == 'no':
            params_dtype = torch.float32

        ac_kwargs = {
            'recompute_granularity': recompute_granularity,
            'recompute_modules': recompute_modules,
            'recompute_method': recompute_method,
            'recompute_num_layers': recompute_num_layers,
        }

        # Initialize TwinkleMegatronArgs BEFORE creating the model
        args = TwinkleMegatronArgs.from_hf_config(
            self.hf_config,
            model_dir=self._model_path,
            device_mesh=self.device_mesh,
            params_dtype=params_dtype,
            sequence_parallel=self.strategy.sequence_parallel,
            **ac_kwargs,
        )
        set_args(args)
        self._initialized = False
        self.model: List[nn.Module] = self._create_megatron_model(load_weights, **kwargs)

        self._model_wrapped = False
        # This correctly handles vocab sharding in Tensor Parallelism
        self.optimizer_group: Dict[str, MegatronOptimizerGroup] = {
            _default_adapter_name: self._construct_default_optimizer_group()
        }
        self.optimizer_group[_default_adapter_name].adapter_name = _default_adapter_name
        self.active_group = _default_adapter_name
        MegatronPeft().__call__()

    def _construct_default_optimizer_group(self):
        return MegatronOptimizerGroup(
            loss_instance=VocabParallelCrossEntropyLoss(),
            template=Template(self.tokenizer_id),
            processor=InputProcessor(self.device_mesh, framework='megatron'),
            _device_mesh=self.device_mesh,
        )

    def _create_megatron_model(
        self,
        load_weights: bool = True,
        **kwargs,
    ) -> List[nn.Module]:
        from .args import get_args
        args = get_args()
        self.initialize(**kwargs)

        model = args.create_model()
        if load_weights:
            bridge = self._bridge
            for _model in model:
                bridge.load_weights(_model, args.model_dir)

        if dist.is_initialized():
            dist.barrier()

        _models = []
        for _model in model:
            _model = self._move_model_to_gpu(_model)
            _models.append(_model)
        return _models

    @staticmethod
    def _move_model_to_gpu(model: nn.Module) -> nn.Module:
        model = model.to(Platform.get_local_device())
        torch_util.synchronize()
        return model

    def _lazy_wrap_model(self):
        if not self._model_wrapped:
            self.model = self.strategy.wrap_model(self.model)
            self._model_wrapped = True

    def _get_default_group(self):
        """Get the only group has optimizer, else return the default one"""
        if len(self.optimizer_group) == 1:
            return next(iter(self.optimizer_group))
        return self.active_group

    @staticmethod
    def _not_encoded(inputs):
        assert isinstance(inputs, dict)
        return 'input_ids' not in inputs and 'input_embedding' not in inputs

    @remote_function()
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        raise NotImplementedError('Megatron only supports `forward_backward` and `forward_only`')

    @remote_function(dispatch='slice_dp', collect='last_pp')
    def forward_only(self,
                     *,
                     inputs: Union[InputFeature, List[InputFeature], List[Trajectory]],
                     micro_batch_size: Optional[int] = None,
                     **kwargs):
        """Forward pass without gradient computation.

        Args:
            inputs: Model inputs.
            **kwargs: Additional arguments.

        Returns:
            Model outputs.
        """
        return self.forward_backward(inputs=inputs, micro_batch_size=micro_batch_size, forward_only=True, **kwargs)

    @remote_function(collect='mean')
    def calculate_loss(self, **kwargs):
        raise NotImplementedError('Megatron only supports `forward_backward` and `forward_only`')

    @remote_function()
    def backward(self, **kwargs):
        raise NotImplementedError('Megatron only supports `forward_backward` and `forward_only`')

    @remote_function(dispatch='slice_dp', collect='mean', sync=True)
    def forward_backward(self,
                         *,
                         inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
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
            micro_batch_size: split and trains by `micro_batch_size`
            **kwargs: Additional arguments.

        Returns:
            Average loss value across all microbatches.
        """
        self._lazy_wrap_model()
        from functools import partial
        from megatron.core import parallel_state as mpu
        from megatron.core.pipeline_parallel import get_forward_backward_func

        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        forward_only = kwargs.pop('forward_only', False)
        optimizer_config = self.optimizer_group[adapter_name]
        loss_instance = self.optimizer_group[adapter_name].loss_instance
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        processor: InputProcessor = optimizer_config.processor
        assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'

        if micro_batch_size is None:
            micro_batch_size = 1
        inputs = processor(inputs, micro_batch_size=micro_batch_size, variable_seq_lengths=self.variable_seq_lengths)

        # Get parallelism settings for sequence padding and splitting
        cp_size = self.device_mesh.cp_world_size
        # Check actual sequence_parallel setting from model config
        # Bridge may auto-enable sequence_parallel for MoE models
        if self.variable_seq_lengths:
            seq_length = 4096
        else:
            original_seq_length = inputs[0]['input_ids'].shape[1]
            if cp_size > 1:
                divisor = 2 * cp_size
            elif self.strategy.sequence_parallel and self.device_mesh.tp_world_size > 1:
                divisor = self.device_mesh.tp_world_size
            else:
                divisor = 1

            if divisor > 1 and original_seq_length % divisor != 0:
                seq_length = original_seq_length + (divisor - original_seq_length % divisor)
            else:
                seq_length = original_seq_length

        num_microbatches = len(inputs)
        loss_extra_kwargs_per_mb = []
        if num_microbatches <= 1:
            loss_extra_kwargs_per_mb = [kwargs]
        else:
            for mb_idx in range(num_microbatches):
                mb_start = mb_idx * micro_batch_size
                mb_end = mb_start + micro_batch_size
                mb_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] > micro_batch_size:
                        mb_kwargs[key] = value[mb_start:mb_end]
                    elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] > micro_batch_size:
                        mb_kwargs[key] = value[mb_start:mb_end]
                    elif isinstance(value, (list, tuple)) and len(value) > micro_batch_size:
                        mb_kwargs[key] = value[mb_start:mb_end]
                    else:
                        # Scalars, small tensors, or non-sliceable values pass through as-is
                        mb_kwargs[key] = value
                loss_extra_kwargs_per_mb.append(mb_kwargs)

        _mb_counter = [0]  # mutable counter for closure

        def post_loss_function(output_tensor, inputs):
            mb_idx = _mb_counter[0]
            _mb_counter[0] += 1
            current_kwargs = loss_extra_kwargs_per_mb[mb_idx % len(loss_extra_kwargs_per_mb)]
            outputs = ModelOutput(logits=output_tensor)
            result = loss_instance(inputs, outputs, **current_kwargs)
            if isinstance(result, tuple):
                losses, counts = result
            else:
                losses = result
                counts = torch.tensor(1, device=losses.device)
            return self.strategy.gather_loss_for_cp(losses, counts, output_tensor)

        # Define forward step function for Megatron
        # forward_step_func(data_iterator, model) -> (output_tensor, partial(loss_func))
        def forward_step_func(data_iterator, model):
            batch = next(data_iterator)
            labels = batch.pop('labels', None)
            output_tensor = model(**batch)
            batch['labels'] = labels
            return output_tensor, partial(post_loss_function, inputs=batch)

        # Get Megatron's forward-backward function
        # This automatically selects the right scheduler based on PP config:
        # - PP > 1: forward_backward_pipelining_without_interleaving (or with interleaving if VPP)
        # - PP = 1: forward_backward_no_pipelining
        forward_backward_func = get_forward_backward_func()
        vpp_size = self.device_mesh.vpp_size

        if vpp_size is None or vpp_size == 1:
            data_iter = iter(inputs)
        else:
            data_iter = [iter(inputs) for _ in range(0, vpp_size)]

        self._accumulate_metric(optimizer_config, is_training=not forward_only)

        # Run forward-backward with Megatron's scheduler
        # Megatron handles all communication internally using proper process groups
        losses = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iter,
            model=self.model,
            num_microbatches=len(inputs),
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=forward_only,
        )

        # Extract loss from results (only last PP stage returns non-empty)
        loss = torch.tensor(0.0).to(Platform.get_local_device())
        logits = []
        count = 0
        if losses:
            for loss_dict in losses:
                if isinstance(loss_dict, dict):
                    if 'loss' in loss_dict:
                        loss += loss_dict['loss']
                        count += 1
                    if 'logits' in loss_dict:
                        logits.append(loss_dict['logits'])
                elif isinstance(loss_dict, torch.Tensor):
                    loss += loss_dict
                    count += 1

        if count > 0:
            loss /= count

        # For PP > 1, broadcast loss from last PP stage to all ranks
        # Note: mpu is imported at module level, no need to reimport
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            loss_tensor = loss.detach().clone()
            # Broadcast from last PP stage (rank with pipeline_model_parallel_rank == pp_size - 1)
            src_rank = mpu.get_pipeline_model_parallel_last_rank()
            pp_group = mpu.get_pipeline_model_parallel_group()

            torch.distributed.broadcast(loss_tensor, src=src_rank, group=pp_group)

            loss = loss_tensor.item()

        if not forward_only:
            optimizer_config.cur_step += 1

        dp_world_size = mpu.get_data_parallel_world_size()
        if dp_world_size > 1:
            if isinstance(loss, (int, float)):
                loss = torch.tensor(loss, device=Platform.get_local_device())
            # Average loss across DP group (with CP if enabled)
            dp_cp_group = mpu.get_data_parallel_group(with_context_parallel=True)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG, group=dp_cp_group)

        optimizer_config.inputs = inputs
        if forward_only:
            if len({logit.shape[0] for logit in logits}) == 1:
                logits = torch.cat(logits, dim=0)
            return {
                'loss': loss,
                'logits': logits,
            }
        else:
            optimizer_config.outputs = ModelOutput(logits=logits, loss=loss)
            if isinstance(loss, torch.Tensor):
                return loss.detach().cpu().float().numpy()
            return float(loss)

    @remote_function(dispatch='all')
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type: int = 2, **kwargs):
        # Megatron optimizer will cover this function.
        pass

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
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]

        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return

        optimizer = optimizer_config.optimizer
        assert optimizer is not None, 'Set optimizer correctly before stepping'
        # Megatron optimizer step() returns (success, grad_norm, num_zeros)

        optim_params = kwargs.pop('optim_params', {})
        if optim_params:
            for group in optimizer.param_groups:
                group['lr'] = optim_params['lr']
                if group['weight_decay'] > 0.0 and optim_params.get('weight_decay', None) is not None:
                    group['weight_decay'] = optim_params['weight_decay']
                if optim_params.get('eps') is not None:
                    group['eps'] = optim_params['eps']
                if optim_params.get('betas') is not None:
                    group['betas'] = optim_params['betas']

        success, grad_norm, num_zeros = optimizer.step()
        # Store grad_norm for later retrieval
        optimizer_config._last_grad_norm = grad_norm if grad_norm is not None else 0.0
        optimizer_config._last_step_success = success

    def _is_model_ddp_wrapped(self) -> bool:
        """Check if model is wrapped with DDP.

        Returns:
            True if model is wrapped with DDP (either Megatron DDP, LoRA DDP, or PyTorch DDP).
        """
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        return isinstance(self.model[0], (MegatronDDP, TorchDDP))

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
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]

        # For DDP-wrapped models, ALWAYS zero the gradient buffer
        # This is essential because Megatron's forward_backward_func uses
        # the buffer's state to track gradient accumulation
        if self._is_model_ddp_wrapped() and hasattr(self.model, 'zero_grad_buffer'):
            self.model.zero_grad_buffer()

        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return

        optimizer = optimizer_config.optimizer
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

    @remote_function()
    def lr_step(self, **kwargs):
        """Learning rate scheduler step.

        Args:
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]

        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return

        lr_scheduler = optimizer_config.lr_scheduler
        if lr_scheduler is not None:
            # Megatron's OptimizerParamScheduler.step() requires increment argument
            increment = kwargs.pop('increment', 1)
            lr_scheduler.step(increment=increment)

    @remote_function(dispatch='all')
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str, Callable[[InputFeature, ModelOutput, ...], torch.Tensor]],
                 **kwargs):
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
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.loss_instance = construct_class(loss_cls, Loss, twinkle.loss, **kwargs)

    def add_metric(self, metric_cls: Union[Metric, str], is_training: Optional[bool] = None, **kwargs):
        """Add an eval metric

        Args:
            metric_cls: A metric class type or id.
            is_training: Whether the metric is for training. If None, it will be used for both training and evaluation.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the metric_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['device_mesh'] = self.device_mesh
        kwargs['process_group'] = optimizer_config._dp_group
        if is_training is None or is_training is True:
            optimizer_config.train_metrics.append(construct_class(metric_cls, Metric, twinkle.metric, **kwargs))
        if not is_training:
            optimizer_config.eval_metrics.append(construct_class(metric_cls, Metric, twinkle.metric, **kwargs))

    @remote_function(dispatch='all')
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str], **kwargs):
        """Set optimizer.

        Args:
            optimizer_cls: Optimizer class or string name.
                - Standard PyTorch optimizers: 'AdamW', 'Adam', 'SGD', etc.
                - 'MegatronDistributed': Use Megatron's distributed optimizer
            **kwargs: Additional arguments.
                - For standard optimizers: lr, weight_decay, etc.
                - For MegatronDistributed: use_distributed_optimizer, clip_grad, etc.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if not self._model_wrapped:
            self.model = self.strategy.wrap_model(self.model)
            self._model_wrapped = True

        # Check if requesting Megatron distributed optimizer
        if not optimizer_cls or optimizer_cls in ('MegatronDistributedOptimizer', 'default', 'Adam'):
            optimizer_config.optimizer = self._create_megatron_optimizer(**kwargs)  # noqa
        else:
            raise NotImplementedError(
                f'Unsupported optimizer: {optimizer_cls}, only support MegatronOptimizer currently.')

    @staticmethod
    def _accumulate_metric(optimizer_config: MegatronOptimizerGroup, is_training):
        optimizer_config.accumulate_metrics(is_training)

    @remote_function(collect='first', lazy_collect=False)
    def calculate_metric(self, is_training, **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        return optimizer_config.calculate_metrics(is_training)

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
        from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer

        # Build optimizer config
        lr = kwargs.pop('lr', 1e-4)
        use_distributed_optimizer: bool = kwargs.pop('use_distributed_optimizer', False)

        opt_config = OptimizerConfig(
            optimizer='adam',
            lr=lr,
            min_lr=kwargs.get('min_lr', 0.0),
            weight_decay=kwargs.get('weight_decay', 0.01),
            adam_beta1=kwargs.get('adam_beta1', 0.9),
            adam_beta2=kwargs.get('adam_beta2', 0.999),
            adam_eps=kwargs.get('adam_eps', 1e-8),
            clip_grad=kwargs.get('clip_grad', 1.0),
            bf16=kwargs.get('bf16', True),
            use_distributed_optimizer=use_distributed_optimizer,
            overlap_param_gather=kwargs.get('overlap_param_gather', False),
            log_num_zeros_in_grad=kwargs.get('log_num_zeros_in_grad', False),
            **kwargs,
        )

        # Ensure each model chunk has ddp_config attached (required by Megatron optimizer)
        from megatron.core.distributed import DistributedDataParallelConfig
        model_chunks = self.model
        for model_chunk in model_chunks:
            assert hasattr(model_chunk, 'ddp_config')
        optimizer = get_megatron_optimizer(
            config=opt_config,
            model_chunks=model_chunks,
        )
        return optimizer

    def _create_megatron_scheduler(self, optimizer, lr_decay_steps, max_lr=1e-4, **kwargs):
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
        return OptimizerParamScheduler(
            optimizer,
            init_lr=kwargs.pop('init_lr', 0.0),
            max_lr=max_lr,
            min_lr=kwargs.pop('min_lr', 0.0),
            lr_warmup_steps=kwargs.pop('lr_warmup_steps', 0),
            lr_decay_steps=lr_decay_steps,
            lr_decay_style=kwargs.pop('lr_decay_style', 'cosine'),
            start_wd=kwargs.pop('start_wd', 0.01),
            end_wd=kwargs.pop('end_wd', 0.01),
            wd_incr_steps=lr_decay_steps,
            wd_incr_style=kwargs.pop('wd_incr_style', 'constant'),
            **kwargs,
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
        for _model in model:
            for name, param in _model.named_parameters():
                if param.requires_grad and (pattern.search(name) or is_default):
                    params[name] = param
        return params

    @remote_function(dispatch='all')
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str], **kwargs):
        """Set learning rate scheduler.

        Args:
            scheduler_cls: Scheduler class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer = optimizer_config.optimizer
        if not scheduler_cls or scheduler_cls in ('OptimizerParamScheduler', 'default'):
            optimizer_config.lr_scheduler = self._create_megatron_scheduler(optimizer, **kwargs)  # noqa
        else:
            raise NotImplementedError(
                f'Unsupported scheduler: {scheduler_cls}, only support OptimizerParamScheduler currently.')

    @remote_function(dispatch='all')
    def clip_grad_and_step(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        self.step(**kwargs)
        self.zero_grad(**kwargs)
        self.lr_step(**kwargs)

    @remote_function(dispatch='all', collect='first', sync=True)
    def save(self,
             name: Optional[str] = None,
             output_dir: Optional[str] = None,
             interval: int = 1,
             save_optimizer: bool = False,
             **kwargs):
        """Save model checkpoint.

        Always saves HF-format model weights. When ``save_optimizer`` is True,
        additionally saves optimizer / lr_scheduler / RNG state in mcore
        distributed-checkpoint format so that training can be resumed later.

        Args:
            name: Checkpoint name. Defaults to ``'checkpoint-step-{cur_step}'``.
            output_dir: Output directory. Defaults to ``'output'``.
            interval: Save each *interval* steps.
            save_optimizer: If True, save optimizer + lr_scheduler + RNG state
                alongside the HF weights for checkpoint resumption.
            **kwargs: Additional arguments forwarded to the underlying save
                methods (e.g. ``adapter_name``).
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if optimizer_config.cur_step % interval != 0:
            return

        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)

        # Always save HF-format weights (for inference / deployment).
        self._save_hf_format(checkpoint_dir, optimizer_config.adapter_name)

        # Optionally save mcore optimizer state (for training resumption).
        if save_optimizer:
            self._save_mcore_optimizer(
                checkpoint_dir,
                optimizer_config=optimizer_config,
                **kwargs,
            )

        self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)

        # Final synchronization to ensure all ranks complete save.
        if dist.is_initialized():
            dist.barrier()

        return checkpoint_dir

    @remote_function(dispatch='all')
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        """Load model weights, and optionally optimizer / scheduler / RNG state.

        Args:
            name: Checkpoint name or HuggingFace Hub model id.
            output_dir: Parent directory that contains the checkpoint folder.
                If None **and** ``load_optimizer`` is False, downloads from Hub.
            load_optimizer: If True, restore optimizer, lr_scheduler and RNG state
                from the mcore sub-checkpoint for training resumption.
            **kwargs: Additional arguments (``adapter_name``, ``no_load_optim``,
                ``no_load_rng``, etc.).
        """
        resume = kwargs.pop('load_optimizer', False)
        if output_dir is None and not resume:
            # Load from hub
            token = kwargs.pop('token', None)
            checkpoint_dir = HubOperation.download_model(name, token=token)
        else:
            if output_dir is None:
                output_dir = 'output'
            checkpoint_dir = os.path.join(output_dir, name)

        adapter_name = kwargs.get('adapter_name', self._get_default_group())

        if resume:
            self._load_mcore_optimizer(
                checkpoint_dir,
                adapter_name=adapter_name,
                **kwargs,
            )
        else:
            bridge = self._bridge
            for _model in self.strategy.unwrap_model(self.model):
                bridge.load_weights(
                    _model,
                    checkpoint_dir,
                    is_peft_format=(adapter_name != _default_adapter_name),
                )

        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _get_rng_state() -> 'ShardedObject':
        from megatron.core import parallel_state as mpu
        from megatron.core import tensor_parallel
        from megatron.core.dist_checkpointing.mapping import ShardedObject

        rng_state = {
            'random_rng_state': random.getstate(),
            'np_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states(),
        }
        rng_state_list = [rng_state]

        pp_rank = mpu.get_pipeline_model_parallel_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()

        return ShardedObject(
            'rng_state',
            rng_state_list,
            (pp_size, tp_size),
            (pp_rank, tp_rank),
            replica_id=mpu.get_data_parallel_rank(with_context_parallel=True),
        )

    @staticmethod
    def _generate_state_dict(
        model: list,
        optimizer=None,
        opt_param_scheduler=None,
        rng_state=None,
        iteration: Optional[int] = None,
        model_sd_kwargs: Optional[dict] = None,
        optim_sd_kwargs: Optional[dict] = None,
        save_optim: bool = True,
        save_rng: bool = True,
    ) -> dict:
        model_sd_kwargs = model_sd_kwargs or {}
        optim_sd_kwargs = optim_sd_kwargs or {}

        state_dict: dict = {
            'checkpoint_version': 3.0,
        }
        if iteration is not None:
            state_dict['iteration'] = iteration

        # Model sharded state dict
        for i, m in enumerate(model):
            key = 'model' if len(model) == 1 else f'model{i}'
            state_dict[key] = m.sharded_state_dict(**model_sd_kwargs)

        # Optimizer + scheduler
        if save_optim and optimizer is not None:
            state_dict['optimizer'] = optimizer.sharded_state_dict(
                state_dict,
                **optim_sd_kwargs,
            )
            if opt_param_scheduler is not None:
                state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()

        # RNG
        if save_rng and rng_state is not None:
            state_dict['rng_state'] = rng_state

        return state_dict

    def _save_mcore_optimizer(
        self,
        checkpoint_dir: str,
        optimizer_config: 'MegatronOptimizerGroup',
        **kwargs,
    ):
        from megatron.core import dist_checkpointing
        from megatron.core import parallel_state as mpu
        from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
        from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelSaveStrategyWrapper

        iteration = optimizer_config.cur_step
        iter_dir = os.path.join(checkpoint_dir, f'iter_{iteration:07d}')
        os.makedirs(iter_dir, exist_ok=True)

        sharded_sd_metadata = {
            'distrib_optim_sharding_type': 'dp_reshardable',
            'singleton_local_shards': False,
            'chained_optim_avoid_prefix': True,
        }

        rng_state = self._get_rng_state()
        model = self.model

        state_dict = self._generate_state_dict(
            model=model,
            optimizer=optimizer_config.optimizer,
            opt_param_scheduler=optimizer_config.lr_scheduler,
            rng_state=rng_state,
            iteration=iteration,
            model_sd_kwargs={'metadata': sharded_sd_metadata},
            optim_sd_kwargs={'metadata': sharded_sd_metadata},
        )

        save_strategy = get_default_save_sharded_strategy()
        if mpu.get_data_parallel_world_size(with_context_parallel=True) > 1:
            save_strategy = FullyParallelSaveStrategyWrapper(
                save_strategy,
                mpu.get_data_parallel_group(with_context_parallel=True),
            )

        dist_checkpointing.save(
            state_dict,
            iter_dir,
            save_strategy,
            async_sharded_save=False,
            validate_access_integrity=True,
            content_metadata=sharded_sd_metadata,
        )

        if dist.is_initialized():
            dist.barrier()

        # Write tracker file (rank 0 only).
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            tracker_path = os.path.join(
                checkpoint_dir,
                'latest_checkpointed_iteration.txt',
            )
            with open(tracker_path, 'w') as f:
                f.write(str(iteration))

        logging.getLogger(__name__).info(f'Saved mcore optimizer state at iteration {iteration} '
                                         f'to {checkpoint_dir}')

    def _load_mcore_optimizer(
        self,
        checkpoint_dir: str,
        adapter_name: str = '',
        **kwargs,
    ):
        from megatron.core import dist_checkpointing
        from megatron.core import parallel_state as mpu
        from megatron.core import tensor_parallel
        from megatron.core.dist_checkpointing.serialization import get_default_load_sharded_strategy
        from megatron.core.dist_checkpointing.strategies.fully_parallel import FullyParallelLoadStrategyWrapper

        no_load_optim = kwargs.pop('no_load_optim', False)
        no_load_rng = kwargs.pop('no_load_rng', False)

        optimizer_config = self.optimizer_group.get(adapter_name or self._get_default_group(), )

        # Read iteration from tracker file.
        tracker_path = os.path.join(
            checkpoint_dir,
            'latest_checkpointed_iteration.txt',
        )
        iteration = self._read_iteration(tracker_path)
        if iteration == 0:
            logging.getLogger(__name__).warning(f'No checkpoint found in {checkpoint_dir}')
            return

        iter_dir = os.path.join(checkpoint_dir, f'iter_{iteration:07d}')

        # Load common (non-sharded) state to inspect content metadata.
        common_state = dist_checkpointing.load_common_state_dict(iter_dir)
        sharded_sd_metadata = dist_checkpointing.load_content_metadata(preloaded_state_dict=common_state, )

        # Build optimizer / scheduler references for the sharded state dict.
        optimizer = optimizer_config.optimizer if not no_load_optim else None
        opt_param_scheduler = (optimizer_config.lr_scheduler if not no_load_optim else None)
        rng_state = self._get_rng_state() if not no_load_rng else None

        optim_sd_kwargs = dict(metadata=sharded_sd_metadata, is_loading=True)
        model_sd_kwargs = dict(metadata=sharded_sd_metadata)

        sharded_state_dict = self._generate_state_dict(
            model=self.model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            rng_state=rng_state,
            iteration=iteration,
            model_sd_kwargs=model_sd_kwargs,
            optim_sd_kwargs=optim_sd_kwargs,
        )

        # Load using fully-parallel strategy for speed.
        load_strategy = get_default_load_sharded_strategy(iter_dir)
        if mpu.get_data_parallel_world_size(with_context_parallel=True) > 1:
            load_strategy = FullyParallelLoadStrategyWrapper(
                load_strategy,
                mpu.get_data_parallel_group(with_context_parallel=True),
            )
        state_dict = dist_checkpointing.load(
            sharded_state_dict,
            iter_dir,
            load_strategy,
        )

        # Restore model weights.
        if len(self.model) == 1:
            self.model[0].load_state_dict(state_dict['model'], strict=False)
        else:
            for i, m in enumerate(self.model):
                key = f'model{i}'
                if key in state_dict:
                    m.load_state_dict(state_dict[key], strict=False)

        # Restore optimizer + LR scheduler.
        if not no_load_optim and optimizer is not None and 'optimizer' in state_dict:
            optimizer.load_state_dict(state_dict['optimizer'])
            if (opt_param_scheduler is not None and 'opt_param_scheduler' in state_dict):
                opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'], )

        if not no_load_rng and 'rng_state' in state_dict:
            rng = state_dict['rng_state']
            rng = rng[0]
            random.setstate(rng['random_rng_state'])
            np.random.set_state(rng['np_rng_state'])
            torch.set_rng_state(rng['torch_rng_state'])
            torch.cuda.set_rng_state(rng['cuda_rng_state'])
            tensor_parallel.get_cuda_rng_tracker().set_states(rng['rng_tracker_states'], )

        # Restore iteration counter.
        if optimizer_config is not None and 'iteration' in state_dict:
            optimizer_config.cur_step = state_dict['iteration']

        if dist.is_initialized():
            dist.barrier()

        logging.getLogger(__name__).info(f'Resumed from mcore checkpoint at iteration {iteration} '
                                         f'from {checkpoint_dir}')

    @staticmethod
    def _read_iteration(tracker_path: str) -> int:
        if not os.path.exists(tracker_path):
            return 0
        with open(tracker_path) as f:
            iteration = int(f.read().strip())
        if torch.distributed.is_initialized():
            iters_cuda = torch.tensor(
                [iteration],
                dtype=torch.long,
                device='cuda',
            )
            torch.distributed.all_reduce(
                iters_cuda,
                op=torch.distributed.ReduceOp.MAX,
            )
            iteration = iters_cuda[0].item()
        return iteration

    def _save_hf_format(self, output_dir: str, adapter_name: str, lora_converter=None):
        """Save in HuggingFace format using bridge adapter.

        For distributed training:
        - All PP ranks participate in export (each has different layers)
        - Only DP rank 0 actually writes to disk
        - Uses barrier for synchronization

        For LoRA training:
        - Saves in PEFT format (adapter_model.safetensors + adapter_config.json)
        """
        # Check if this is LoRA training
        is_peft_format = (adapter_name != _default_adapter_name)

        # Create output directory on rank 0 only
        from megatron.core import parallel_state as mpu
        dp_rank = mpu.get_data_parallel_rank() if mpu.is_initialized() else 0

        if dp_rank == 0:
            os.makedirs(output_dir, exist_ok=True)

        # Synchronize before saving
        if dist.is_initialized():
            dist.barrier()

        # Get the model (unwrap if DDP wrapped)
        model = self.strategy.unwrap_model(self.model)

        self._bridge.save_weights(
            model, output_dir, is_peft_format=is_peft_format, adapter_name=adapter_name, lora_converter=lora_converter)

        # Save config on rank 0 only
        if dp_rank == 0:
            self.hf_config.save_pretrained(output_dir)

    def _save_megatron_format(self, output_dir: str, adapter_name: str, lora_converter=None):
        """Save in Megatron checkpoint format."""
        os.makedirs(output_dir, exist_ok=True)

        state_dict = self._get_trainable_parameters(adapter_name)
        cpu_state_dict = {}
        for k, v in state_dict.items():
            if lora_converter is not None:
                k, v = lora_converter(k, v)
            if k is not None and v is not None:
                cpu_state_dict[k] = v.cpu()

        # Save with rank info for distributed checkpointing
        rank = dist.get_rank() if dist.is_initialized() else 0
        checkpoint_path = os.path.join(output_dir, f'model_rank{rank}.pt')
        torch.save(cpu_state_dict, checkpoint_path)

    def _save_tokenizer(self, output_dir: str, **kwargs):
        from twinkle.utils import is_last_rank
        if not is_last_rank():
            return

        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        template_ins = optimizer_config.template
        if template_ins is not None:
            template_ins.processor.save_pretrained(output_dir)
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
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        return self._get_trainable_parameters(adapter_name)

    def get_hf_state_dict(self, adapter_name: str = '') -> Generator[Tuple[str, torch.Tensor], None, None]:
        """Get model weights in HuggingFace format as a generator.

        This method exports Megatron model weights to HuggingFace format using
        the bridge's export_weights method. Returns a generator to avoid OOM
        for large models - weights are converted one by one.

        This is the preferred method for weight synchronization to vLLM, as it:
        1. Converts Megatron format to HF format on-the-fly
        2. Uses generator pattern to avoid loading all weights into memory
        3. Works with IPCWeightLoader's bucket-based transfer

        Args:
            adapter_name: Name of the adapter. Empty string for base model.

        Yields:
            Tuple of (parameter_name, tensor) in HuggingFace format.

        Example:
            >>> for name, tensor in model.get_hf_state_dict():
            ...     print(f"{name}: {tensor.shape}")
        """
        model = self.strategy.unwrap_model(self.model)
        yield from self._bridge.export_weights(
            model,
            target_device=None,  # Keep on current device for IPC transfer
            only_last_rank=False,  # All ranks participate in weight sync
            is_peft_format=bool(adapter_name),
            adapter_name=adapter_name if adapter_name else None,
            tqdm_desc='Weight sync: ',
        )

    def _patch_adapter(self, adapter_name: str, config_or_dir: Union[PeftConfig, str, Dict[str, Any]], **kwargs):
        from .tuners.utils import get_target_modules, patch_deepcopy, set_linear_is_expert
        assert adapter_name, 'Use a non-empty adapter_name'
        model = self.strategy.unwrap_model(self.model)
        if isinstance(config_or_dir, str):
            config_or_dir = HubOperation.download_model(config_or_dir)

        _models = []
        for _model in model:
            # Mark expert layers for MoE models
            set_linear_is_expert(_model)
            if isinstance(config_or_dir, str):
                _model = PeftModel.from_pretrained(
                    _model, config_or_dir, adapter_name=adapter_name, is_trainable=kwargs.get('is_trainable', True))
                config = _model.peft_config
            else:
                if isinstance(config_or_dir, dict):
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
                    _model = get_peft_model(_model, config, adapter_name=adapter_name)
                # setting average_gradients_across_tp_domain
                for m in _model.modules():
                    if isinstance(m, LoraLinear):
                        # just check
                        # TODO untested code
                        from .args import get_args
                        args = get_args()
                        from .tuners import LoraParallelLinear
                        assert args.is_multimodal and not isinstance(m, LoraParallelLinear)
                        for p in m.parameters():
                            if p.requires_grad:
                                p.average_gradients_across_tp_domain = True
            _models.append(_model)
        self.model = _models

        # Create optimizer group for adapter
        self.optimizer_group[adapter_name] = self._construct_default_optimizer_group()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config
        self.optimizer_group[adapter_name].gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        # Fix: use .processor instead of .tokenizer - Template class uses self.processor
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor
        self.active_group = adapter_name

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
        self._patch_adapter(adapter_name, config_or_dir, **kwargs)

    @remote_function()
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs):
        apply_patch(self.model, patch_cls, **kwargs)

    @remote_function(dispatch='all')
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """Set template for input encoding.

        Args:
            template_cls: Template class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.template = construct_class(template_cls, Template, twinkle.template, **kwargs)

    @remote_function(dispatch='all')
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str, Callable], **kwargs):
        """Set input processor.

        Args:
            processor_cls: Processor class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['framework'] = 'megatron'
        # processor/base.py: self.device_mesh.cp_world_size
        kwargs['device_mesh'] = kwargs.get('device_mesh', self.device_mesh)
        optimizer_config.processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)

    @remote_function(execute='first', lazy_collect=False)
    def get_train_configs(self, **kwargs):
        """Get training configuration summary.

        Args:
            **kwargs: Additional arguments.

        Returns:
            Configuration summary string.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]

        expr = 'Backend: Megatron-Core\n'
        expr += f'DP size: {self.device_mesh.dp_world_size}\n'
        expr += f'TP size: {self.device_mesh.tp_world_size}\n'
        expr += f'  - VPP size: {self.device_mesh.vpp_size}\n'
        expr += f'PP size: {self.device_mesh.pp_world_size}\n'
        expr += f'CP size: {self.device_mesh.cp_world_size}\n'
        expr += f'EP size: {self.device_mesh.ep_size}\n'
        expr += f'Sequence Parallel: {self.strategy.sequence_parallel}\n'

        if optimizer_config.adapter_config is not None:
            config = optimizer_config.adapter_config.__dict__
            config = {key: str(value) for key, value in config.items() if value is not None}
            expr += f'Adapter config:\n{json.dumps(config, indent=2, ensure_ascii=False)}\n'

        if optimizer_config.optimizer:
            expr += f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
            expr += f'Learning rate: {optimizer_config.optimizer.chained_optimizers[0].config.lr}\n'
        if optimizer_config.lr_scheduler:
            expr += f'LR scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
        expr += f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n'

        return expr

    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return

        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

        from .args import get_args
        self._try_init_process_group()
        args = get_args()
        init_kwargs = {
            'tensor_model_parallel_size': args.tensor_model_parallel_size,
            'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
            'context_parallel_size': args.context_parallel_size,
            'virtual_pipeline_model_parallel_size': args.virtual_pipeline_model_parallel_size,
            'expert_model_parallel_size': args.expert_model_parallel_size,
        }

        if args.order:
            init_kwargs['order'] = args.order

        if exists('megatron_core>=0.13'):
            init_kwargs['expert_tensor_parallel_size'] = args.expert_tensor_parallel_size

        # Filter out kwargs that are not valid for initialize_model_parallel
        # Dynamically check the signature to exclude unsupported parameters
        valid_params = set(inspect.signature(parallel_state.initialize_model_parallel).parameters.keys())
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
        init_kwargs.update(filtered_kwargs)
        parallel_state.initialize_model_parallel(**init_kwargs)
        model_parallel_cuda_manual_seed(self._seed)

        self._parallel_state = parallel_state
        self._initialized = True

    @property
    def _bridge(self) -> 'GPTBridge':
        if not hasattr(self, '_bridge_instance'):
            from .args import get_args
            from .model import get_megatron_model_meta
            args = get_args()
            megatron_model_meta = get_megatron_model_meta(args.hf_model_type)
            assert megatron_model_meta is not None, f'Model: {args.hf_model_type} is not supported.'
            self._bridge_instance = megatron_model_meta.bridge_cls()

        return self._bridge_instance

    # ── Checkpoint Engine (from CheckpointEngineMixin) ──────────────────
    # prepare_checkpoint_engine, init_checkpoint_process_group, and
    # finalize_checkpoint_engine are inherited from CheckpointEngineMixin.
    #
    # Key difference from TransformersModel: Megatron uses TP/PP, so
    # get_hf_state_dict() internally performs TP allgather and handles PP
    # layer distribution.  All model ranks MUST execute the weight generator
    # concurrently for the collective communications to complete.  Only
    # model_actor[0] (rank=0 in the checkpoint engine) actually broadcasts
    # via NCCL; others consume the generator silently (rank=-1).

    @remote_function(dispatch='all', lazy_collect=True)
    def send_weights(
        self,
        adapter_name: str = None,
        base_sync_done: bool = False,
        merge_and_sync: bool = False,
    ):
        if adapter_name is None:
            adapter_name = self._get_default_group()
        engine = self._get_or_create_checkpoint_engine()

        is_peft_format = (adapter_name != _default_adapter_name)

        # Megatron uses padded_vocab_size for TP alignment (rounded up to
        # TP * 128).  vLLM creates its embedding / lm_head from the original
        # HF vocab_size, so weight_loader asserts shape[0] == org_vocab_size.
        # Trim any tensor whose dim-0 equals padded_vocab_size back to
        # org_vocab_size — this is shape-based, not name-based, so it works
        # regardless of the model architecture's naming convention.
        from .args import get_args
        args = get_args()
        org_vocab_size = getattr(self.hf_config, 'vocab_size', args.padded_vocab_size)
        _padded_vocab_size = args.padded_vocab_size

        def _trim_vocab(name, tensor):
            if _padded_vocab_size != org_vocab_size and tensor.shape[0] == _padded_vocab_size:
                tensor = tensor[:org_vocab_size]
            return name, tensor

        if base_sync_done and adapter_name:
            if merge_and_sync:

                def weight_generator():
                    for _model in self.strategy.unwrap_model(self.model):
                        if isinstance(_model, PeftModel):
                            _model.merge_adapter()
                    for name, tensor in self.get_hf_state_dict(adapter_name=''):
                        if name is None or tensor is None:
                            continue
                        # Skip LoRA-specific weights for base model sync
                        if 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name:
                            continue
                        yield _trim_vocab(name, tensor)
                    for _model in self.strategy.unwrap_model(self.model):
                        if isinstance(_model, PeftModel):
                            _model.unmerge_adapter()
            else:
                # ── LoRA-only mode ────────────────────────────────────────────
                # Export only LoRA adapter weights via the bridge.
                # The bridge may also yield non-LoRA weights (e.g. embed_tokens
                # for modules_to_save), filter to only lora_A/lora_B tensors.
                def weight_generator():
                    for name, tensor in self.get_hf_state_dict(adapter_name=adapter_name):
                        if name is None or tensor is None:
                            continue
                        if 'lora' not in name:
                            continue
                        yield name, tensor

        else:

            def _raw_weights():
                for name, tensor in self.get_hf_state_dict(adapter_name=''):
                    if name is None or tensor is None:
                        continue
                    # Skip LoRA-specific weights for base model sync
                    if 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name:
                        continue
                    yield _trim_vocab(name, tensor)

            def weight_generator():
                if is_peft_format:
                    yield from _add_base_layer_suffix(_raw_weights())
                else:
                    yield from _raw_weights()

        is_sender = (engine.rank is not None and engine.rank == 0)

        if not is_sender:
            for _name, _tensor in weight_generator():
                pass
            return

        import queue
        buf: queue.Queue = queue.Queue(maxsize=4)
        error: list = []

        def _send():

            def _iter():
                while (item := buf.get()) is not None:
                    yield item

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(engine.send_weights(_iter()))
            except Exception as exc:
                error.append(exc)
            finally:
                loop.close()

        sender = threading.Thread(target=_send, name='ce-broadcast', daemon=True)
        sender.start()
        try:
            for name, tensor in weight_generator():
                buf.put((name, tensor.clone()))
                if error:
                    break
        finally:
            buf.put(None)  # sentinel
        sender.join()
        if error:
            raise error[0]

    @remote_function(collect='first')
    def get_peft_config_dict(self, adapter_name: str = None) -> dict:
        """Return the PEFT config as a dict for vLLM's PEFTHelper.

        Used by CheckpointEngineManager for LoRA-only weight sync.

        Returns:
            PEFT config dict, or None if no LoRA adapter is present.
        """
        if adapter_name is None:
            adapter_name = self._get_default_group()
        optimizer_config = self.optimizer_group.get(adapter_name)
        if optimizer_config is None or optimizer_config.adapter_config is None:
            return None
        config = optimizer_config.adapter_config
        if isinstance(config, dict):
            config = config.get(adapter_name, next(iter(config.values())))
        return config.to_dict() if hasattr(config, 'to_dict') else dict(config)
