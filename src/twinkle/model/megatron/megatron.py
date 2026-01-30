# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft import PeftModel, PeftConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoConfig
from transformers import PretrainedConfig

import twinkle
import twinkle.metric
from twinkle import DeviceMesh, remote_class, remote_function, template, Platform
from twinkle import requires
from twinkle import torch_util
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss, VocabParallelCrossEntropyLoss
from twinkle.metric import Metric, LossMetric, Accuracy, TrainMetric
from twinkle.model.base import TwinkleModel
from twinkle.processor import InputProcessor
from twinkle.template import Template
from .strategy import MegatronStrategy
from twinkle.utils import construct_class, exists
from .args import get_args, set_args, TwinkleMegatronArgs
from .model import get_megatron_model_meta, GPTBridge


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
    train_metrics: List[Metric] = field(default_factory=list)
    eval_metrics: List[Metric] = field(default_factory=list)
    _device_mesh: DeviceMesh = None
    # Megatron optimizer specific fields
    _last_grad_norm: float = 0.0
    _last_step_success: bool = True

    def do_grad_sync(self,
                     gradient_accumulation_steps: Optional[int] = None
                     ) -> bool:
        """Check if gradient synchronization should happen."""
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        return (self.cur_step-1) % gradient_accumulation_steps == 0 and self.cur_step > 0


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
                metric.accumulate(self.inputs, {**self.outputs, 'lr': self._get_lr(), 'step': self.cur_step})

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
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
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
        self.variable_seq_lengths = kwargs.get('variable_seq_lengths', False)
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
        }
        if kwargs.get('recompute_method'):
            ac_kwargs['recompute_method'] = kwargs.get('recompute_method')
        if kwargs.get('recompute_num_layers'):
            ac_kwargs['recompute_num_layers'] = kwargs.get('recompute_num_layers')

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
        self.optimizer_group: Dict[str, MegatronOptimizerGroup] = {_default_adapter_name: self._construct_default_optimizer_group()}
        MegatronPeft().patch()


    def _construct_default_optimizer_group(self):
        return MegatronOptimizerGroup(
            loss_instance=VocabParallelCrossEntropyLoss(),
            template=Template(self.tokenizer_id),
            processor=InputProcessor(self.device_mesh),
            _device_mesh=self.device_mesh,
        )

    def _create_megatron_model(
        self,
        load_weights: bool = True,
        **kwargs,
    ) -> List[nn.Module]:
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

    @staticmethod
    def _not_encoded(inputs):
        assert isinstance(inputs, dict)
        return 'input_ids' not in inputs and 'input_embedding' not in inputs

    @remote_function()
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature],
                                       Trajectory, List[Trajectory]],
                **kwargs):
        raise NotImplementedError(f'Megatron only supports `forward_backward` and `forward_only`')

    @remote_function(dispatch='slice_dp', collect='last_pp')
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
        return self.forward_backward(inputs=inputs, micro_batch_size=micro_batch_size, forward_only=True, **kwargs)

    @remote_function(collect='mean')
    def calculate_loss(self, **kwargs):
        raise NotImplementedError(f'Megatron only supports `forward_backward` and `forward_only`')

    @remote_function()
    def backward(self, **kwargs):
        raise NotImplementedError(f'Megatron only supports `forward_backward` and `forward_only`')

    @remote_function(dispatch='slice_dp', collect='mean', sync=True)
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
            micro_batch_size: split and trains by `micro_batch_size`
            **kwargs: Additional arguments.

        Returns:
            Average loss value across all microbatches.
        """
        self._lazy_wrap_model()
        from functools import partial
        from megatron.core.pipeline_parallel import get_forward_backward_func
        from megatron.core import parallel_state as mpu

        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        forward_only = kwargs.pop('forward_only', False)
        optimizer_config = self.optimizer_group[adapter_name]
        loss_instance = self.optimizer_group[adapter_name].loss_instance

        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list) and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs) # noqa
        processor: InputProcessor = optimizer_config.processor
        assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'

        vpp_size = self.device_mesh.vpp_size
        if vpp_size is None or vpp_size == 1:
            inputs = [processor(inputs)]
            micro_batch_size = inputs[0]['input_ids'].shape[0]
        else:
            if micro_batch_size is None:
                micro_batch_size = 1
            inputs = processor(inputs, micro_batch_size=micro_batch_size, variable_seq_lengths=self.variable_seq_lengths)

        # Get parallelism settings for sequence padding and splitting
        cp_size = self.device_mesh.cp_world_size
        # Check actual sequence_parallel setting from model config
        # Bridge may auto-enable sequence_parallel for MoE models
        if self.variable_seq_lengths:
            seq_length = None
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

        def post_loss_function(output_tensor, inputs):
            outputs = {}
            outputs['logits'] = output_tensor
            losses, counts = loss_instance(inputs, outputs)
            return self.strategy.gather_loss_for_cp(losses, counts, output_tensor)

        # Define forward step function for Megatron
        # forward_step_func(data_iterator, model) -> (output_tensor, partial(loss_func))
        def forward_step_func(data_iterator, model):
            batch = next(data_iterator)
            batch = self.strategy.split_inputs_for_cp(batch)
            input_ids = batch.get('input_ids')
            position_ids = batch.get('position_ids')
            attention_mask = batch.get('attention_mask')
            batch_labels = batch.get('labels')

            extra_kwargs = self.get_extra_vlm_kwargs(batch)

            # Forward pass with labels - Megatron will compute loss internally
            # This uses Megatron's compute_language_model_loss which properly handles
            # vocab parallel cross entropy
            output_tensor = model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                # labels=batch_labels,  # Pass labels to let Megatron compute loss
                **extra_kwargs,
            )
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

            torch.distributed.broadcast(loss_tensor,
                                        src=src_rank,
                                        group=pp_group)

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
            if len(set([logit.shape[0] for logit in logits])) == 1:
                logits = torch.cat(logits, dim=0)
            return {
                'loss': loss,
                'logits': logits,
            }
        else:
            optimizer_config.outputs = {
                'loss': loss,
                'logits': logits,
            }
            if isinstance(loss, torch.Tensor):
                return loss.detach().cpu().float().numpy()
            return float(loss)

    @remote_function(dispatch='all')
    def clip_grad_norm(self,
                       max_grad_norm: float = 1.0,
                       norm_type: int = 2,
                       **kwargs):
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]

        if not optimizer_config.do_grad_sync(
                kwargs.pop('gradient_accumulation_steps', None)):
            return

        optimizer = optimizer_config.optimizer
        assert optimizer is not None, 'Set optimizer correctly before stepping'
        # Megatron optimizer step() returns (success, grad_norm, num_zeros)
        success, grad_norm, num_zeros = optimizer.step()
        optimizer_config.outputs['grad_norm'] = grad_norm
        # Store grad_norm for later retrieval
        optimizer_config._last_grad_norm = grad_norm if grad_norm is not None else 0.0
        optimizer_config._last_step_success = success

    def _is_model_ddp_wrapped(self) -> bool:
        """Check if model is wrapped with DDP.

        Returns:
            True if model is wrapped with DDP (either Megatron DDP, LoRA DDP, or PyTorch DDP).
        """
        from torch.nn.parallel import DistributedDataParallel as TorchDDP
        from megatron.core.distributed import DistributedDataParallel as MegatronDDP
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
            # Megatron's OptimizerParamScheduler.step() requires increment argument
            increment = kwargs.pop('increment', 1)
            lr_scheduler.step(increment=increment)

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
        is_training = kwargs.pop('is_training', None)
        if is_training is None or is_training is True:
            optimizer_config.train_metrics.append(construct_class(metric_cls, Metric, twinkle.metric, **kwargs))
        if not is_training:
            optimizer_config.eval_metrics.append(construct_class(metric_cls, Metric, twinkle.metric, **kwargs))

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
        if not self._model_wrapped:
            self.model = self.strategy.wrap_model(self.model)
            self._model_wrapped = True

        # Check if requesting Megatron distributed optimizer
        if not optimizer_cls or optimizer_cls in ('MegatronDistributedOptimizer', 'default'):
            optimizer_config.optimizer = self._create_megatron_optimizer(**kwargs) # noqa
        else:
            raise NotImplementedError(f'Unsupported optimizer: {optimizer_cls}, only support MegatronOptimizer currently.')

    @staticmethod
    def _accumulate_metric(optimizer_config: MegatronOptimizerGroup, is_training):
        optimizer_config.accumulate_metrics(is_training)

    @remote_function(collect='first', lazy_collect=False)
    def calculate_metric(self, is_training, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
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
        from megatron.core.optimizer import get_megatron_optimizer, OptimizerConfig

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
        for _model in model:
            for name, param in _model.named_parameters():
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
        optimizer = optimizer_config.optimizer
        if not scheduler_cls or scheduler_cls in ('OptimizerParamScheduler', 'default'):
            optimizer_config.lr_scheduler = self._create_megatron_scheduler(optimizer, **kwargs) # noqa
        else:
            raise NotImplementedError(f'Unsupported scheduler: {scheduler_cls}, only support OptimizerParamScheduler currently.')

    @remote_function(dispatch='all')
    def clip_grad_and_step(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        self.step(**kwargs)
        self.zero_grad(**kwargs)
        self.lr_step(**kwargs)

    @remote_function(dispatch='all', sync=True)
    def save(self, name: Optional[str] = None, output_dir: Optional[str] = None, interval: int = 1, **kwargs):
        """Save model checkpoint.

        Args:
            output_dir: Output directory.
            interval: Save each interval steps.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if optimizer_config.cur_step % interval != 0:
            return

        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        save_format = kwargs.pop('save_format', 'hf')  # 'hf' or 'megatron'

        if save_format == 'hf':
            self._save_hf_format(checkpoint_dir, optimizer_config.adapter_name)
        else:
            self._save_megatron_format(checkpoint_dir, optimizer_config.adapter_name)

        self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)
        
        # Final synchronization to ensure all ranks complete save
        if dist.is_initialized():
            dist.barrier()

    def load(self, name: Optional[str], output_dir: Optional[str] = None, **kwargs):
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        bridge = self._bridge
        for _model in self.model:
            bridge.load_weights(_model, checkpoint_dir)

        if dist.is_initialized():
            dist.barrier()

    def _save_hf_format(self, output_dir: str, adapter_name: str, lora_converter = None):
        """Save in HuggingFace format using bridge adapter.

        For distributed training:
        - All PP ranks participate in export (each has different layers)
        - Only DP rank 0 actually writes to disk
        - Uses barrier for synchronization

        For LoRA training:
        - Saves in PEFT format (adapter_model.safetensors + adapter_config.json)
        """
        # Check if this is LoRA training
        is_peft_format = isinstance(self.strategy.unwrap_model(self.model)[0], PeftModel)

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

        self._bridge.save_weights(model,
                             output_dir,
                             is_peft_format=is_peft_format,
                             adapter_name=adapter_name,
                             lora_converter=lora_converter)

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

    def _save_tokenizer(self,
                        output_dir: str,
                        **kwargs):
        from twinkle.utils.platform import is_last_rank
        if not is_last_rank():
            return
            
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

    def _patch_adapter(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        from .tuners.utils import set_linear_is_expert, get_target_modules, patch_deepcopy
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
        self.optimizer_group[adapter_name] = self._construct_default_optimizer_group()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config
        self.optimizer_group[
            adapter_name].gradient_accumulation_steps = kwargs.get(
            'gradient_accumulation_steps', 1)
        # Fix: use .processor instead of .tokenizer - Template class uses self.processor
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor

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


    @remote_function(dispatch='all')
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """Set template for input encoding.

        Args:
            template_cls: Template class or string name.
            **kwargs: Additional arguments.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.template = construct_class(template_cls, Template, twinkle.template, **kwargs)

    @remote_function(dispatch='all')
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs):
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
            expr += f'Learning rate: {optimizer_config.optimizer.chained_optimizers[0].config.lr}\n'
        if optimizer_config.lr_scheduler:
            expr += f'LR scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
        expr += f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n'

        return expr

    def get_extra_vlm_kwargs(self, batch):
        extra_kwargs = {}
        for key in ['pixel_values', 'pixel_values_videos', 'image_grid_thw', 
                    'video_grid_thw', 'packed_seq_params']:
            if key in batch and batch[key] is not None:
                extra_kwargs[key] = batch[key]
        return extra_kwargs

    def initialize(self, **kwargs) -> None:
        if self._initialized:
            return

        from megatron.core import parallel_state
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl', device_id=torch.device(Platform.get_local_device()))
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
    def _bridge(self) -> GPTBridge:
        if not hasattr(self, '_bridge_instance'):
            args = get_args()
            megatron_model_meta = get_megatron_model_meta(args.hf_model_type)
            assert megatron_model_meta is not None, f'Model: {args.hf_model_type} is not supported.'
            self._bridge_instance = megatron_model_meta.bridge_cls()
            
        return self._bridge_instance
