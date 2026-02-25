# Copyright (c) ModelScope Contributors. All rights reserved.
import asyncio
import contextlib
import json
import os
import re
import threading
import torch
import torch.distributed as dist
import transformers
from dataclasses import dataclass, field
from peft import PeftConfig, PeftModel, get_peft_model
from peft.utils import load_peft_weights, set_peft_model_state_dict
from safetensors.torch import save_file
from torch import GradScaler
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union, overload

import twinkle
import twinkle.module.scheduler
from twinkle import DeviceMesh, Platform, remote_class, remote_function
from twinkle.checkpoint_engine import CheckpointEngine
from twinkle.checkpoint_engine.mixin import CheckpointEngineMixin
from twinkle.data_format import InputFeature, ModelOutput, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import CrossEntropyLoss, Loss
from twinkle.metric import Accuracy, LossMetric, Metric, TrainMetric
from twinkle.model.base import TwinkleModel
from twinkle.model.transformers.moe import apply_expert_parallel
from twinkle.model.transformers.strategy import AccelerateStrategy, NativeFSDPStrategy
from twinkle.patch import Patch, apply_patch
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils import construct_class, torch_util
from twinkle.utils.framework import Torch
from twinkle.utils.grad_clip import normalize_and_clip_grad_norm


@dataclass
class OptimizerGroup:
    adapter_name: str = None
    adapter_config: PeftConfig = None
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None
    inputs: List[InputFeature] = None
    outputs: ModelOutput = None
    loss_instance: Loss = CrossEntropyLoss
    loss_value: Any = None
    template: Template = None
    processor: InputProcessor = None
    scaler: GradScaler = None
    _last_grad_norm: float = 0.0
    scaler_has_nan: bool = False
    gradient_accumulation_steps: int = 1
    cur_step: int = 0
    num_tokens: int = 0
    train_metrics: List[Metric] = field(default_factory=list)
    eval_metrics: List[Metric] = field(default_factory=list)
    checkpoint_engine: CheckpointEngine = None
    _dp_group = None
    _device_mesh: DeviceMesh = None

    def do_grad_sync(self, gradient_accumulation_steps: Optional[int] = None) -> bool:
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        else:
            self.gradient_accumulation_steps = gradient_accumulation_steps
        return (self.cur_step - 1) % gradient_accumulation_steps == 0 and self.cur_step > 1

    def __post_init__(self):
        self._ensure_dp_group()
        self._build_metrics()

    def _build_metrics(self):
        self.train_metrics = [
            LossMetric(self._device_mesh, self._dp_group, loss_reduction='sum'),
            Accuracy(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]

        self.eval_metrics = [
            LossMetric(self._device_mesh, self._dp_group, loss_reduction='sum'),
            Accuracy(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]

    def _ensure_dp_group(self):
        if self._dp_group is not None or self._device_mesh is None:
            return
        if self._device_mesh.data_world_size <= 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        if dist.get_world_size() < self._device_mesh.data_world_size:
            # World size is smaller than the requested dp group; skip to avoid crash.
            return
        dims = [dim for dim in ('dp', 'fsdp') if self._device_mesh.has_dim(dim)]
        if not dims:
            return
        self._dp_group = self._device_mesh.create_process_group(dims)

    def _get_lr(self):
        _lrs = []
        _default_lr = self.optimizer.defaults.get('lr')
        for param_group in self.optimizer.param_groups:
            _lrs.append(param_group.get('lr', _default_lr))
        return _lrs

    def accumulate_metrics(self, is_training):
        self._ensure_dp_group()
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
        self.inputs = None
        self.outputs = None
        return results


_default_adapter_name = ''
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.01


@remote_class()
class TransformersModel(TwinkleModel, PreTrainedModel, CheckpointEngineMixin):
    """The transformers model wrapper.

    Args:
        model_cls: The PreTrainedModel model class, only needed when creating a blank(not pretrained) model.
        config: The config of the model.
        model_id: The model id or path, this argument will be used in `from_pretrained`.
        device_mesh: The model device mesh to follow.
        mixed_precision: The mixed precision type.
        strategy: The training strategy to use.
        ddp_config: The DDP config to use.
        fsdp_config: The fsdp config to use.
        grad_scaler_config: The gradient scaler config to use.
        kwargs: Any kwargs used in `from_pretrained` or `__init__`.

    If model_id is passed in, `from_pretrained` will be used, else `__init__` will be used.
    """

    @overload
    def __init__(self, *, model_cls: Type[PreTrainedModel], config: PretrainedConfig, remote_group, **kwargs) -> None:
        ...

    @overload
    def __init__(self, *, model_id: str, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        ...

    def __init__(
            self,  # noqa
            model_cls: Optional[Union[Type[PreTrainedModel], str, Type[_BaseAutoModelClass]]] = AutoModelForCausalLM,
            model_id: Optional[str] = None,
            config: Optional[PretrainedConfig] = None,
            device_mesh: Optional[DeviceMesh] = None,
            mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
            strategy: Literal['accelerate', 'native_fsdp'] = 'accelerate',
            ddp_config: Dict[str, Any] = None,
            fsdp_config: Dict[str, Any] = None,
            grad_scaler_config: Dict[str, Any] = None,
            **kwargs):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        self._try_init_process_group()
        super(PreTrainedModel, self).__init__()
        self.model_id = model_id
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        # The Default tokenizer will be used to save with a model if no template was set.
        self._default_tokenizer = None
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self._fsdp_config = dict(fsdp_config or {})
        self._ddp_config = ddp_config or {}
        self._decide_strategy(strategy)
        self.grad_scaler_config = grad_scaler_config
        if isinstance(model_cls, str):
            model_cls = getattr(transformers, model_cls)
        if model_id is None:
            self.model = model_cls.from_config(config, **kwargs)
        else:
            model_id = HubOperation.download_model(model_id)
            self.model = model_cls.from_pretrained(model_id, config=config, **kwargs)
        # Construct sequence-parallel strategy lazily during wrapping to reduce init-time side effects.
        self.model.gradient_checkpointing_enable()
        self.sp_strategy = None
        self._model_wrapped = False
        self.optimizer_group: Dict[str, OptimizerGroup] = {
            _default_adapter_name: self._construct_default_optimizer_group()
        }
        self.optimizer_group[_default_adapter_name].adapter_name = _default_adapter_name
        self.active_group = _default_adapter_name

    def _decide_strategy(self, strategy: Literal['accelerate', 'native_fsdp']):
        self._expert_parallel_config = self._fsdp_config.pop('expert_parallel', None)
        self._enable_expert_parallel = self._should_enable_expert_parallel(self._expert_parallel_config,
                                                                           self.device_mesh)
        self._expert_parallel_applied = False
        use_native_fsdp = self._enable_expert_parallel or strategy == 'native_fsdp'
        if use_native_fsdp:
            self.strategy = NativeFSDPStrategy(
                mixed_precision=self.mixed_precision,
                fsdp_config=self._fsdp_config,
                device_mesh=self.device_mesh,
                enable_ep=self._enable_expert_parallel,
            )
        else:
            self.strategy = AccelerateStrategy(
                mixed_precision=self.mixed_precision,
                ddp_config=self._ddp_config,
                fsdp_config=self._fsdp_config,
                device_mesh=self.device_mesh)

        # Sequence parallel ("ulysses") is derived from dp/fsdp ranks; it does not change world size.
        # We construct `sp_strategy` after the underlying HF model is initialized (see __init__).
        self._enable_sp = False
        if self.device_mesh is not None:
            sp_size = getattr(self.device_mesh, 'ulysses_size', None)
            self._enable_sp = bool(sp_size and sp_size > 1)

    def _ensure_sp_strategy(self) -> None:
        if not getattr(self, '_enable_sp', False):
            return
        if self.sp_strategy is not None:
            return
        from .strategy.sequence_parallel import SequenceParallelStrategy

        sp_config = {}
        # When data-parallel gradient averaging runs across SP shards (native FSDP or
        # accelerate DDP/FSDP paths), compensate SP loss backward to keep gradient scale.
        if isinstance(self.strategy, (NativeFSDPStrategy, AccelerateStrategy)) and self.device_mesh is not None:
            if (self.device_mesh.ulysses_size or 1) > 1 and (self.device_mesh.data_world_size or 1) > 1:
                sp_config['compensate_fsdp_avg'] = True
        self.sp_strategy = SequenceParallelStrategy(
            self.device_mesh,
            sp_config,
            model=self.model,
            tokenizer_id=self.tokenizer_id,
        )

    def _get_default_group(self):
        """Get the only group has optimizer, else return the default one"""
        if len(self.optimizer_group) == 1:
            return next(iter(self.optimizer_group))
        return self.active_group

    @staticmethod
    def _not_encoded(inputs):
        assert isinstance(inputs, dict)
        return 'input_ids' not in inputs and 'input_embedding' not in inputs

    def _lazy_wrap_model(self):
        if not self._model_wrapped:
            optimizer_groups = [og for og in self.optimizer_group.values() if og.optimizer is not None]
            self._maybe_apply_expert_parallel()
            self._ensure_sp_strategy()
            if self.sp_strategy is not None:
                self.sp_strategy.initialize()
            if len(optimizer_groups) == 1:
                optimizer_group = optimizer_groups[0]
                optimizer = optimizer_group.optimizer
                assert optimizer is not None
                self.model, optimizer = self.strategy.wrap_model(self.model, optimizer)
                optimizer_group.optimizer = optimizer
            else:
                # maybe forward_only, no optimizer_group available
                self.model = self.strategy.wrap_model(self.model)
            self._model_wrapped = True

    @staticmethod
    def _should_enable_expert_parallel(expert_parallel_config: Optional[Dict[str, Any]],
                                       device_mesh: Optional[DeviceMesh]) -> bool:
        if expert_parallel_config is None or device_mesh is None:
            return False
        if not device_mesh.has_dim('ep'):
            return False
        ep_world_size = device_mesh.ep_world_size or 1
        if ep_world_size <= 1:
            return False
        return expert_parallel_config.get('enabled', True)

    def _maybe_apply_expert_parallel(self):
        if not self._enable_expert_parallel or self._expert_parallel_applied:
            return
        self._ensure_optimizer_dp_groups()
        model = self.strategy.unwrap_model(self.model)
        apply_expert_parallel(
            model,
            self.device_mesh,
            config=self._expert_parallel_config,
        )
        self._expert_parallel_applied = True

    def _ensure_optimizer_dp_groups(self):
        for optimizer_group in self.optimizer_group.values():
            if not isinstance(optimizer_group, OptimizerGroup):
                continue
            before = optimizer_group._dp_group
            optimizer_group._ensure_dp_group()
            if before is None and optimizer_group._dp_group is not None:
                optimizer_group._build_metrics()

    def _construct_default_optimizer_group(self):
        return OptimizerGroup(
            loss_instance=CrossEntropyLoss(reduction='sum'),
            template=Template(self.tokenizer_id),
            processor=InputProcessor(self.device_mesh),
            _device_mesh=self.device_mesh,
        )

    @remote_function()
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        """Call forward function and record the inputs and outputs.

        Args:
            inputs: The model inputs. Can be an encoded batch, or a list of `Trajectory`
            **kwargs:
                adapter_name: Lora adapter name.
        Returns:
            The output of the model forward.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        self.model.train()
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        processor: InputProcessor = optimizer_config.processor
        assert isinstance(processor, InputProcessor), 'Set a correct `InputProcessor` before forwarding'
        inputs: Dict[str, Any] = processor(inputs)
        if self.sp_strategy is not None:
            inputs = self.sp_strategy.preprocess_inputs(inputs)
        labels: torch.Tensor = inputs.pop('labels', None)
        optimizer_config.accumulate_metrics(True)
        outputs = self.model(**inputs)
        if self.sp_strategy is not None and labels is None:
            outputs = self.sp_strategy.postprocess_outputs(outputs)
        inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
        optimizer_config.loss_value = outputs.get('aux_loss', 0)
        return outputs

    @remote_function(dispatch='slice_dp', collect='flatten')
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        """Call forward function without grad and record the inputs and outputs.

        Args:
            inputs: The model inputs. Can be an encoded batch, or a list of `Trajectory`
            **kwargs:
                adapter_name: Lora adapter name.
        Returns:
            The output of the model forward.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if not inputs:
            raise ValueError('inputs empty, check your DataLoader outputs')
        self.model.eval()
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list)
                                                                        and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs)  # noqa
        with torch.no_grad():
            processor: InputProcessor = optimizer_config.processor
            assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
            inputs: Dict[str, Any] = processor(inputs)
            if self.sp_strategy is not None:
                inputs = self.sp_strategy.preprocess_inputs(inputs)
            labels = inputs.pop('labels', None)
            optimizer_config.accumulate_metrics(False)
            outputs = self.model(**inputs)
            if self.sp_strategy is not None and labels is None:
                outputs = self.sp_strategy.postprocess_outputs(outputs)
            inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
        optimizer_config.loss_value = outputs.get('aux_loss', 0)
        return outputs

    @remote_function(collect='mean')
    def calculate_loss(self, **kwargs):
        """Calculate loss

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for the specific loss type.
        Returns:
            A scalar loss value.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        loss_instance: Loss = optimizer_config.loss_instance
        assert isinstance(loss_instance, Loss), 'Set a loss_instance before calculating loss'
        inputs = optimizer_config.inputs
        outputs = optimizer_config.outputs
        assert inputs is not None and outputs is not None, 'Cannot calculate loss of empty inputs and outputs'
        result = loss_instance(inputs, outputs, **kwargs)
        if isinstance(result, tuple):
            loss_value, counts = result
        else:
            loss_value = result
            counts = torch.tensor(0, device=loss_value.device)
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.num_tokens += counts.item()
        if self.sp_strategy is not None and 'labels' in inputs:
            reduction = getattr(loss_instance, 'reduction', None)
            if reduction is not None:
                self.sp_strategy.sp_config['loss_reduction'] = str(reduction)
            loss_value = self.sp_strategy.reduce_loss(loss_value, inputs['labels'])
        optimizer_config.loss_value += loss_value
        outputs['loss'] = optimizer_config.loss_value
        return optimizer_config.loss_value.item()

    @remote_function()
    def backward(self, **kwargs):
        """Backward propagation.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                gradient_accumulation_steps: Number of gradient accumulation steps.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        loss_value = optimizer_config.loss_value
        assert loss_value is not None, 'Do forwarding and calculating loss before backward'
        scaler = optimizer_config.scaler
        if scaler is None and self.mixed_precision == 'fp16':
            # Auto set a grad scaler
            self.set_grad_scaler(adapter_name=adapter_name)
            scaler = optimizer_config.scaler
        if scaler is not None:
            scaler.scale(loss_value).backward()
        else:
            loss_value.backward()
        optimizer_config.cur_step += 1
        optimizer_config.loss_value = None

    @remote_function(dispatch='slice_dp', collect='mean')
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
                         **kwargs):
        """Do forward, calculate loss, and backward.

        Args:
            inputs: The model inputs. Can be an encoded batch, or a list of `Trajectory`
            **kwargs:
                adapter_name: Lora adapter name.
                gradient_accumulation_steps: Number of gradient accumulation steps.
                Any parameters needed for the specific loss type.
        Returns:
            The output of the model forward.
        """
        self.forward(inputs=inputs, **kwargs)
        loss = self.calculate_loss(**kwargs)
        self.backward(**kwargs)
        return loss

    @remote_function()
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        """ Clip the gradient norm

        Args:
            max_grad_norm: The maximum grad norm, default `1.0`.
            norm_type: Default `2`.
            **kwargs:
                adapter_name: Lora adapter name.
        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return

        optimizer = optimizer_config.optimizer
        scaler = optimizer_config.scaler
        context = contextlib.nullcontext
        if self.device_mesh is not None and self.device_mesh.tp_world_size > 1:
            from torch.distributed.tensor.experimental import implicit_replication
            context = implicit_replication

        with context():
            if scaler is not None:
                scaler.unscale_(optimizer)

            optimizer_config._ensure_dp_group()
            num_tokens = optimizer_config.num_tokens
            num_tokens = torch_util.gather_object([num_tokens], self.device_mesh, optimizer_config._dp_group)
            num_tokens = sum(num_tokens)
            parameters = list(self._get_trainable_parameters(adapter_name).values())
            grad_norm = normalize_and_clip_grad_norm(
                parameters,
                num_tokens=num_tokens,
                max_grad_norm=max_grad_norm,
                norm_type=norm_type,
                group=optimizer_config._dp_group,
            )
            optimizer_config._last_grad_norm = grad_norm
            optimizer_config.num_tokens = 0
            return grad_norm

    @remote_function(dispatch='all')
    def clip_grad_and_step(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        grad_norm = self.clip_grad_norm(max_grad_norm, norm_type, **kwargs)
        self.step(**kwargs)
        self.zero_grad(**kwargs)
        self.lr_step(**kwargs)
        return grad_norm

    def _create_param_group(self,
                            adapter_name: str,
                            lr: float = DEFAULT_LEARNING_RATE,
                            weight_decay: float = DEFAULT_WEIGHT_DECAY,
                            **kwargs):
        # Some code borrowed from transformers

        def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
            forbidden_layer_patterns = ([re.compile(pattern) for pattern in forbidden_layer_names]
                                        if forbidden_layer_names is not None else [])
            result = []
            for name, child in model.named_children():
                child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
                result += [
                    f'{name}.{n}' for n in child_params
                    if not isinstance(child, tuple(forbidden_layer_types)) and not any(
                        pattern.search(f'{name}.{n}'.lower()) for pattern in forbidden_layer_patterns)
                ]
            # Add model specific parameters that are not in any child
            result += [
                k for k in model._parameters
                if not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
            ]

            return result

        forbidden_name_patterns = [r'bias', r'layernorm', r'rmsnorm', r'(?:^|\.)norm(?:$|\.)', r'_norm(?:$|\.)']
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm], forbidden_name_patterns)
        params = self._get_trainable_parameters(adapter_name)
        decay_param_names = [n for n, p in params.items() if (n in decay_parameters and p.requires_grad)]
        no_decay_param_names = [n for n, p in params.items() if (n not in decay_parameters and p.requires_grad)]
        optimizer_grouped_parameters = [
            {
                'params': [params[n] for n in decay_param_names],
                'param_names': decay_param_names,
                'weight_decay': weight_decay,
                'lr': lr
            },
            {
                'params': [params[n] for n in no_decay_param_names],
                'param_names': no_decay_param_names,
                'weight_decay': 0.0,
                'lr': lr
            },
        ]
        return optimizer_grouped_parameters

    @remote_function()
    def step(self, **kwargs):
        """Optimizer step.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for `optimizer.step`.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        grad_accum_steps = kwargs.pop('gradient_accumulation_steps', None)
        if not optimizer_config.do_grad_sync(grad_accum_steps):
            return
        optimizer = optimizer_config.optimizer
        scaler = optimizer_config.scaler
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'

        context = contextlib.nullcontext
        if self.device_mesh is not None and self.device_mesh.tp_world_size > 1:
            from torch.distributed.tensor.experimental import implicit_replication
            context = implicit_replication

        optim_params = kwargs.pop('optim_params', {})
        if optim_params:
            assert isinstance(optimizer, (AdamW, Adam))
            for group in optimizer.param_groups:
                group['lr'] = optim_params['lr']
                if group['weight_decay'] > 0.0 and optim_params.get('weight_decay', None) is not None:
                    group['weight_decay'] = optim_params['weight_decay']
                if optim_params.get('eps') is not None:
                    group['eps'] = optim_params['eps']
                if optim_params.get('betas') is not None:
                    group['betas'] = optim_params['betas']

        with context():
            if scaler is not None:
                scaler.step(optimizer, **kwargs)
                scaler.update()
                optimizer_config.scaler_has_nan = sum(v.item()
                                                      for v in scaler._found_inf_per_device(optimizer).values()) > 0
            else:
                optimizer.step(**kwargs)

    @remote_function()
    def zero_grad(self, **kwargs):
        """Optimizer zero_grad.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for `optimizer.zero_grad`.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'
        optimizer.zero_grad(set_to_none=True)

    @remote_function()
    def lr_step(self, **kwargs):
        """Do lr_scheduler step.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for `lr_scheduler.step`.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return
        if optimizer_config.scaler_has_nan:
            return
        lr_scheduler = optimizer_config.lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str, Callable[[InputFeature, ModelOutput, ...], torch.Tensor]],
                 **kwargs):
        """Set the loss instance.

        Args:
            loss_cls: A loss class name, a loss plugin id, or a loss class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the loss instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer_config.loss_instance = construct_class(loss_cls, Loss, twinkle.loss, **kwargs)

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str, Optimizer], **kwargs):
        """Set the optimizer.

        Args:
            optimizer_cls: An optimizer class name, an optimizer plugin id, or an optimizer class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                lr: Learning rate
                weight_decay: Weight decay
                Any parameters needed to construct the optimizer instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if isinstance(optimizer_cls, Optimizer):
            optimizer_config.optimizer = optimizer_cls
            return

        params = kwargs.pop('params', None)
        if params is None:
            lr = kwargs.get('lr', DEFAULT_LEARNING_RATE)
            weight_decay = kwargs.get('weight_decay', DEFAULT_WEIGHT_DECAY)
            params = self._create_param_group(adapter_name, lr=lr, weight_decay=weight_decay)
        if self._enable_expert_parallel and 'foreach' not in kwargs:
            is_adam_family = (
                optimizer_cls in ('AdamW', 'Adam')
                or (isinstance(optimizer_cls, type) and issubclass(optimizer_cls, (AdamW, Adam))))
            if is_adam_family:
                kwargs['foreach'] = False
        optimizer_config.optimizer = construct_class(
            optimizer_cls,
            Optimizer,
            torch.optim,
            params=params,
            **kwargs,
        )

    def _get_trainable_parameters(self, adapter_name=_default_adapter_name):
        is_default = adapter_name == _default_adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        params = {}
        model = self.strategy.unwrap_model(self.model)
        for name, param in model.named_parameters():
            if param.requires_grad and (pattern.search(name) or is_default):
                params[name] = param
        return params

    @remote_function()
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str, LRScheduler], **kwargs):
        """Set the lr_scheduler.

        Args:
            scheduler_cls: An lr_scheduler class name, an lr_scheduler plugin id, or an lr_scheduler class type.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the lr_scheduler instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before setting lr_scheduler'
        kwargs['optimizer'] = optimizer
        scheduler = construct_class(scheduler_cls, LRScheduler, [torch.optim.lr_scheduler, twinkle.module.scheduler],
                                    **kwargs)
        optimizer_config.lr_scheduler = scheduler

    @remote_function()
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs):
        apply_patch(self.model, patch_cls, **kwargs)

    def __del__(self):
        HubOperation.wait_for()

    @remote_function()
    def save(self, name: Optional[str] = None, output_dir: Optional[str] = None, interval: int = 1, **kwargs):
        """Save model.

        Args:
            name: The name of checkpoint to save.
            output_dir: An output_dir to save the model.
            interval: Save each interval steps.
            **kwargs:
                adapter_name: Lora adapter name.
                save_optimizer: Whether to save optimizer state.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        if optimizer_config.cur_step % interval != 0:
            return
        model = self.strategy.unwrap_model(self.model)
        state_dict = self.get_state_dict(adapter_name=adapter_name, **kwargs)
        processed_state_dict = {}

        save_kwargs = {}

        for key, value in state_dict.items():
            key = key.replace(f'.{adapter_name}.', '.')
            processed_state_dict[key] = torch_util.to_local_tensor(value).cpu()

        if isinstance(model, PeftModel):
            if Platform.is_master():
                model.peft_config[adapter_name].save_pretrained(checkpoint_dir)
                save_file(processed_state_dict, os.path.join(checkpoint_dir, 'adapter_model.safetensors'))
        else:
            model.save_pretrained(
                checkpoint_dir, state_dict=processed_state_dict, is_main_process=Platform.is_master(), **save_kwargs)

        self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)

        if kwargs.get('save_optimizer', False):
            self._save_optimizer(checkpoint_dir, adapter_name=adapter_name)

        return checkpoint_dir

    def _save_optimizer(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]

        if Platform.is_master():
            optimizer = optimizer_config.optimizer
            lr_scheduler = optimizer_config.lr_scheduler
            if optimizer is not None:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))

    def _save_tokenizer(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        template_ins = optimizer_config.template
        if Platform.is_master():
            if template_ins is not None:
                template_ins.processor.save_pretrained(output_dir)
            else:
                self._default_tokenizer.save_pretrained(output_dir)

    @remote_function()
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        """Load model state and optionally optimizer state from a checkpoint.

        Args:
            name: The name of checkpoint to load.
            output_dir: An output_dir to load the model.
            **kwargs:
                adapter_name: Adapter to load.
                load_optimizer: Whether to load optimizer and scheduler states.
        """
        load_optimizer = kwargs.get('load_optimizer', False)
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())

        if output_dir is None:
            # load from hub
            token = kwargs.pop('token', None)
            checkpoint_dir = HubOperation.download_model(name, token=token)
        else:
            checkpoint_dir = os.path.join(output_dir, name)
        model = self.strategy.unwrap_model(self.model)
        if isinstance(model, PeftModel):
            adapter_weights = load_peft_weights(checkpoint_dir, device='cpu')

            def load_peft_weights_for_fsdp2(model, adapter_weights, adapter_name='default'):
                from torch.distributed.tensor import DTensor, distribute_tensor

                model_sd = model.state_dict()
                converted_weights = {}
                for key, value in adapter_weights.items():
                    if f'.{adapter_name}.weight' not in key:
                        key = key.replace('.weight', f'.{adapter_name}.weight')
                    if key in model_sd:
                        param = model_sd[key]
                        if isinstance(param, DTensor) and not isinstance(value, DTensor):
                            value = distribute_tensor(value.to(param.device), param.device_mesh, param.placements)
                    converted_weights[key] = value

                set_peft_model_state_dict(model, converted_weights, adapter_name=adapter_name)

            if self.device_mesh.fsdp_world_size > 1:
                load_peft_weights_for_fsdp2(model, adapter_weights, adapter_name=adapter_name)
            else:
                set_peft_model_state_dict(model, adapter_weights, adapter_name=adapter_name)
        else:
            raise NotImplementedError

        if load_optimizer:
            self._load_optimizer(checkpoint_dir, adapter_name=adapter_name)

    def _load_optimizer(self, checkpoint_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        # assume optimizer and lr_scheduler are created
        optimizer_config = self.optimizer_group[adapter_name]

        optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
        scheduler_path = os.path.join(checkpoint_dir, 'scheduler.pt')

        if os.path.exists(optimizer_path) and optimizer_config.optimizer is not None:
            state_dict = torch.load(optimizer_path, map_location='cpu')
            optimizer_config.optimizer.load_state_dict(state_dict)

        if os.path.exists(scheduler_path) and optimizer_config.lr_scheduler is not None:
            state_dict = torch.load(scheduler_path, map_location='cpu')
            optimizer_config.lr_scheduler.load_state_dict(state_dict)

    @remote_function(collect='first')
    def get_state_dict(self, **kwargs):
        return self._get_trainable_parameters(kwargs.pop('adapter_name', self._get_default_group()))

    @remote_function(collect='first')
    def get_peft_config_dict(self, adapter_name: str = None) -> dict:
        """Return the PEFT config as a dict for vLLM's PEFTHelper.

        Used by CheckpointEngineManager to pass peft_config to the sampler
        when doing LoRA-only weight sync.

        Returns:
            PEFT config dict, or None if the model has no LoRA adapter.
        """
        if adapter_name is None:
            adapter_name = self._get_default_group()
        optimizer_config = self.optimizer_group.get(adapter_name)
        if optimizer_config is None or optimizer_config.adapter_config is None:
            return None
        config = optimizer_config.adapter_config
        # PeftConfig can be a dict-like mapping (e.g. {adapter_name: LoraConfig})
        # or a single LoraConfig.  Normalize to a single config.
        if isinstance(config, dict):
            config = config.get(adapter_name, next(iter(config.values())))
        return config.to_dict() if hasattr(config, 'to_dict') else dict(config)

    @remote_function(collect='first', lazy_collect=False)
    def calculate_metric(self, is_training, **kwargs):
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        return optimizer_config.calculate_metrics(is_training)

    def _patch_adapter(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        assert adapter_name, 'Use a different adapter_name, current is empty.'
        unwrapped_model = self.strategy.unwrap_model(self.model)
        if isinstance(config_or_dir, str):
            config_or_dir = HubOperation.download_model(config_or_dir)
            _adapted_model = PeftModel.from_pretrained(
                unwrapped_model,
                model_id=config_or_dir,
                adapter_name=adapter_name,
                is_trainable=kwargs.get('is_trainable', True))
            if unwrapped_model is self.model:
                self.model = _adapted_model
            else:
                # post check: unwrapped_model must be already a peft model before wrapping ddp
                assert isinstance(unwrapped_model, PeftModel)
            config = _adapted_model.peft_config
        else:
            config = config_or_dir
            if not isinstance(unwrapped_model, PeftModel):
                assert unwrapped_model is self.model, 'Cannot wrap model with peft after DDP/FSDP!'
                self.model = get_peft_model(unwrapped_model, config, adapter_name=adapter_name)
            else:
                unwrapped_model.add_adapter(adapter_name, config)

        self.optimizer_group[adapter_name] = self.optimizer_group.pop(_default_adapter_name,
                                                                      self._construct_default_optimizer_group())
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config
        _gas_default = kwargs.get('gradient_accumulation_steps', 1)
        self.optimizer_group[adapter_name].gradient_accumulation_steps = _gas_default
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor
        self.active_group = adapter_name

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        """Add adapter to model.

        Args:
            adapter_name: The lora adapter name.
            config_or_dir:  The lora adapter config.
            **kwargs:
                is_trainable: Whether the adapter is trainable.
                gradient_accumulation_steps: The number of gradient accumulation steps
        """
        self._patch_adapter(adapter_name, config_or_dir, **kwargs)

    @remote_function()
    def set_template(self, template_cls: Union[Type[Template], str, Template], **kwargs):
        """Set template. This is optional, if you need to input `Trajectory`,
            you need to set the template to encode them.

        Args:
            template_cls: A template_cls class name, a template_cls plugin id, or a template_cls class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the template_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['model_id'] = self.tokenizer_id
        template = construct_class(template_cls, Template, twinkle.template, **kwargs)
        optimizer_config.template = template

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, InputProcessor, Callable], **kwargs):
        """Set task processor to prepare the task inputs.
        Args:
            processor_cls: A processor_cls class name, a processor_cls plugin id,
                or a processor_cls class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the processor_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['device_mesh'] = self.device_mesh
        processor = construct_class(processor_cls, InputProcessor, twinkle.processor, **kwargs)
        optimizer_config.processor = processor

    @remote_function()
    def set_grad_scaler(self, **kwargs):
        """Set the grad scaler.
        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the GradScaler instance.
        """
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        from torch.amp.grad_scaler import GradScaler
        grad_scaler_config = self.grad_scaler_config.copy()
        grad_scaler_config.update(kwargs)
        optimizer_config.scaler = GradScaler(**grad_scaler_config)

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

    def _get_nb_trainable_parameters(self, adapter_name, model):
        return PeftModel.get_nb_trainable_parameters(model)

    def _get_trainable_parameters_example(self, adapter_name, model):
        trainable_param_names = []
        for name, parameter in self.model.named_parameters():
            if parameter.requires_grad:
                trainable_param_names.append(name)
        trainable_param_names = trainable_param_names[:5] + ['...'] + trainable_param_names[-5:]
        trainable_param_names = '\n'.join(trainable_param_names)
        return trainable_param_names

    @remote_function(execute='first', lazy_collect=False)
    def get_train_configs(self, **kwargs) -> str:
        expr = ''
        adapter_name = kwargs.pop('adapter_name', self._get_default_group())
        optimizer_config = self.optimizer_group[adapter_name]
        if optimizer_config.adapter_config is not None:
            config = optimizer_config.adapter_config.__dict__
        else:
            config = {}
        config = {key: str(value) for key, value in config.items() if value is not None}
        trainable_params, all_param = self._get_nb_trainable_parameters(adapter_name, self.model)
        trainable_param_names = self._get_trainable_parameters_example(adapter_name, self.model)
        if optimizer_config.optimizer is not None:
            expr += (f'Adapter config:\n'
                     f'{json.dumps(config, indent=2, ensure_ascii=False)}\n'
                     f'Trainable parameters examples:\n'
                     f'{trainable_param_names}\n'
                     f'Trainable params: {trainable_params:,d} || all params: {all_param:,d} || '
                     f'trainable%: {100 * trainable_params / all_param:.4f}\n'
                     f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
                     f'Learning rate: {optimizer_config.optimizer.defaults.get("lr", "No default lr")}\n'
                     f'Lr scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
                     f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n')
        else:
            expr += (f'Adapter config:\n'
                     f'{json.dumps(config, indent=2, ensure_ascii=False)}\n'
                     f'Trainable parameters examples:\n'
                     f'{trainable_param_names}\n'
                     f'Trainable params: {trainable_params:,d} || all params: {all_param:,d} || '
                     f'trainable%: {100 * trainable_params / all_param:.4f}%\n')
        return expr

    # =========================================================================
    # Checkpoint Engine — Weight Sync (from CheckpointEngineMixin)
    # =========================================================================
    # prepare_checkpoint_engine, init_checkpoint_process_group, and
    # finalize_checkpoint_engine are inherited from CheckpointEngineMixin.
    # Only send_weights_via_checkpoint_engine is model-specific.

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
        # Get state dict from unwrapped model
        model = self.strategy.unwrap_model(self.model)

        if base_sync_done and adapter_name:
            if merge_and_sync:

                def weight_generator():
                    if isinstance(model, PeftModel):
                        model.merge_adapter()
                    for name, tensor in model.state_dict().items():
                        # Skip LoRA-specific weights for base model sync
                        if 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name:
                            continue
                        tensor = Torch.to_local_tensor(tensor)
                        # Keep original names (including .base_layer for PEFT models).
                        # The sampler side will strip .base_layer based on whether
                        # vLLM has enable_lora=True/False.
                        yield name, tensor
                    if isinstance(model, PeftModel):
                        model.unmerge_adapter()
            else:
                # ── LoRA-only mode: send only adapter weights ────────────────
                # Use PEFT's get_peft_model_state_dict for clean LoRA extraction
                from peft.utils import get_peft_model_state_dict
                lora_state_dict = get_peft_model_state_dict(model, adapter_name=adapter_name)

                def weight_generator():
                    for name, tensor in lora_state_dict.items():
                        tensor = Torch.to_local_tensor(tensor)
                        yield name, tensor

        else:
            # ── Full model mode: send all weights (base model sync) ──────
            state_dict = model.state_dict()

            def weight_generator():
                for name, tensor in state_dict.items():
                    # Skip LoRA-specific weights for base model sync
                    if 'lora_A' in name or 'lora_B' in name or 'lora_embedding' in name:
                        continue
                    tensor = Torch.to_local_tensor(tensor)
                    # Keep original names (including .base_layer for PEFT models).
                    # The sampler side will strip .base_layer based on whether
                    # vLLM has enable_lora=True/False.
                    yield name, tensor

        # Run async send_weights in a dedicated event loop thread.
        # We cannot use the Ray worker's event loop because it may already
        # be occupied, and send_weights uses run_in_executor internally.
        async def _send():
            await engine.send_weights(weight_generator())

        result_container = {'error': None}

        def _run():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(_send())
                finally:
                    loop.close()
            except Exception as e:
                result_container['error'] = e

        thread = threading.Thread(target=_run)
        thread.start()
        thread.join()

        if result_container['error'] is not None:
            raise result_container['error']
