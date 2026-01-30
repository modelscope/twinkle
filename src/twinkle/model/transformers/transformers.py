# Copyright (c) ModelScope Contributors. All rights reserved.
import contextlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Literal
from typing import overload, Type, Optional, Union

import torch
import transformers
from peft import PeftConfig
from peft import get_peft_model, PeftModel
from peft.utils import set_peft_model_state_dict, load_peft_weights
from safetensors.torch import save_file
from torch import GradScaler
from torch.optim import Optimizer, AdamW, Adam
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM
from transformers.models.auto.auto_factory import _BaseAutoModelClass

import twinkle
import twinkle.module.scheduler
from twinkle import Platform
from twinkle import remote_class, remote_function, template, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss, CrossEntropyLoss
from twinkle.metric import Metric
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils import torch_util, construct_class
from twinkle.model.base import TwinkleModel
from twinkle.model.transformers.strategy import AccelerateStrategy
from twinkle.metric import LossMetric, Accuracy, TrainMetric


@dataclass
class OptimizerGroup:
    adapter_name: str = None
    adapter_config: PeftConfig = None
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None
    inputs: Dict[str, Any] = None
    outputs: Dict[str, Any] = None
    loss_instance: Loss = CrossEntropyLoss
    loss_value: Any = None
    template: Template = None
    processor: InputProcessor = None
    scaler: GradScaler = None
    scaler_has_nan: bool = False
    gradient_accumulation_steps: int = 1
    cur_step: int = 0
    train_metrics: List[Metric] = field(default_factory=list)
    eval_metrics: List[Metric] = field(default_factory=list)
    _dp_group = None
    _device_mesh: DeviceMesh = None

    def do_grad_sync(self, gradient_accumulation_steps: Optional[int] = None) -> bool:
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        return (self.cur_step-1) % gradient_accumulation_steps == 0 and self.cur_step > 0

    def __post_init__(self):
        if self._device_mesh.data_world_size > 1:
            self._dp_group = self._device_mesh.create_process_group(['dp', 'fsdp'])
        self.train_metrics = [
            LossMetric(self._device_mesh, self._dp_group),
            Accuracy(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]

        self.eval_metrics = [
            LossMetric(self._device_mesh, self._dp_group),
            Accuracy(self._device_mesh, self._dp_group),
            TrainMetric(self._device_mesh, self._dp_group),
        ]

    def _get_lr(self):
        _lrs = []
        _default_lr = self.optimizer.defaults.get('lr')
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
DEFAULT_LEARNING_RATE = 1e-5
DEFAULT_WEIGHT_DECAY = 0.01

@remote_class()
class TransformersModel(TwinkleModel, PreTrainedModel):
    """The transformers model wrapper.

    Args:
        model_cls: The PreTrainedModel model class, only needed when creating a blank(not pretrained) model.
        config: The config of the model.
        model_id: The model id or path, this argument will be used in `from_pretrained`.
        device_mesh: The model device mesh to follow.
        mixed_precision: The mixed precision type.
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

    def __init__(self, # noqa
                 model_cls: Optional[Union[Type[PreTrainedModel], str, Type[_BaseAutoModelClass]]] = AutoModelForCausalLM,
                 model_id: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 ddp_config: Dict[str, Any] = None,
                 fsdp_config: Dict[str, Any] = None,
                 grad_scaler_config: Dict[str, Any] = None,
                 **kwargs):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        super(PreTrainedModel, self).__init__()
        if isinstance(model_cls, str):
            model_cls = getattr(transformers, model_cls)
        if model_id is None:
            self.model = model_cls.from_config(config, **kwargs)
        else:
            model_id = HubOperation.download_model(model_id)
            self.model = model_cls.from_pretrained(model_id, config=config, **kwargs)
        self.model_id = model_id
        self.tokenizer_id = kwargs.get('tokenizer_id', self.model_id)
        # The Default tokenizer will be used to save with a model if no template was set.
        self._default_tokenizer = None
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self.strategy = AccelerateStrategy(mixed_precision=mixed_precision, ddp_config=ddp_config,
                                           fsdp_config=fsdp_config, device_mesh=device_mesh)
        self.grad_scaler_config = grad_scaler_config
        self._model_wrapped = False
        self.optimizer_group: Dict[str, OptimizerGroup] = {_default_adapter_name: self._construct_default_optimizer_group()}

    @staticmethod
    def _not_encoded(inputs):
        assert isinstance(inputs, dict)
        return 'input_ids' not in inputs and 'input_embedding' not in inputs

    def _lazy_wrap_model(self):
        if not self._model_wrapped:
            assert len(self.optimizer_group) == 1
            optimizer_groups = [og for og in self.optimizer_group.values() if og.optimizer is not None]
            assert optimizer_groups == 1
            optimizer_group = optimizer_groups[0]
            optimizer = optimizer_group.optimizer
            assert optimizer 
            self.model, optimizer = self.strategy.wrap_model(self.model, optimizer)
            optimizer_group.optimizer = optimizer
            self._model_wrapped = True

    def _construct_default_optimizer_group(self):
        return OptimizerGroup(
            loss_instance=CrossEntropyLoss(),
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list) and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs) # noqa
        processor: InputProcessor = optimizer_config.processor
        assert isinstance(processor, InputProcessor), 'Set a correct `InputProcessor` before forwarding'
        inputs: Dict[str, Any] = processor(inputs)
        labels = inputs.pop('labels', None)
        self._accumulate_metric(optimizer_config, is_training=True)
        outputs = self.model(**inputs)
        inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
        if (isinstance(inputs, dict) and self._not_encoded(inputs)) or (isinstance(inputs, list) and self._not_encoded(inputs[0])):
            # Trajectory or List[Trajectory]
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            if isinstance(inputs, dict):
                inputs = [inputs]
            inputs = optimizer_config.template.batch_encode(inputs) # noqa
        with torch.no_grad():
            processor: InputProcessor = optimizer_config.processor
            assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
            inputs: Dict[str, Any] = processor(inputs)
            labels = inputs.pop('labels', None)
            self._accumulate_metric(optimizer_config, is_training=False)
            outputs = self.model(**inputs)
            inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
        return outputs

    @staticmethod
    def _accumulate_metric(optimizer_config: OptimizerGroup, is_training):
        optimizer_config.accumulate_metrics(is_training)

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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        loss_instance: Loss = optimizer_config.loss_instance
        assert isinstance(loss_instance, Loss), 'Set a loss_instance before calculating loss'
        inputs = optimizer_config.inputs
        outputs = optimizer_config.outputs
        assert inputs is not None and outputs is not None, 'Cannot calculate loss of empty inputs and outputs'
        loss_value: torch.Tensor = loss_instance(inputs, outputs, **kwargs)
        optimizer_config.loss_value = loss_value
        return loss_value.item()

    @remote_function()
    def backward(self, **kwargs):
        """Backward propagation.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                gradient_accumulation_steps: Number of gradient accumulation steps.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        loss_value = optimizer_config.loss_value
        assert loss_value is not None, 'Do forwarding and calculating loss before backward'
        scaler = optimizer_config.scaler
        _gas = optimizer_config.gradient_accumulation_steps
        if 'gradient_accumulation_steps' in kwargs:
            _gas = kwargs['gradient_accumulation_steps']
        if scaler is None and self.mixed_precision == 'fp16':
            # Auto set a grad scaler
            self.set_grad_scaler(adapter_name=adapter_name)
            scaler = optimizer_config.scaler
        loss_value = loss_value / _gas
        if scaler is not None:
            scaler.scale(loss_value).backward()
        else:
            loss_value.backward()
        optimizer_config.cur_step += 1

    @remote_function(dispatch='slice_dp', collect='mean')
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
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
        output = self.forward(inputs=inputs, **kwargs)
        loss = self.calculate_loss(**kwargs)
        output['loss'] = loss
        self.backward(**kwargs)
        return loss

    @remote_function()
    def clip_grad_norm(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        """ Clip the gradient norm

        Args:
            max_grad_norm: The maximum grad norm, default `1.0`.
            norm_type: Default `2`.
            **kwargs:
                adapter_name: Lora adapter name.
        Returns:
            Total norm of the parameter gradients (viewed as a single vector).
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer = optimizer_config.optimizer
        scaler = optimizer_config.scaler
        outputs = optimizer_config.outputs

        context = contextlib.nullcontext
        if self.device_mesh is not None and self.device_mesh.tp_world_size > 1:
            from torch.distributed.tensor.experimental import implicit_replication
            context = implicit_replication

        with context():
            if scaler is not None:
                scaler.unscale_(optimizer)

            parameters = self._get_trainable_parameters(adapter_name).values()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, norm_type=norm_type)
            # Convert DTensor to local tensor for FSDP2 compatibility
            grad_norm = torch_util.to_local_tensor(grad_norm)
            grad_norm = grad_norm.item()
            outputs['grad_norm'] = grad_norm
            return grad_norm

    @remote_function(dispatch='all')
    def clip_grad_and_step(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        grad_norm = self.clip_grad_norm(max_grad_norm, norm_type, **kwargs)
        self.step(**kwargs)
        self.zero_grad(**kwargs)
        self.lr_step(**kwargs)
        return grad_norm

    def _create_param_group(self, adapter_name: str, lr: float=DEFAULT_LEARNING_RATE, weight_decay:float=DEFAULT_WEIGHT_DECAY, **kwargs):
        # Some code borrowed from transformers

        def get_parameter_names(model, forbidden_layer_types, forbidden_layer_names=None):
            forbidden_layer_patterns = (
                [re.compile(pattern) for pattern in forbidden_layer_names] if forbidden_layer_names is not None else []
            )
            result = []
            for name, child in model.named_children():
                child_params = get_parameter_names(child, forbidden_layer_types, forbidden_layer_names)
                result += [
                    f"{name}.{n}"
                    for n in child_params
                    if not isinstance(child, tuple(forbidden_layer_types))
                       and not any(pattern.search(f"{name}.{n}".lower()) for pattern in forbidden_layer_patterns)
                ]
            # Add model specific parameters that are not in any child
            result += [
                k for k in model._parameters if
                not any(pattern.search(k.lower()) for pattern in forbidden_layer_patterns)
            ]

            return result

        forbidden_name_patterns = [r"bias", r"layernorm", r"rmsnorm", r"(?:^|\.)norm(?:$|\.)", r"_norm(?:$|\.)"]
        decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm], forbidden_name_patterns)
        params = self._get_trainable_parameters(adapter_name)
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in params.items() if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": weight_decay, 'lr': lr
            },
            {
                "params": [
                    p for n, p in params.items() if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0, 'lr': lr
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
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
                optimizer_config.scaler_has_nan = sum(v.item() for v in scaler._found_inf_per_device(optimizer).values()) > 0
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'
        optimizer.zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        """Do lr_scheduler step.

        Args:
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed for `lr_scheduler.step`.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.pop('gradient_accumulation_steps', None)):
            return
        if optimizer_config.scaler_has_nan:
            return
        lr_scheduler = optimizer_config.lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str], **kwargs):
        """Set the loss instance.

        Args:
            loss_cls: A loss class name, a loss plugin id, or a loss class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the loss instance.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if isinstance(optimizer_cls, Optimizer):
            optimizer_config.optimizer = optimizer_cls
            return

        params = kwargs.pop('params', None)
        if params is None:
            lr = kwargs.get('lr', DEFAULT_LEARNING_RATE)
            weight_decay = kwargs.get('weight_decay', DEFAULT_WEIGHT_DECAY)
            params = self._create_param_group(adapter_name, lr=lr, weight_decay=weight_decay)
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
    def set_lr_scheduler(self,
                         scheduler_cls: Union[Type[LRScheduler], str, LRScheduler],
                         **kwargs):
        """Set the lr_scheduler.

        Args:
            scheduler_cls: An lr_scheduler class name, an lr_scheduler plugin id, or an lr_scheduler class type.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the lr_scheduler instance.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before setting lr_scheduler'
        kwargs['optimizer'] = optimizer
        scheduler = construct_class(scheduler_cls, LRScheduler, [torch.optim.lr_scheduler, twinkle.module.scheduler], **kwargs)
        optimizer_config.lr_scheduler = scheduler

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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
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
            processed_state_dict[key] = torch_util.to_local_tensor(value).cpu()

        if isinstance(model, PeftModel):
            if Platform.is_master():
                model.peft_config[adapter_name].save_pretrained(checkpoint_dir)
                save_file(processed_state_dict, os.path.join(checkpoint_dir, "adapter_model.safetensors"))
        else:
            model.save_pretrained(checkpoint_dir, state_dict=processed_state_dict, is_main_process=Platform.is_master(), **save_kwargs)

        self._save_tokenizer(checkpoint_dir, adapter_name=adapter_name)
        
        if kwargs.get('save_optimizer', False):
            self._save_optimizer(checkpoint_dir, adapter_name=adapter_name)

        push_to_hub = kwargs.get('push_to_hub', False)
        hub_model_id = kwargs.get('hub_model_id', None)
        hub_token = kwargs.get('hub_token', None)
        async_upload = kwargs.get('async_upload', True)
        if push_to_hub:
            assert hub_model_id is not None and hub_token is not None
            if async_upload:
                HubOperation.async_push_to_hub(repo_id=hub_model_id, folder_path=checkpoint_dir, token=hub_token, private=True)
            else:
                HubOperation.push_to_hub(repo_id=hub_model_id, folder_path=checkpoint_dir, token=hub_token, private=True)

    def _save_optimizer(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        
        if Platform.is_master():
            optimizer = optimizer_config.optimizer
            lr_scheduler = optimizer_config.lr_scheduler
            if optimizer is not None:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            if lr_scheduler is not None:
                torch.save(lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

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
    def load(self, name: Optional[str] = None, output_dir: Optional[str] = None, **kwargs):
        """Load model state and optionally optimizer state from a checkpoint.

        Args:
            name: The name of checkpoint to save.
            output_dir: An output_dir to save the model.
            **kwargs:
                adapter_name: Adapter to load.
                load_optimizer: Whether to load optimizer and scheduler states.
        """
        load_optimizer = kwargs.get('load_optimizer', False)
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if name is None:
            name = f'checkpoint-step-{optimizer_config.cur_step}'
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        model = self.strategy.unwrap_model(self.model)
        if isinstance(model, PeftModel):
            # Load to CPU to avoid safetensors device issues in Ray environment
            adapter_weights = load_peft_weights(checkpoint_dir, device="cpu")
            set_peft_model_state_dict(model, adapter_weights, adapter_name=adapter_name)
        
        if load_optimizer:
            self._load_optimizer(checkpoint_dir, adapter_name=adapter_name)

    def _load_optimizer(self, checkpoint_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        # assume optimizer and lr_scheduler are created
        optimizer_config = self.optimizer_group[adapter_name]
        
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        
        if os.path.exists(optimizer_path) and optimizer_config.optimizer is not None:
            state_dict = torch.load(optimizer_path, map_location='cpu')
            optimizer_config.optimizer.load_state_dict(state_dict)
            
        if os.path.exists(scheduler_path) and optimizer_config.lr_scheduler is not None:
            state_dict = torch.load(scheduler_path, map_location='cpu')
            optimizer_config.lr_scheduler.load_state_dict(state_dict)

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        return self._get_trainable_parameters(kwargs.pop('adapter_name', _default_adapter_name))

    @remote_function(collect='first')
    def calculate_metric(self, is_training, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        return optimizer_config.calculate_metrics(is_training)

    def _patch_adapter(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], train_group: str, **kwargs):
        assert adapter_name, 'Use a different adapter_name, current is empty.'
        unwrapped_model = self.strategy.unwrap_model(self.model)
        if isinstance(config_or_dir, str):
            config_or_dir = HubOperation.download_model(config_or_dir)
            _adapted_model = PeftModel.from_pretrained(unwrapped_model, model_id=config_or_dir,
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

        self.optimizer_group[adapter_name] = self._construct_default_optimizer_group()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config
        _gas_default = kwargs.get('gradient_accumulation_steps', 1)
        self.optimizer_group[adapter_name].gradient_accumulation_steps = _gas_default
        self._default_tokenizer = self.optimizer_group[adapter_name].template.processor

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
        self._patch_adapter(adapter_name, config_or_dir, _default_adapter_name, **kwargs)

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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        kwargs['model_id'] = self.tokenizer_id
        template = construct_class(template_cls, Template, twinkle.template, **kwargs)
        optimizer_config.template = template

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str, InputProcessor], **kwargs):
        """Set task processor to prepare the task inputs.
        Args:
            processor_cls: A processor_cls class name, a processor_cls plugin id, or a processor_cls class type/instance.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the processor_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        from torch.amp.grad_scaler import GradScaler
        grad_scaler_config = self.grad_scaler_config.copy()
        grad_scaler_config.update(kwargs)
        optimizer_config.scaler = GradScaler(**grad_scaler_config)

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

    @remote_function(execute='first')
    def get_train_configs(self, **kwargs):
        expr = ''
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
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
                    f'Trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}\n'
                    f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
                    f'Learning rate: {optimizer_config.optimizer.defaults.get("lr", "No default lr")}\n'
                    f'Lr scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
                    f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n')
        else:
            expr += (f'Adapter config:\n'
                     f'{json.dumps(config, indent=2, ensure_ascii=False)}\n'
                     f'Trainable parameters examples:\n'
                     f'{trainable_param_names}\n'
                     f'Trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}%\n'
                     )
        return expr
