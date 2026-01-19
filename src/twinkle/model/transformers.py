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
from torch import GradScaler
from torch.optim import Optimizer, AdamW, Adam
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.auto.auto_factory import _BaseAutoModelClass

import twinkle
from twinkle import remote_class, remote_function, template, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss, CrossEntropyLoss
from twinkle.module import scheduler
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils import torch_util
from twinkle.utils.plugin import Plugin
from .base import TwinkleModel
from .strategy import AccelerateStrategy
from twinkle.metric import Metric
from ..metric.accuracy import Accuracy
from ..metric.loss import LossMetric


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
    metrics: List[Metric] = field(default_factory=list)
    _dp_group = None
    _device_mesh: DeviceMesh = None

    def do_grad_sync(self, gradient_accumulation_steps: Optional[int] = None) -> bool:
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        return self.cur_step % gradient_accumulation_steps == 0 and self.cur_step > 0

    def __post_init__(self):
        if self._device_mesh.data_parallel_world_size > 1:
            self._dp_group = self._device_mesh.create_process_group(['dp', 'fsdp'])
            for metric in self.metrics:
                metric.process_group = self._dp_group


_default_adapter_name = ''


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
        super(PreTrainedModel, self).__init__()
        if isinstance(model_cls, str):
            model_cls = getattr(transformers, model_cls)
        if model_id is None:
            self.model = model_cls(config, **kwargs)
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
            optimizer = self.optimizer_group[_default_adapter_name].optimizer
            if optimizer is None:
                optimizer = AdamW(self._create_param_group(_default_adapter_name, 1e-5, 0.01), lr=1e-5)
                self.optimizer_group[_default_adapter_name].optimizer = optimizer
            self.model, optimizer = self.strategy.wrap_model(self.model, optimizer)
            self.optimizer_group[_default_adapter_name].optimizer = optimizer
            self._model_wrapped = True

    def _construct_default_optimizer_group(self):
        return OptimizerGroup(
            loss_instance=CrossEntropyLoss(),
            template=Template(self.tokenizer_id),
            processor=InputProcessor(self.device_mesh),
            _device_mesh=self.device_mesh,
        )

    @remote_function()
    def forward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        """Call forward function and record the inputs and outputs.

        Args:
            inputs: The model inputs. Can be an encoded batch, or a list of `Trajectory`
            **kwargs:
                adapter_name: Lora adapter name.
        Returns:
            The output of the model forward.
        """
        # breakpoint()
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        self._lazy_wrap_model()
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
        inputs: Dict[str, Any] = processor(inputs)
        labels = inputs.pop('labels', None)
        self._accumulate_metric(optimizer_config)
        outputs = self.model(**inputs)
        inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
        return outputs

    @remote_function()
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
        if isinstance(inputs, dict) and self._not_encoded(inputs):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = optimizer_config.template.encode(inputs) # noqa
        if isinstance(inputs, list) and self._not_encoded(inputs[0]):
            assert optimizer_config.template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = optimizer_config.template.batch_encode(inputs) # noqa
        with torch.no_grad():
            processor: InputProcessor = optimizer_config.processor
            assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
            inputs: Dict[str, Any] = processor(inputs)
            labels = inputs.pop('labels', None)
            self._accumulate_metric(optimizer_config)
            outputs = self.model(**inputs)
            inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
        return outputs

    @staticmethod
    def _accumulate_metric(optimizer_config: OptimizerGroup):
        if len(optimizer_config.metrics) > 0 and optimizer_config.inputs is not None and optimizer_config.outputs is not None:
            for metric in optimizer_config.metrics:
                metric.accumulate(optimizer_config.inputs, optimizer_config.outputs)

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
        assert isinstance(loss_instance, Loss), 'Set loss_instance correctly before calculating loss'
        inputs = optimizer_config.inputs
        outputs = optimizer_config.outputs
        assert inputs is not None and outputs is not None, 'Cannot calculate loss of empty inputs and outputs'
        loss_value: torch.Tensor = loss_instance(inputs, outputs, **kwargs)
        optimizer_config.loss_value = loss_value
        return loss_value.detach().cpu().float().numpy()

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
            self.set_grad_scaler()
            scaler = optimizer_config.scaler
        loss_value = loss_value / _gas
        if scaler is not None:
            scaler.scale(loss_value).backward()
        else:
            loss_value.backward()
        optimizer_config.cur_step += 1

    @remote_function(collect='mean')
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
            return grad_norm.detach().cpu().numpy()

    def clip_grad_and_step(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        self.clip_grad_norm(max_grad_norm, norm_type, **kwargs)
        self.step(**kwargs)
        self.zero_grad(**kwargs)
        self.lr_step(**kwargs)

    def _create_param_group(self, adapter_name: str, lr: float=1e-5, weight_decay:float=0.01, **kwargs):
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
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
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
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return
        if optimizer_config.scaler_has_nan:
            return
        lr_scheduler = optimizer_config.lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler.step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        """Set the loss instance.

        Args:
            loss_cls: A loss class name, a loss plugin id, or a loss class type.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the loss instance.
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
        """Set the optimizer.

        Args:
            optimizer_cls: An optimizer class name, an optimizer plugin id, or an optimizer class type.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the optimizer instance.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if isinstance(optimizer_cls, str):
            import torch
            if hasattr(torch.optim, optimizer_cls):
                optimizer_cls = getattr(torch.optim, optimizer_cls)
            else:
                optimizer_cls = Plugin.load_plugin(optimizer_cls, Optimizer)
        optimizer_config.optimizer = optimizer_cls(self._create_param_group(adapter_name, **kwargs), **kwargs)

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
                         scheduler_cls: Union[Type[LRScheduler], str],
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
        if isinstance(scheduler_cls, str):
            import torch
            if hasattr(torch.optim.lr_scheduler, scheduler_cls):
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cls)
            if hasattr(scheduler, scheduler_cls):
                scheduler_cls = getattr(scheduler, scheduler_cls)
            else:
                scheduler_cls = Plugin.load_plugin(scheduler_cls, LRScheduler)
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before setting lr_scheduler'
        optimizer_config.lr_scheduler = scheduler_cls(optimizer, **kwargs)

    def __del__(self):
        HubOperation.wait_for()

    @remote_function()
    def save(self, name, output_dir=None, **kwargs):
        """Save model.

        Args:
            name: The name of checkpoint to save.
            output_dir: An output_dir to save the model.
            **kwargs:
                adapter_name: Lora adapter name.
        """
        if output_dir is None:
            output_dir = 'output'
        checkpoint_dir = os.path.join(output_dir, name)
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        model = self.strategy.unwrap_model(self.model)
        state_dict = self._get_trainable_parameters(adapter_name=adapter_name)
        processed_state_dict = {}

        for key, value in state_dict.items():
            processed_state_dict[key] = torch_util.to_local_tensor(value).cpu()

        model.save_pretrained(checkpoint_dir, state_dict=processed_state_dict)
        self._save_tokenizer(checkpoint_dir)
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

    def _save_tokenizer(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        template_ins = optimizer_config.template
        if template_ins is not None:
            template_ins.tokenizer.save_pretrained(output_dir)
        else:
            self._default_tokenizer.save_pretrained(output_dir)

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        return self._get_trainable_parameters(kwargs.pop('adapter_name', _default_adapter_name))

    @remote_function(collect='first')
    def calculate_metric(self, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        results = {}
        for metric in optimizer_config.metrics[:1]:
            metric.accumulate(optimizer_config.inputs, optimizer_config.outputs)
            results.update(metric.calculate())
        return results

    def _patch_adapter(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], train_group: str, **kwargs):
        assert adapter_name, 'Use a different adapter_name, current is empty.'
        unwrapped_model = self.strategy.unwrap_model(self.model)
        config = None
        if isinstance(config_or_dir, str):
            config_or_dir = HubOperation.download_model(config_or_dir)
            _adapted_model = PeftModel.from_pretrained(unwrapped_model, model_id=config_or_dir,
                                                        adapter_name=adapter_name,
                                                        is_trainable=kwargs.get('is_trainable', True))
            if unwrapped_model is self.model:
                self.model = _adapted_model
            else:
                assert isinstance(unwrapped_model, PeftModel)
        else:
            config = config_or_dir
            if not isinstance(unwrapped_model, PeftModel):
                assert unwrapped_model is self.model, 'Cannot wrap model with peft after DDP/FSDP!'
                self.model = get_peft_model(unwrapped_model, config, adapter_name=adapter_name)
            else:
                unwrapped_model.add_adapter(adapter_name, config)

        self.optimizer_group[train_group] = self._construct_default_optimizer_group()
        self._default_tokenizer = self.optimizer_group[train_group].template.tokenizer
        self.optimizer_group[train_group].adapter_name = adapter_name
        self.optimizer_group[train_group].adapter_config = config
        _gas_default = kwargs.get('gradient_accumulation_steps', 1)
        self.optimizer_group[train_group].gradient_accumulation_steps = _gas_default
        default_config = self.optimizer_group[_default_adapter_name]
        if default_config.template:
            self.optimizer_group[train_group].template = default_config.template
        if default_config.processor:
            self.optimizer_group[train_group].processor = default_config.processor
        dp_group = self.optimizer_group[train_group]._dp_group
        self.optimizer_group[train_group].metrics = [
                LossMetric(self.device_mesh, dp_group),
                Accuracy(self.device_mesh, dp_group),
            ]

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
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        """Set template. This is optional, if you need to input `Trajectory`,
            you need to set the template to encode them.

        Args:
            template_cls: A template_cls class name, a template_cls plugin id, or a template_cls class type.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the template_cls instance.
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
        """Set task processor to prepare the task inputs.
        Args:
            processor_cls: A processor_cls class name, a processor_cls plugin id, or a processor_cls class type.
            **kwargs:
                adapter_name: Lora adapter name.
                Any parameters needed to construct the processor_cls instance.
        """
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if isinstance(processor_cls, str):
            if hasattr(twinkle.processor, processor_cls):
                processor_cls: Type[InputProcessor] = getattr(twinkle.processor, processor_cls)
            else:
                processor_cls: Type[InputProcessor] = Plugin.load_plugin(processor_cls, InputProcessor)
        optimizer_config.processor = processor_cls(device_mesh=self.device_mesh, **kwargs)

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
        if isinstance(metric_cls, str):
            if hasattr(twinkle.metric, metric_cls):
                metric_cls: Type[Metric] = getattr(twinkle.metric, metric_cls)
            else:
                metric_cls: Type[Metric] = Plugin.load_plugin(metric_cls, Metric)
            optimizer_config.metrics.append(metric_cls(self.device_mesh, optimizer_config._dp_group, **kwargs))

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
        if optimizer_config.optimizer is not None:
            expr += (f'Adapter config:\n'
                    f'{json.dumps(config, indent=2, ensure_ascii=False)}\n'
                    f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
                    f'Learning rate: {optimizer_config.optimizer.defaults.get("lr", "No default lr")}\n'
                    f'Lr scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
                    f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n')
        else:
            expr += (f'Adapter config:\n'
                    f'{json.dumps(config, indent=2, ensure_ascii=False)}\n')
        return expr
