import contextlib
import json
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Literal
from typing import overload, Type, Optional, Union

import torch
import transformers
from peft import PeftConfig
from peft import get_peft_model, PeftModel
from torch import GradScaler
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM
from transformers.models.auto.auto_factory import _BaseAutoModelClass

import twinkle
from twinkle import remote_class, remote_function, template, DeviceMesh
from twinkle.data_format import InputFeature, Trajectory
from twinkle.hub import HubOperation
from twinkle.loss import Loss, CrossEntropyLoss
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils import torch_util
from twinkle.utils.plugin import Plugin
from .base import TwinkleModel
from .strategy import AccelerateStrategy


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
    dp_group = None

    def do_grad_sync(self, gradient_accumulation_steps: Optional[int] = None) -> bool:
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = self.gradient_accumulation_steps
        return self.cur_step % gradient_accumulation_steps == 0 and self.cur_step > 0


_default_adapter_name = ''


@remote_class()
class TransformersModel(TwinkleModel, PreTrainedModel):
    """The transformers model wrapper.

    Args:
        model_cls: The PreTrainedModel model class, only needed when creating a blank(not pretrained) model.
        config: The config of the model.
        pretrained_model_name_or_path: The model id or path, this argument will be used in `from_pretrained`.
        kwargs: Any kwargs used in `from_pretrained` or `__init__`.

    If pretrained_model_name_or_path is passed in, `from_pretrained` will be used, else `__init__` will be used.
    """

    @overload
    def __init__(self, *, model_cls: Type[PreTrainedModel], config: PretrainedConfig, remote_group, **kwargs) -> None:
        ...

    @overload
    def __init__(self, *, pretrained_model_name_or_path: str, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        ...

    def __init__(self, # noqa
                 model_cls: Optional[Union[Type[PreTrainedModel], str, Type[_BaseAutoModelClass]]] = AutoModelForCausalLM,
                 pretrained_model_name_or_path: Optional[str] = None,
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
        if pretrained_model_name_or_path is None:
            self.model = model_cls(config, **kwargs)
        else:
            pretrained_model_name_or_path = HubOperation.download_model(pretrained_model_name_or_path)
            self.model = model_cls.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        self.model_id = pretrained_model_name_or_path
        self.device_mesh = device_mesh
        self.mixed_precision = mixed_precision
        self.strategy = AccelerateStrategy(mixed_precision=mixed_precision, ddp_config=ddp_config,
                                           fsdp_config=fsdp_config, device_mesh=device_mesh)
        self.grad_scaler_config = grad_scaler_config
        self._model_wrapped = False
        self.optimizer_group: Dict[str, OptimizerGroup] = {_default_adapter_name: OptimizerGroup()}

    @staticmethod
    def _not_encoded(inputs):
        assert isinstance(inputs, dict)
        return not 'input_ids' not in inputs and 'input_embedding' not in inputs

    def _lazy_wrap_model(self):
        if not self._model_wrapped:
            assert len(self.optimizer_group) == 1
            optimizer = self.optimizer_group[_default_adapter_name].optimizer
            assert isinstance(optimizer, Optimizer)
            self.model, optimizer = self.strategy.wrap_model(self.model, optimizer)
            self.optimizer_group[_default_adapter_name].optimizer = optimizer
            self._model_wrapped = True

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
        outputs = self.model(**inputs)
        inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
        return outputs

    @remote_function()
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
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
        import torch
        with torch.no_grad():
            processor: InputProcessor = optimizer_config.processor
            assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
            inputs: Dict[str, Any] = processor(inputs)
            labels = inputs.pop('labels', None)
            outputs = self.model(**inputs)
            inputs['labels'] = labels
        optimizer_config.inputs = inputs
        optimizer_config.outputs = outputs
        return outputs

    @remote_function(collect='avg')
    def calculate_loss(self, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        loss_instance: Loss = optimizer_config.loss_instance
        assert isinstance(loss_instance, Loss), 'Set loss_instance correctly before calculating loss'
        inputs = optimizer_config.inputs
        outputs = optimizer_config.outputs
        assert inputs is not None and outputs is not None, 'Cannot calculate loss of empty inputs and outputs'
        loss_value = loss_instance(inputs, outputs, **kwargs)
        optimizer_config.loss_value = loss_value
        return loss_value.detach().cpu().float().numpy()

    @remote_function()
    def backward(self, **kwargs):
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

    @remote_function(collect='avg')
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        output = self.forward(inputs=inputs, **kwargs)
        loss = self.calculate_loss(**kwargs)
        output['loss'] = loss
        self.backward(**kwargs)
        return loss

    @remote_function()
    def clip_grad_norm(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
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

    @remote_function()
    def step(self, **kwargs):
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

        with context():
            if scaler is not None:
                scaler.step(optimizer, **kwargs)
                scaler.update()
                optimizer_config.scaler_has_nan = sum(v.item() for v in scaler._found_inf_per_device(optimizer).values()) > 0
            else:
                optimizer.step(**kwargs)

    @remote_function()
    def zero_grad(self, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'
        optimizer.zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if not optimizer_config.do_grad_sync(kwargs.get('gradient_accumulation_steps')):
            return
        if optimizer_config.scaler_has_nan:
            return
        lr_scheduler = optimizer_config.lr_scheduler
        assert isinstance(lr_scheduler, LRScheduler), 'Set lr_scheduler correctly before forwarding'
        lr_scheduler.step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if isinstance(optimizer_cls, str):
            import torch
            if hasattr(torch.optim, optimizer_cls):
                optimizer_cls = getattr(torch.optim, optimizer_cls)
            else:
                optimizer_cls = Plugin.load_plugin(optimizer_cls, Optimizer)
        optimizer_config.optimizer = optimizer_cls(self._get_trainable_parameters(adapter_name).values(), **kwargs)

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
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str], **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        if isinstance(scheduler_cls, str):
            import torch
            if hasattr(torch.optim.lr_scheduler, scheduler_cls):
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cls)
            else:
                scheduler_cls = Plugin.load_plugin(scheduler_cls, LRScheduler)
        optimizer = optimizer_config.optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before setting lr_scheduler'
        optimizer_config.lr_scheduler = scheduler_cls(optimizer, **kwargs)

    @remote_function()
    def save(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        model = self.strategy.unwrap_model(self.model)
        state_dict = self._get_trainable_parameters(adapter_name=adapter_name)
        processed_state_dict = {}

        for key, value in state_dict.items():
            if isinstance(value, DTensor):
                processed_state_dict[key] = value.full_tensor().cpu()
            else:
                processed_state_dict[key] = value.cpu()

        model.save_pretrained(output_dir, state_dict=processed_state_dict)
        self._save_tokenizer(output_dir)

    def _save_tokenizer(self, output_dir, **kwargs):
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        template_ins = optimizer_config.template
        if template_ins is not None:
            template_ins.tokenizer.save_pretrained(output_dir)

    @remote_function(execute='first')
    def get_state_dict(self, **kwargs):
        return self._get_trainable_parameters(kwargs.pop('adapter_name', _default_adapter_name))

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

        self.optimizer_group[train_group] = OptimizerGroup()
        self.optimizer_group[train_group].adapter_name = adapter_name
        self.optimizer_group[train_group].adapter_config = config
        _gas_default = kwargs.get('gradient_accumulation_steps', 1)
        self.optimizer_group[train_group].gradient_accumulation_steps = _gas_default
        default_config = self.optimizer_group[_default_adapter_name]
        if default_config.template:
            self.optimizer_group[train_group].template = default_config.template
        if default_config.processor:
            self.optimizer_group[train_group].processor = default_config.processor

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config_or_dir: Union[PeftConfig, str], **kwargs):
        self._patch_adapter(adapter_name, config_or_dir, _default_adapter_name, **kwargs)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
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
        adapter_name = kwargs.pop('adapter_name', _default_adapter_name)
        optimizer_config = self.optimizer_group[adapter_name]
        from torch.amp.grad_scaler import GradScaler
        grad_scaler_config = self.grad_scaler_config.copy()
        grad_scaler_config.update(kwargs)
        optimizer_config.scaler = GradScaler(**grad_scaler_config)

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
        expr += (f'Adapter config:\n'
                 f'{json.dumps(config, indent=2, ensure_ascii=False)}\n'
                 f'Optimizer: {optimizer_config.optimizer.__class__.__name__}\n'
                 f'Learning rate: {optimizer_config.optimizer.defaults.get("lr", "No default lr")}\n'
                 f'Lr scheduler: {optimizer_config.lr_scheduler.__class__.__name__}\n'
                 f'Gradient accumulation steps: {optimizer_config.gradient_accumulation_steps}\n')
        return expr
