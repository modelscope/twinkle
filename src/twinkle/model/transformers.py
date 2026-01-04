import contextlib
import re
from dataclasses import dataclass
from typing import Dict, Any, List, Literal
from typing import overload, Type, Optional, Union
from transformers.models.auto.auto_factory import _BaseAutoModelClass
from peft import PeftConfig, LoraConfig
from peft import get_peft_model, PeftModel
import torch
from torch import GradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM
import transformers
import twinkle
from twinkle import remote_class, remote_function, template, DeviceMesh
from twinkle.loss import Loss
from twinkle.hub import HubOperation
from .base import TwinkleModel
from twinkle.processor import InputProcessor
from twinkle.template import Template
from twinkle.utils.plugin import Plugin
from .strategy import AccelerateStrategy
from twinkle.data_format import InputFeature, Trajectory


@dataclass
class OptimizerGroup:
    adapter_name: str = None
    adapter_config: PeftConfig = None
    optimizer: Optimizer = None
    lr_scheduler: LRScheduler = None
    inputs: Dict[str, Any] = None
    outputs: Dict[str, Any] = None
    loss_instance: Loss = None
    loss_value: Any = None
    template: Template = None
    processor: InputProcessor = None
    scaler: GradScaler = None
    scaler_has_nan: bool = False


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

    _default_adapter_name = ''

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
        from twinkle.patch.multi_adapter import MultiAdapter
        config = LoraConfig(r=1, target_modules='all-linear')
        # pre wrap
        self.model = get_peft_model(self.model, config, adapter_name='__dummy_adapter_inner__')
        self.model: PreTrainedModel = MultiAdapter()(self.model) # patch multiple loras
        self.mixed_precision = mixed_precision
        self.strategy = AccelerateStrategy(mixed_precision=mixed_precision, ddp_config=ddp_config,
                                           fsdp_config=fsdp_config)
        self.grad_scaler_config = grad_scaler_config
        self.optimizer_group: Dict[str, OptimizerGroup] = {self._default_adapter_name: OptimizerGroup()}
        self.model = self.strategy.wrap_model(self.model)

    def _check_adapter_valid(self, adapter_name: str):
        assert adapter_name in self.optimizer_group, f'Use a valid {adapter_name} first, current is: {adapter_name}'

    @staticmethod
    def _not_encoded(inputs):
        assert isinstance(inputs, dict)
        return not 'input_ids' not in inputs and 'input_embedding' not in inputs

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
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)

        if isinstance(inputs, dict) and self._not_encoded(inputs):
            assert self.optimizer_group[adapter_name].template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = self.optimizer_group[adapter_name].template.encode(inputs)
        if isinstance(inputs, list) and self._not_encoded(inputs[0]):
            assert self.optimizer_group[adapter_name].template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = self.optimizer_group[adapter_name].template.batch_encode(inputs)
        processor: InputProcessor = self.optimizer_group[adapter_name].processor
        assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
        inputs: Dict[str, Any] = processor(inputs)
        if adapter_name:
            self.strategy.unwrap_model(self.model).set_current_adapter_name(adapter_name)
        outputs = self.model(**inputs)
        self.optimizer_group[adapter_name].inputs = inputs
        self.optimizer_group[adapter_name].outputs = outputs
        return outputs

    @remote_function()
    def forward_only(self, *, inputs: Union[InputFeature, List[InputFeature], List[Trajectory]], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if isinstance(inputs, dict) and self._not_encoded(inputs):
            assert self.optimizer_group[adapter_name].template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = self.optimizer_group[adapter_name].template.encode(inputs)
        if isinstance(inputs, list) and self._not_encoded(inputs[0]):
            assert self.optimizer_group[adapter_name].template is not None, \
                'Use set_template to add a template when trying to input `List[Trajectory]`'
            inputs = self.optimizer_group[adapter_name].template.batch_encode(inputs)
        import torch
        with torch.no_grad():
            processor: InputProcessor = self.optimizer_group[adapter_name].processor
            assert isinstance(processor, InputProcessor), 'Set InputProcessor correctly before forwarding'
            inputs: Dict[str, Any] = processor(inputs)
            if adapter_name:
                self.strategy.unwrap_model(self.model).set_current_adapter_name(adapter_name)
            outputs = self.model(**inputs)
        self.optimizer_group[adapter_name].inputs = inputs
        self.optimizer_group[adapter_name].outputs = outputs
        return outputs

    @remote_function()
    def calculate_loss(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        loss_instance = self.optimizer_group[adapter_name].loss_instance
        assert isinstance(loss_instance, Loss), 'Set loss_instance correctly before calculating loss'
        inputs = self.optimizer_group[adapter_name].inputs
        outputs = self.optimizer_group[adapter_name].outputs
        assert inputs is not None and outputs is not None, 'Cannot calculate loss of empty inputs and outputs'
        loss_value = loss_instance(inputs, outputs, **kwargs)
        self.optimizer_group[adapter_name].loss_value = loss_value
        return loss_value

    @remote_function()
    def backward(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        loss_value = self.optimizer_group[adapter_name].loss_value
        assert loss_value is not None, 'Do forwarding and calculating loss before backward'
        if adapter_name:
            self.model.set_current_adapter_name(adapter_name)

        scaler = self.optimizer_group[adapter_name].scaler
        if scaler is None and self.mixed_precision == 'fp16':
            self.set_grad_scaler(adapter_name=adapter_name)
            scaler = self.optimizer_group[adapter_name].scaler
        loss_value = loss_value / self.gradient_accumulation_steps
        if scaler is not None:
            scaler.scale(loss_value).backward(**kwargs)
        else:
            loss_value.backward(**kwargs)

        if adapter_name:
            import torch.distributed as dist
            for p in self._get_trainable_parameters(adapter_name):
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG, group=self.device_mesh.ddp_group)

    @remote_function()
    def forward_backward(self, *, inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]], **kwargs):
        self.forward(inputs=inputs, **kwargs)
        self.calculate_loss(**kwargs)
        self.backward(**kwargs)

    def clip_grad_norm(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        optimizer = self.optimizer_group[adapter_name].optimizer
        scaler = self.optimizer_group[adapter_name].scaler

        context = contextlib.nullcontext
        if self.device_mesh is not None and self.device_mesh.tp_world_size > 1:
            from torch.distributed.tensor.experimental import implicit_replication
            context = implicit_replication

        with context():
            if scaler is not None:
                scaler.unscale_(optimizer)

            parameters = self._get_trainable_parameters(adapter_name=adapter_name)
            return torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, norm_type=norm_type)

    @remote_function()
    def step(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        optimizer = self.optimizer_group[adapter_name].optimizer
        scaler = self.optimizer_group[adapter_name].scaler
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'

        context = contextlib.nullcontext
        if self.device_mesh is not None and self.device_mesh.tp_world_size > 1:
            from torch.distributed.tensor.experimental import implicit_replication
            context = implicit_replication

        with context():
            if scaler is not None:
                scaler.step(optimizer, **kwargs)
                scaler.update()
                self.optimizer_group[adapter_name].scaler_has_nan = sum(v.item() for v in scaler._found_inf_per_device(optimizer).values()) > 0
            else:
                optimizer.step(**kwargs)

    @remote_function()
    def zero_grad(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        optimizer = self.optimizer_group[adapter_name].optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before forwarding'
        optimizer.zero_grad(**kwargs)

    @remote_function()
    def lr_step(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if self.optimizer_group[adapter_name].scaler_has_nan:
            return
        lr_scheduler = self.optimizer_group[adapter_name].lr_scheduler
        assert isinstance(lr_scheduler, LRScheduler), 'Set lr_scheduler correctly before forwarding'
        lr_scheduler.step(**kwargs)

    @remote_function()
    def set_loss(self, loss_cls: Union[Type[Loss], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if isinstance(loss_cls, str):
            if hasattr(twinkle.loss, loss_cls):
                loss_cls = getattr(twinkle.loss, loss_cls)
            else:
                loss_cls = Plugin.load_plugin(loss_cls, Loss)
        self.optimizer_group[adapter_name].loss_instance = loss_cls()

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if isinstance(optimizer_cls, str):
            import torch
            if hasattr(torch.optim, optimizer_cls):
                optimizer_cls = getattr(torch.optim, optimizer_cls)
            else:
                optimizer_cls = Plugin.load_plugin(optimizer_cls, Optimizer)
        self.optimizer_group[adapter_name].optimizer = optimizer_cls(self._get_trainable_parameters(
            adapter_name=adapter_name).values(), **kwargs)

    def _get_trainable_parameters(self, adapter_name=''):
        self._check_adapter_valid(adapter_name)
        is_default = adapter_name == self._default_adapter_name
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        params = {}
        model = self.strategy.unwrap_model(self.model)
        for name, param in model.named_parameters():
            if param.requires_grad and (pattern.search(name) or is_default):
                params[name] = param
        return params

    def _activate_adapter_grad(self, adapter_name: str):
        self._check_adapter_valid(adapter_name)
        pattern = re.compile(rf'\.lora_\w+\.{re.escape(adapter_name)}\.')
        model = self.strategy.unwrap_model(self.model)
        for name, param in model.named_parameters():
            if pattern.search(name):
                param.requires_grad = True

    @remote_function()
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if isinstance(scheduler_cls, str):
            import torch
            if hasattr(torch.optim.lr_scheduler, scheduler_cls):
                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_cls)
            else:
                scheduler_cls = Plugin.load_plugin(scheduler_cls, LRScheduler)
        optimizer = self.optimizer_group[adapter_name].optimizer
        assert isinstance(optimizer, Optimizer), 'Set optimizer correctly before setting lr_scheduler'
        self.optimizer_group[adapter_name].lr_scheduler = scheduler_cls(optimizer, **kwargs)

    @remote_function(execute='first')
    def save(self, output_dir, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if not adapter_name:
            self.model.save_pretrained(output_dir)
        else:
            state_dict = self.get_state_dict(adapter_name)
            self.model.save_pretrained(output_dir, state_dict=state_dict)
        self._save_tokenizer(output_dir, adapter_name=adapter_name)

    def _save_tokenizer(self, output_dir, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        template_ins = self.optimizer_group[adapter_name].template
        template_ins.tokenizer.save_pretrained(output_dir)

    @remote_function(execute='first')
    def get_state_dict(self, adapter_name: str = ''):
        return self._get_trainable_parameters(adapter_name=adapter_name)

    @remote_function()
    def add_adapter_to_model(self, adapter_name: str, config: PeftConfig):
        assert adapter_name not in self.optimizer_group, f'{adapter_name} already exists.'
        assert adapter_name, 'Use a different adapter_name, current is empty.'
        self.optimizer_group[adapter_name] = OptimizerGroup()
        self.optimizer_group[adapter_name].adapter_name = adapter_name
        self.optimizer_group[adapter_name].adapter_config = config
        self.strategy.unwrap_model(self.model).add_adapter(adapter_name, config)
        self._activate_adapter_grad(adapter_name)

    @remote_function()
    def set_template(self, template_cls: Union[Type[template.Template], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if isinstance(template_cls, str):
            if hasattr(template, template_cls):
                template_cls = getattr(template, template_cls)
            else:
                template_cls = Plugin.load_plugin(template_cls, template.Template)
        self.optimizer_group[adapter_name].template = template_cls(self.model_id, **kwargs)

    @remote_function()
    def set_processor(self, processor_cls: Union[Type[InputProcessor], str], **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        if isinstance(processor_cls, str):
            if hasattr(__file__.__module__, processor_cls):
                processor_cls: Type[InputProcessor] = getattr(__file__.__module__, processor_cls)
            else:
                processor_cls: Type[InputProcessor] = Plugin.load_plugin(processor_cls, InputProcessor)
        self.optimizer_group[adapter_name].processor = processor_cls(device_mesh=self.device_mesh, **kwargs)

    @remote_function()
    def set_grad_scaler(self, **kwargs):
        adapter_name = kwargs.pop("adapter_name", None) or ''
        self._check_adapter_valid(adapter_name)
        from torch.amp.grad_scaler import GradScaler
        grad_scaler_config = self.grad_scaler_config.copy()
        grad_scaler_config.update(kwargs)
        self.optimizer_group[adapter_name].scaler = GradScaler(**grad_scaler_config)

    def remove_adapter(self, adapter_name: str):
        if adapter_name in self.optimizer_group:
            self.optimizer_group.pop(adapter_name)
        if isinstance(self.model, PeftModel):
            self.model.delete_adapter(adapter_name)
