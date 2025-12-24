from typing import overload, Type, Optional, Union, Callable

from peft import PeftConfig, get_peft_model, PeftModel
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import PreTrainedModel, PretrainedConfig

from twinkle import remote_class, remote_function


@remote_class()
class TransformersModel(PreTrainedModel):

    @overload
    def __init__(self, *, model_cls: Type[PreTrainedModel], config: PretrainedConfig, remote_group, **kwargs) -> None:
        ...

    @overload
    def __init__(self, *, pretrained_model_name_or_path: str, config: Optional[PretrainedConfig] = None, **kwargs) -> None:
        ...

    def __init__(self, # noqa
                 model_cls: Optional[Type[PreTrainedModel]] = None,
                 pretrained_model_name_or_path: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 **kwargs):
        if pretrained_model_name_or_path is None:
            self.model = model_cls(config, **kwargs)
        elif model_cls :
            self.model = model_cls.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        self.loss_instance = None
        self.loss_value = None
        self.input = None
        self.output = None
        self.optimizer = None
        self.lr_scheduler = None

    @remote_function()
    def forward(self, *, input_ids, adapter_name: str, **kwargs):
        kwargs['input_ids'] = input_ids
        self.input = kwargs
        output = self.model(**kwargs)
        self.output = output
        return output

    @remote_function()
    def set_loss(self, loss: Union[Type, str, Callable]):
        self.loss_instance = loss()

    @remote_function()
    def loss(self, *, loss_func: Callable = None, **kwargs):
        loss_instance = loss_func or self.loss_instance
        self.loss_value = loss_instance(self.input, self.output, **kwargs)
        return self.loss_value

    @remote_function()
    def backward(self):
        self.loss_value.backward()

    @remote_function()
    def forward_backward(self, *, input_ids, adapter_name: str, **kwargs):
        self.forward(input_ids=input_ids, **kwargs)
        self.loss(**kwargs)
        self.backward()

    @remote_function()
    def step(self):
        self.optimizer.step()

    @remote_function()
    def zero_grad(self):
        self.optimizer.zero_grad()

    @remote_function()
    def lr_step(self):
        self.lr_scheduler.step()

    @remote_function()
    def set_optimizer(self, optimizer_cls: Union[Type[Optimizer], str]):
        self.optimizer = optimizer_cls()

    @remote_function()
    def set_lr_scheduler(self, scheduler_cls: Union[Type[LRScheduler], str]):
        self.lr_scheduler = scheduler_cls()

    def save_state(self):
        pass

    def add_adapter_to_model(self, adapter_name: str, config: Union[PeftConfig, Callable]):
        if isinstance(config, PeftConfig):
            if isinstance(self.model, PeftModel):
                self.model = self.model.add_adapter(adapter_name, config)
            else:
                self.model = get_peft_model(self.model, config)
        else:
            self.model = config(self.model)

    def add_loss_scale(self, loss_scale):
