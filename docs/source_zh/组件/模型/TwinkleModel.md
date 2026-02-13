# TwinkleModel

TwinkleModel是twinkle所有模型的基类。twinkle的模型不单单包含了模型本身，也包含了模型配套的训练组件。我们在其他文档中介绍的组件基本均在这里进行组合使用。

任何模型符合TwinkleModel的基类设定均可以配合框架的其他组件使用：

```python
class TwinkleModel(ABC):

    @abstractmethod
    def forward(self, *, inputs: Dict[str, Any], **kwargs):
        # 进行一次forward，并返回logits
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def forward_only(self, *, inputs: Dict[str, Any], **kwargs):
        # 以推理模式进行一次forward，并返回logits
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def calculate_loss(self, **kwargs):
        # 使用Loss的子类完成loss计算
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def backward(self, **kwargs):
        # 进行一次backward
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def forward_backward(self, *, inputs: Dict[str, Any], **kwargs):
        # 组合了forward、loss计算、backward过程，并返回loss值
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        # 梯度裁剪，发生在gradient_accumulation_steps完成的条件下，可以在kwargs传入gradient_accumulation_steps
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def step(self, **kwargs):
        # 梯度更新，发生在gradient_accumulation_steps完成的条件下，可以在kwargs传入gradient_accumulation_steps
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def zero_grad(self, **kwargs):
        # 梯度清理，发生在gradient_accumulation_steps完成的条件下，可以在kwargs传入gradient_accumulation_steps
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def lr_step(self, **kwargs):
        # lr更新，发生在gradient_accumulation_steps完成的条件下，可以在kwargs传入gradient_accumulation_steps
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def clip_grad_and_step(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        # 组合了clip、step、zero_grad、lr_step
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str, Callable[[InputFeature, ModelOutput, ...], torch.Tensor]], **kwargs):
        # 设置loss
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str], **kwargs):
        # 设置optimizer
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str], **kwargs):
        # 设置lr_scheduler
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def save(self, name: str, output_dir: Optional[str] = None, **kwargs):
        # 保存checkpoint
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        # 加载checkpoint
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def get_state_dict(self, **kwargs):
        # 获取state_dict
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs):
        # 对模型应用一个补丁

    @abstractmethod
    def add_metric(self, metric_cls: Union[Metric, str], is_training, **kwargs):
        # 增加一个训练指标，可以设置is_training参数，代表在forward/forward_only中累加。如果不设置，则对forward/forward_only分别生效
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def calculate_metric(self, is_training: bool, **kwargs):
        # 计算metric并返回
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        # 增加一个lora

    @abstractmethod
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        # 设置template
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs):
        # 设置任务数据处理
        # 支持adapter_name参数，对某个lora生效

    @abstractmethod
    def get_train_configs(self, **kwargs) -> str:
        # 获取模型训练配置，用于打印
        # 支持adapter_name参数，对某个lora生效
```
