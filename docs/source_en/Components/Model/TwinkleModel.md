# TwinkleModel

TwinkleModel is the base class for all models in Twinkle. Twinkle's models not only include the model itself, but also the supporting training components of the model. The components we introduce in other documents are basically combined and used here.

Any model that conforms to the base class settings of TwinkleModel can be used with other components of the framework:

```python
class TwinkleModel(ABC):

    @abstractmethod
    def forward(self, *, inputs: Dict[str, Any], **kwargs):
        # Perform a forward pass and return logits
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def forward_only(self, *, inputs: Dict[str, Any], **kwargs):
        # Perform a forward pass in inference mode and return logits
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def calculate_loss(self, **kwargs):
        # Complete loss calculation using Loss subclass
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def backward(self, **kwargs):
        # Perform a backward pass
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def forward_backward(self, *, inputs: Dict[str, Any], **kwargs):
        # Combines forward, loss calculation, and backward process, and returns loss value
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type=2, **kwargs):
        # Gradient clipping, occurs when gradient_accumulation_steps are complete, can pass gradient_accumulation_steps in kwargs
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def step(self, **kwargs):
        # Gradient update, occurs when gradient_accumulation_steps are complete, can pass gradient_accumulation_steps in kwargs
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def zero_grad(self, **kwargs):
        # Gradient clearing, occurs when gradient_accumulation_steps are complete, can pass gradient_accumulation_steps in kwargs
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def lr_step(self, **kwargs):
        # Learning rate update, occurs when gradient_accumulation_steps are complete, can pass gradient_accumulation_steps in kwargs
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def clip_grad_and_step(self, max_grad_norm: float=1.0, norm_type=2, **kwargs):
        # Combines clip, step, zero_grad, lr_step
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def set_loss(self, loss_cls: Union[Loss, Type[Loss], str, Callable[[InputFeature, ModelOutput, ...], torch.Tensor]], **kwargs):
        # Set loss
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def set_optimizer(self, optimizer_cls: Union[Optimizer, Type[Optimizer], str], **kwargs):
        # Set optimizer
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def set_lr_scheduler(self, scheduler_cls: Union[LRScheduler, Type[LRScheduler], str], **kwargs):
        # Set lr_scheduler
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def save(self, name: str, output_dir: Optional[str] = None, **kwargs):
        # Save checkpoint
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def load(self, name: str, output_dir: Optional[str] = None, **kwargs):
        # Load checkpoint
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def get_state_dict(self, **kwargs):
        # Get state_dict
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def apply_patch(self, patch_cls: Union[Patch, Type[Patch], str], **kwargs):
        # Apply a patch to the model

    @abstractmethod
    def add_metric(self, metric_cls: Union[Metric, str], is_training, **kwargs):
        # Add a training metric, can set is_training parameter, representing accumulation in forward/forward_only. If not set, it will take effect separately for forward/forward_only
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def calculate_metric(self, is_training: bool, **kwargs):
        # Calculate metric and return
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        # Add a lora

    @abstractmethod
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        # Set template
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def set_processor(self, processor_cls: Union[InputProcessor, Type[InputProcessor], str], **kwargs):
        # Set task data processing
        # Supports adapter_name parameter to take effect on a specific lora

    @abstractmethod
    def get_train_configs(self, **kwargs) -> str:
        # Get model training configuration for printing
        # Supports adapter_name parameter to take effect on a specific lora
```
