# TransformersModel

This model encapsulates the transformers LLM and can start and train models using FSDP2, DDP and other methods.

```python
class TransformersModel:
 
    def __init__(self, # noqa
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
        ...

    ...
```

- model_cls: Which class to use to start the model, default is `AutoModelForCausalLM`
- model_id: Model id
- config: Configuration for starting the model
- device_mesh: DeviceMesh information
- mixed_precision: Mixed precision information, default `bf16`, recommended to keep unchanged if you have GPUs with 30 series or above
- strategy: How to encapsulate the model for multi-GPU training, default uses `accelerate` framework.
- ddp_config: DDP configuration when strategy is `accelerate`, see: [DDPKwargs](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/dataclasses.py#L155)
- fsdp_config: FSDP configuration when strategy is `accelerate`, see: [FSDPConfig](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/dataclasses.py#L1566)
- grad_scaler_config: PyTorch's grad_scaler initialization configuration, see: [PyTorch's GradScaler constructor](https://github.com/pytorch/pytorch/blob/main/torch/cuda/amp/grad_scaler.py#L25)
- kwargs:
  - If you don't want to pass the model config field, you can put scattered configurations here. These parameters will be passed to `from_pretrained` or `from_config` later.

TransformersModel supports the `@remote_class` annotation and supports device_mesh, which means it can run in Ray workers.

Usage example:
```python
from twinkle.model import TransformersModel
from twinkle import DeviceMesh
from twinkle.dataloader import DataLoader
dataloader = DataLoader(...)
model = TransformersModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', device_mesh=DeviceMesh.from_sizes(dp_size=2, fsdp_size=2), remote_group='actor')
model.add_adapter_to_model(...)
model.set_optimizer(..., adapter_name='...')
for data in dataloader:
  model.forward_backward(...)
  model.clip_grad_and_step(..., gradient_accumulation_steps=16)
```
