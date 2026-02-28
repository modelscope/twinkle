# TransformersModel

这个模型封装了transformers的LLM，并可以使用FSDP2、DDP等方式启动并训练模型。

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

- model_cls: 使用哪个类拉起模型，默认为`AutoModelForCausalLM`
- model_id: 模型id
- config: 拉起模型的配置
- device_mesh: DeviceMesh信息
- mixed_precision: 混合精度信息，默认`bf16`，如果有30系以上显卡建议维持不变
- strategy: 如何封装模型使用多卡训练，默认使用`accelerate`框架。
- ddp_config: strategy为`accelerate`时的DDP配置，参见：[DDPKwargs](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/dataclasses.py#L155)
- fsdp_config: strategy为`accelerate`时的FSDP配置，参见：[FSDPConfig](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/dataclasses.py#L1566)
- grad_scaler_config: PyTorch的grad_scaler初始化配置，参见：[PyTorch的GradScaler构造](https://github.com/pytorch/pytorch/blob/main/torch/cuda/amp/grad_scaler.py#L25)
- kwargs:
  - 如果你不希望传递模型config字段，可以把零星的配置从这里放置进去。后续这些参数会传递到`from_pretrained`或者`from_config`中。

TransformersModel支持`@remote_class`注解，并且支持device_mesh，这意味着它可以运行在ray的worker中。

使用样例：
```python
from twinkle.model import TransformersModel
from twinkle import DeviceMesh
from twinkle.dataloader import DataLoader
dataloader = DataLoader(...)
model = TransformersModel(model_id='ms://Qwen/Qwen3-4B', device_mesh=DeviceMesh.from_sizes(dp_size=2, fsdp_size=2), remote_group='actor')
model.add_adapter_to_model(...)
model.set_optimizer(..., adapter_name='...')
for data in dataloader:
  model.forward_backward(...)
  model.clip_grad_and_step(..., gradient_accumulation_steps=16)
```
