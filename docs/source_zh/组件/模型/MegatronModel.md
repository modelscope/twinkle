# MegatronModel

这个模型封装了Megatron的LLM，并可以使用TP/DP/CP/PP/EP组合启动模型。

> 注意：VPP的支持目前存在问题，请暂时不要配置使用。

```python
class MegatronModel:

    def __init__(
        self,
        model_id: str,
        config: Optional[PretrainedConfig] = None,
        device_mesh: Optional[DeviceMesh] = None,
        mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
        **kwargs,
    ):
        ...

    ...
```

- model_id: 模型id
- config: 拉起模型的配置
- device_mesh: DeviceMesh信息
- mixed_precision: 混合精度信息，默认`bf16`，如果有30系以上显卡建议维持不变
- kwargs:
  - 所有Megatron初始化的参数，即[`TransformersConfig`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py#L34)的配置均可以传递到kwargs中。

MegatronModel支持`@remote_class`注解，并且支持device_mesh，这意味着它可以运行在ray的worker中。

使用样例：
```python
from twinkle.model import MegatronModel
from twinkle import DeviceMesh
from twinkle.dataloader import DataLoader
dataloader = DataLoader(...)
model = MegatronModel(model_id='ms://Qwen/Qwen3-4B', device_mesh=DeviceMesh.from_sizes(dp_size=2, tp_size=2, pp_size=2), remote_group='actor')
model.add_adapter_to_model(...)
model.set_optimizer('default', adapter_name='...')
for data in dataloader:
  model.forward_backward(...)
  model.clip_grad_and_step(..., gradient_accumulation_steps=16)
```

> 注意：
> 1. megatron模型不支持使用AdamW的原始optimizer，仅支持配置`MegatronDistributedOptimizer`, 你可以传递`MegatronDistributedOptimizer`, `default`来使用它
> 2. megatron模型不支持使用其他lr_scheduler，仅支持使用`OptimizerParamScheduler`，你可以传递`OptimizerParamScheduler`, `default`来使用它
> 3. 你需要将tp/cp/dp/ep/pp/sequence_parallel配置传入device_mesh参数中，以方便twinkle管理数据分配。这些参数会由device_mesh代为传递到megatron初始化流程中
