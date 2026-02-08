# MegatronModel

This model encapsulates Megatron LLM and can start the model using TP/DP/CP/PP/EP combinations.

> Note: VPP support currently has issues, please do not configure and use it for now.

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

- model_id: Model id
- config: Configuration for starting the model
- device_mesh: DeviceMesh information
- mixed_precision: Mixed precision information, default `bf16`, recommended to keep unchanged if you have GPUs with 30 series or above
- kwargs:
  - All Megatron initialization parameters, i.e., [`TransformersConfig`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/transformer_config.py#L34) configurations can be passed into kwargs.

MegatronModel supports the `@remote_class` annotation and supports device_mesh, which means it can run in Ray workers.

Usage example:
```python
from twinkle.model import MegatronModel
from twinkle import DeviceMesh
from twinkle.dataloader import DataLoader
dataloader = DataLoader(...)
model = MegatronModel(model_id='ms://Qwen/Qwen2.5-7B-Instruct', device_mesh=DeviceMesh.from_sizes(dp_size=2, tp_size=2, pp_size=2), remote_group='actor')
model.add_adapter_to_model(...)
model.set_optimizer('default', adapter_name='...')
for data in dataloader:
  model.forward_backward(...)
  model.clip_grad_and_step(..., gradient_accumulation_steps=16)
```

> Note:
> 1. Megatron models do not support using AdamW's original optimizer, only support configuring `MegatronDistributedOptimizer`, you can pass `MegatronDistributedOptimizer`, `default` to use it
> 2. Megatron models do not support using other lr_schedulers, only support using `OptimizerParamScheduler`, you can pass `OptimizerParamScheduler`, `default` to use it
> 3. You need to pass tp/cp/dp/ep/pp/sequence_parallel configurations into the device_mesh parameter to facilitate twinkle to manage data distribution. These parameters will be passed by device_mesh to the megatron initialization process
