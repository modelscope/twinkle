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
                 memory_efficient_init: bool = False,
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
- fsdp_config: strategy为`accelerate`时的FSDP配置，参见：[FSDPConfig](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/dataclasses.py#L1566)。strategy为`native_fsdp`时额外支持`mixed_precision`子配置（独立精度配置，字段对应FSDP2的[MixedPrecisionPolicy](https://docs.pytorch.org/docs/stable/distributed.fsdp.html#torch.distributed.fsdp.MixedPrecisionPolicy)）：
  - `param_dtype`: 计算精度。前向/反向前，FSDP将all-gather出的完整参数cast到该精度参与计算；
  - `reduce_dtype`: 梯度归约精度。跨卡reduce-scatter累加在该精度下进行，设为`fp32`可降低多卡累加误差，代价是归约通信量翻倍（即"BF16计算、FP32归约"）；
  - `output_dtype`: 每个FSDP包装单元forward输出cast到的精度。

  三个字段均可缺省，取值支持`'bf16'` / `'fp16'` / `'fp32'`（或对应的`torch.dtype`）；缺省字段回落到顶层`mixed_precision`字符串定出的基准精度，完全不配置时行为与旧版本一致。例如：

  ```python
  TransformersModel(
      ...,
      strategy='native_fsdp',
      fsdp_config={'mixed_precision': {'reduce_dtype': 'fp32'}},  # BF16计算、FP32梯度归约
  )
  ```

  注意：该子配置仅对`native_fsdp`生效；启用专家并行（EP）时，因NCCL约束专家组的梯度归约恒在`fp32`下进行。

  约束：生效的`output_dtype`必须与`param_dtype`一致（或缺省），否则构造时抛出`ValueError`——两者不一致会在FSDP单元边界引入dtype转换，梯度检查点重算时张量dtype不一致会导致训练报错；`reduce_dtype`只作用于梯度归约，不受该约束。
- grad_scaler_config: PyTorch的grad_scaler初始化配置，参见：[PyTorch的GradScaler构造](https://github.com/pytorch/pytorch/blob/main/torch/cuda/amp/grad_scaler.py#L25)
- memory_efficient_init: 是否启用FSDP内存高效初始化。启用后仅rank 0加载完整权重，其余rank通过广播获取分片参数，降低初始化阶段的内存和显存峰值。默认`False`。注意：该优化目前仅适用于 transformers <= 4.57.6；对于 transformers >= 5.0.0，可能会导致负面性能影响。
- kwargs:
  - 如果你不希望传递模型config字段，可以把零星的配置从这里放置进去。后续这些参数会传递到`from_pretrained`或者`from_config`中。

TransformersModel支持`@remote_class`注解，并且支持device_mesh，这意味着它可以运行在ray的worker中。

使用样例：
```python
from twinkle.model import TransformersModel
from twinkle import DeviceMesh
from twinkle.dataloader import DataLoader
dataloader = DataLoader(...)
model = TransformersModel(model_id='ms://Qwen/Qwen3.5-4B', device_mesh=DeviceMesh.from_sizes(dp_size=2, fsdp_size=2), remote_group='actor')
model.add_adapter_to_model(...)
model.set_optimizer(..., adapter_name='...')
for data in dataloader:
  model.forward_backward(...)
  model.clip_grad_and_step(..., gradient_accumulation_steps=16)
```

## 检查点保存与续训

`TransformersModel.save()` 既可以只保存权重，也可以保存可续训的训练检查点。

- `model.save(name, save_optimizer=True, consumed_train_samples=...)` 保存权重、优化器、调度器、scaler、RNG 状态和 `trainer_state.json`。
- `model.resume_from_checkpoint(checkpoint_dir)` 恢复完整训练状态（权重、优化器、调度器、scaler、RNG），返回 `{'cur_step', 'consumed_train_samples', 'gradient_accumulation_steps'}`。
- `model.resume_from_checkpoint(checkpoint_dir, resume_only_model=True)` 仅加载权重并返回进度元数据，不恢复优化器状态。
- `dataloader.resume_from_checkpoint(consumed_train_samples)` 跳过已消费的样本。
- `dataloader.get_state()` 返回 `{'consumed_train_samples': int}` — DataLoader 会自动追踪已消费样本数，无需手动维护计数器。

对于全参训练，恢复模型权重时需要在创建 `TransformersModel` 时直接把 checkpoint 路径传给 `model_id`，例如 `TransformersModel(model_id='./output/fsdp2/last-checkpoint')`，随后再调用 `resume_from_checkpoint(...)` 恢复优化器和训练进度。

如果需要完整的断点续训流程，包括 dataloader 跳过已消费数据的逻辑，建议直接参考 `cookbook/transformers/fsdp2.py`。
