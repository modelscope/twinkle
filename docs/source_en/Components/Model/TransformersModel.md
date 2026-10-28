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
                 memory_efficient_init: bool = False,
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
- fsdp_config: FSDP configuration when strategy is `accelerate`, see: [FSDPConfig](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/dataclasses.py#L1566). When strategy is `native_fsdp`, it additionally supports the `mixed_precision` sub-config (independent precision settings, mapping to FSDP2's [MixedPrecisionPolicy](https://docs.pytorch.org/docs/stable/distributed.fsdp.html#torch.distributed.fsdp.MixedPrecisionPolicy) fields):
  - `param_dtype`: Computation precision. Before forward/backward, FSDP casts the all-gathered full parameters to this dtype for computation;
  - `reduce_dtype`: Gradient reduction precision. The cross-rank reduce-scatter accumulation runs in this dtype; `fp32` lowers multi-rank accumulation error at the cost of doubled reduction traffic (i.e. "BF16 compute, FP32 reduce");
  - `output_dtype`: The dtype each FSDP-wrapped unit's forward output is cast to.

  All three fields are optional and accept `'bf16'` / `'fp16'` / `'fp32'` (or the corresponding `torch.dtype`); omitted fields fall back to the base dtype derived from the top-level `mixed_precision` string, so leaving the sub-config out keeps the legacy behavior. Example:

  ```python
  TransformersModel(
      ...,
      strategy='native_fsdp',
      fsdp_config={'mixed_precision': {'reduce_dtype': 'fp32'}},  # BF16 compute, FP32 gradient reduction
  )
  ```

  Note: this sub-config only takes effect for `native_fsdp`; with expert parallelism (EP) enabled, NCCL constraints force the expert group's gradient reduction to run in `fp32`.

  Constraint: the effective `output_dtype` must equal `param_dtype` (or be omitted); otherwise construction raises `ValueError` — a mismatch introduces a dtype conversion at FSDP unit boundaries, and gradient checkpointing recompute then fails on mixed-dtype tensors. `reduce_dtype` only affects gradient reduction and is unconstrained.
- grad_scaler_config: PyTorch's grad_scaler initialization configuration, see: [PyTorch's GradScaler constructor](https://github.com/pytorch/pytorch/blob/main/torch/cuda/amp/grad_scaler.py#L25)
- memory_efficient_init: Whether to enable memory-efficient model initialization for FSDP. When enabled, only rank 0 loads full weights and broadcasts sharded parameters to other ranks, reducing peak memory usage during initialization. Default `False`. Note: The optimization currently only applies to transformers <= 4.57.6; for transformers >= 5.0.0, it may lead to negative performance impact.
- kwargs:
  - If you don't want to pass the model config field, you can put scattered configurations here. These parameters will be passed to `from_pretrained` or `from_config` later.

TransformersModel supports the `@remote_class` annotation and supports device_mesh, which means it can run in Ray workers.

Usage example:
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

## Checkpoint and Resume

`TransformersModel.save()` can save either weights only or a resumable training checkpoint.

- `model.save(name, save_optimizer=True, consumed_train_samples=...)` saves weights together with optimizer, scheduler, scaler, RNG, and `trainer_state.json`.
- `model.resume_from_checkpoint(checkpoint_dir)` restores full training state (weights, optimizer, scheduler, scaler, RNG) and returns `{'cur_step', 'consumed_train_samples', 'gradient_accumulation_steps'}`.
- `model.resume_from_checkpoint(checkpoint_dir, resume_only_model=True)` loads weights only and returns progress metadata without restoring optimizer state.
- `dataloader.resume_from_checkpoint(consumed_train_samples)` skips already-consumed samples.
- `dataloader.get_state()` returns `{'consumed_train_samples': int}` — the dataloader automatically tracks consumed samples, so you don't need to maintain a counter manually.

For full-parameter training, restore model weights by constructing `TransformersModel` with the checkpoint path as `model_id`, for example `TransformersModel(model_id='./output/fsdp2/last-checkpoint')`, and then call `resume_from_checkpoint(...)` to restore optimizer state and training progress.

For end-to-end resume logic, including dataloader skipping, refer to `cookbook/transformers/fsdp2.py`.
