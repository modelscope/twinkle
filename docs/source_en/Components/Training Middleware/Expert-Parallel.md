# Expert Parallel (EP)

Expert Parallel distributes Mixture-of-Experts (MoE) model experts across multiple GPUs, allowing each rank to hold a subset of experts. This reduces per-GPU memory and enables training of large MoE models.

## Overview

| Concept | Description |
|---------|-------------|
| **ExpertParallelConfig** | Configuration dataclass controlling EP behavior |
| **apply_expert_parallel()** | Entry point that shards experts and patches forward |
| **shard_experts()** | Evenly splits experts across EP ranks |
| **patch_forward()** | Replaces MoE block forward with EP-aware all-to-all communication |

## Configuration

```python
from twinkle.model.transformers.moe.expert_parallel import ExpertParallelConfig

config = ExpertParallelConfig(
    enabled=True,              # Enable expert parallel
    router_dtype='fp32',       # Router computation dtype: 'fp32', 'bf16', 'fp16'
    keep_router_logits=True,   # Return router logits alongside hidden states
    ignore_shared_experts=False,# Skip shared expert computation (e.g. DeepSeek)
    ep_size=None,              # EP world size (consumed by TransformersModel)
)
```

## Usage with DeviceMesh

EP is activated by setting `ep_size` in `DeviceMesh.from_sizes()`. The framework automatically calls `apply_expert_parallel()` during model initialization.

```python
from twinkle.utils import DeviceMesh

# 8 GPUs: 2-way EP × 4-way data parallel
device_mesh = DeviceMesh.from_sizes(
    world_size=8,
    dp_size=4,
    ep_size=2,
)
```

For combined EP + FSDP sharding on the expert parameters:

```python
# 8 GPUs: 2-way EP with FSDP within each EP group
device_mesh = DeviceMesh.from_sizes(
    world_size=8,
    dp_size=2,
    ep_size=2,
    ep_fsdp_size=2,
)
```

## Communication Pattern

The EP forward pass follows a 4-stage pipeline:

1. **Preprocess** — compute per-expert token counts and split sizes
2. **Token Pre-All2All** — permute tokens by expert assignment, then all-to-all exchange across EP ranks
3. **Expert Compute** — each rank runs its local experts on received tokens
4. **Token Post-All2All** — all-to-all exchange results back, unpermute and apply routing weights

```
Input tokens → Router → [preprocess] → [pre_all2all] → [local experts] → [post_all2all] → Output
```

## Requirements

- `num_experts` must be divisible by `ep_size`
- `torch.distributed` must be initialized
- MoE blocks must define a `gate`/`router` module and `experts` (either `nn.ModuleList` or tensor-style `gate_up_proj`/`down_proj`)
- Both ModuleList-style and tensor-style (fused) experts are supported
- Shared experts (e.g. DeepSeek MoE) are handled automatically unless `ignore_shared_experts=True`
