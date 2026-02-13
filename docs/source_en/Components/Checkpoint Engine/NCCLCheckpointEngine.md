# NCCLCheckpointEngine

A checkpoint engine that uses NCCL for high-speed weight transfer between GPUs.

## Usage Example

```python
from twinkle.checkpoint_engine import NCCLCheckpointEngine

# In training process (rank 0)
engine = NCCLCheckpointEngine(bucket_size=512<<20)  # 512MB bucket
engine.is_master = True
engine.prepare()
engine.init_process_group(rank=0, world_size=5)

# Send weights
await engine.send_weights(model.named_parameters())
engine.finalize()

# In inference process (rank 1-4)
engine = NCCLCheckpointEngine(bucket_size=512<<20)
engine.prepare()
engine.init_process_group(rank=1, world_size=5, master_metadata=metadata)

# Receive weights
async for name, tensor in engine.receive_weights():
    model.load_state_dict({name: tensor}, strict=False)
engine.finalize()
```

## Features

- **High-Speed Transfer**: Uses NCCL for GPU-to-GPU point-to-point high-speed transfer
- **Zero-Copy**: Direct transfer between GPU memories without going through CPU
- **Bucketed Transfer**: Supports bucketed transfer for large models

## Configuration Parameters

- **bucket_size**: Weight bucket size, controls the amount of data transferred each time. Larger buckets can improve transfer efficiency but consume more memory
- **timeout**: Transfer timeout duration

> NCCLCheckpointEngine is the recommended choice for GPU training, providing the highest transfer performance.
