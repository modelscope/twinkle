# HCCLCheckpointEngine

A checkpoint engine that uses HCCL for weight transfer between Ascend NPUs.

## Usage Example

```python
from twinkle.checkpoint_engine import HCCLCheckpointEngine

engine = HCCLCheckpointEngine(bucket_size=512<<20)
# Usage is the same as NCCLCheckpointEngine
```

## Features

- **NPU Optimized**: Weight transfer optimized specifically for Ascend NPUs
- **Efficient Communication**: Uses HCCL for high-speed communication between NPUs
- **Compatible Interface**: Maintains consistent interface with NCCLCheckpointEngine

## Use Cases

HCCLCheckpointEngine is specifically designed for Ascend NPU environments:

- Training on Huawei Ascend NPUs
- Synchronizing model weights between NPUs
- Large-scale NPU cluster deployment

> In Ascend NPU environments, HCCLCheckpointEngine provides performance comparable to NCCL.
