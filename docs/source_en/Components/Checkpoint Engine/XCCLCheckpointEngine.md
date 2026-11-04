# XCCLCheckpointEngine

A checkpoint engine for Kunlunxin XPU that transfers weights over BKCL (mounted into PyTorch as the `nccl`/XCCL backend).

## Usage Example

```python
from twinkle.checkpoint_engine import XCCLCheckpointEngine

engine = XCCLCheckpointEngine(bucket_size=512<<20)
# Usage is the same as NCCLCheckpointEngine
```

On XPU this engine is selected automatically by `CheckpointEngineManager` / `CheckpointEngineMixin`, so you normally do not construct it directly.

## Features

- **Drop-in for NCCL**: Inherits `NCCLCheckpointEngine`; bucketing, ZMQ metadata handshake, and double buffering are reused unchanged.
- **XCCL group + relay fallback**: Replaces the direct `ProcessGroupNCCL` construction (which deadlocks when two ranks share a local device index) with a stateless `ProcessGroupXCCL`, plus a relay path for colliding ranks.
- **Store-based init barrier**: The readiness barrier runs through the TCPStore instead of a device collective.

## Why a Separate Engine

BKCL identifies a rank's device by its **local device index** and does not translate `CUDA_VISIBLE_DEVICES` to physical cards. In a separated trainer/sampler deployment on a single host, both sides see their device as local `0`, producing duplicate `(host, index)` pairs. Directly building `ProcessGroupNCCL` then fails to form the ring and the first broadcast deadlocks (600s WorkXCCL timeout).

`XCCLCheckpointEngine` exchanges `(hostname, local_device_index)` through the store, keeps colliding ranks out of the XCCL group, and serves them through the lowest-rank XCCL member over direct TCP sockets (relay). Non-colliding ranks use a flat stateless `ProcessGroupXCCL`.

## Use Cases

- Weight synchronization (train ↔ sample) on Kunlunxin XPU
- Colocate / separated RL (GRPO) with full-weight sync (`merge_and_sync=True`)

## Limitations

- Multi-host (inter/intra-node) broadcast is implemented but not yet validated; only single-host relay mode is verified.

> On Kunlunxin XPU, use `merge_and_sync=True` for weight sync; LoRA sampling is currently blocked by vendor kernel dtype limitations (see the [XPU Support](../../Usage%20Guide/XPU-Support.md) guide).
