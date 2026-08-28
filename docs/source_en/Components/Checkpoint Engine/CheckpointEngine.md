# CheckpointEngine

CheckpointEngine is a component used to synchronize model weights between trainer and inference processes, primarily used in RLHF training to synchronize weights between Actor models and Rollout samplers.

`CheckpointEngineManager` exposes four modes:

- `auto`: local objects use `naive`; Ray actor handlers use `standalone`.
- `naive`: stream the model's weight generator directly into a local sampler without creating a checkpoint engine.
- `colocate`: synchronize Ray actors sharing GPUs through CUDA IPC.
- `standalone`: synchronize disaggregated Ray actors through NCCL on GPU or HCCL on NPU.

`auto` never infers `colocate`, because actor placement cannot be determined reliably from the driver.

## Basic Interface

```python
class CheckpointEngine(ABC):
    """Checkpoint engine base class

    The checkpoint engine handles weight synchronization between trainer and inference processes.
    """

    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """Prepare for weight synchronization"""
        ...

    @abstractmethod
    def init_process_group(self, rank: int, world_size: int, **kwargs):
        """Initialize process group"""
        ...

    @abstractmethod
    async def send_weights(self, weight_generator):
        """Send weights (called in trainer process)"""
        ...

    @abstractmethod
    def receive_weights(self) -> AsyncGenerator:
        """Receive weights (called in inference process)"""
        ...

    @abstractmethod
    def finalize(self):
        """Clean up resources"""
        ...
```

## Available Checkpoint Engines

Twinkle provides three cross-process checkpoint engine implementations. `naive` mode bypasses them.

### NCCLCheckpointEngine

A checkpoint engine that uses NCCL for high-speed weight transfer between GPUs.

- High-Speed Transfer: Uses NCCL for GPU-to-GPU point-to-point high-speed transfer
- Zero-Copy: Direct transfer between GPU memories without going through CPU
- Bucketed Transfer: Supports bucketed transfer for large models

See: [NCCLCheckpointEngine](NCCLCheckpointEngine.md)

### HCCLCheckpointEngine

A checkpoint engine that uses HCCL for weight transfer between Ascend NPUs.

- NPU Optimized: Weight transfer optimized specifically for Ascend NPUs
- Efficient Communication: Uses HCCL for high-speed communication between NPUs
- Compatible Interface: Maintains consistent interface with NCCLCheckpointEngine

See: [HCCLCheckpointEngine](HCCLCheckpointEngine.md)

### IPCCheckpointEngine

A CUDA IPC engine for model and sampler Ray actors placed on the same physical GPUs. NCCL cannot be
used for this topology because it rejects multiple ranks bound to one GPU. Weight buckets are mapped
between the actor processes rather than broadcast between devices.

## How to Choose

- **NCCLCheckpointEngine**: Suitable for GPU environments, provides the highest transfer performance
- **HCCLCheckpointEngine**: Suitable for Ascend NPU environments
- **IPCCheckpointEngine**: Required for colocated Ray actors sharing physical GPUs
- **No engine (`naive`)**: Local model and sampler objects in the same process

> Checkpoint engine is a key component of RLHF training infrastructure, ensuring that trainers and samplers use consistent model weights.
> Currently, synchronization is divided into two cases based on merge_and_sync=True/False. When set to True, the LoRA is merged into the base model and then synchronized.
> When set to False, only the LoRA weights are synchronized. Additionally, for multi-tenant scenarios, LoRA files are directly attached to vLLM.
> When merge_and_sync=False or in multi-tenant mode, vLLM's startup parameter enable_lora=True needs to be enabled. When merge_and_sync=True or using full parameters, this value should be set to False.
