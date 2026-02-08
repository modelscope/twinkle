# CheckpointEngine

CheckpointEngine is a component used to synchronize model weights between trainer and inference processes, primarily used in RLHF training to synchronize weights between Actor models and Rollout samplers.

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

Twinkle provides two checkpoint engine implementations:

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

## How to Choose

- **NCCLCheckpointEngine**: Suitable for GPU environments, provides the highest transfer performance
- **HCCLCheckpointEngine**: Suitable for Ascend NPU environments

> Checkpoint engine is a key component of RLHF training infrastructure, ensuring that trainers and samplers use consistent model weights.
