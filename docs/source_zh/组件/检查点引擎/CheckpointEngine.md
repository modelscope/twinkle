# CheckpointEngine

CheckpointEngine (检查点引擎) 是用于在训练器和推理进程之间同步模型权重的组件,主要用于 RLHF 训练中 Actor 模型和 Rollout 采样器之间的权重同步。

## 基本接口

```python
class CheckpointEngine(ABC):
    """检查点引擎基类
    
    检查点引擎处理训练器和推理进程之间的权重同步。
    """
    
    @abstractmethod
    def prepare(self) -> dict[str, Any]:
        """准备权重同步前的准备工作"""
        ...
    
    @abstractmethod
    def init_process_group(self, rank: int, world_size: int, **kwargs):
        """初始化进程组"""
        ...
    
    @abstractmethod
    async def send_weights(self, weight_generator):
        """发送权重(在训练器进程中调用)"""
        ...
    
    @abstractmethod
    def receive_weights(self) -> AsyncGenerator:
        """接收权重(在推理进程中调用)"""
        ...
    
    @abstractmethod
    def finalize(self):
        """清理资源"""
        ...
```

## 可用的检查点引擎

Twinkle 提供了两种检查点引擎实现:

### NCCLCheckpointEngine

使用 NCCL 进行 GPU 间高速权重传输的检查点引擎。

- 高速传输: 使用 NCCL 实现 GPU 间点对点高速传输
- 零拷贝: 直接在 GPU 内存间传输,无需经过 CPU
- 分桶传输: 支持大模型的分桶传输

详见: [NCCLCheckpointEngine](NCCLCheckpointEngine.md)

### HCCLCheckpointEngine

使用 HCCL 进行昇腾 NPU 间权重传输的检查点引擎。

- NPU 优化: 专为昇腾 NPU 优化的权重传输
- 高效通信: 使用 HCCL 实现 NPU 间高速通信
- 兼容接口: 与 NCCLCheckpointEngine 保持一致的接口

详见: [HCCLCheckpointEngine](HCCLCheckpointEngine.md)

## 如何选择

- **NCCLCheckpointEngine**: 适用于 GPU 环境,提供最高的传输性能
- **HCCLCheckpointEngine**: 适用于昇腾 NPU 环境

> 检查点引擎是 RLHF 训练基础设施的关键组件,确保训练器和采样器使用一致的模型权重。

