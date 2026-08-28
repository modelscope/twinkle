# CheckpointEngine

CheckpointEngine (检查点引擎) 是用于在训练器和推理进程之间同步模型权重的组件,主要用于 RLHF 训练中 Actor 模型和 Rollout 采样器之间的权重同步。

`CheckpointEngineManager` 提供四种模式：

- `auto`：本地对象使用 `naive`；Ray actor handler 使用 `standalone`。
- `naive`：模型的权重生成器直接流式传入本地 sampler，不创建 CheckpointEngine。
- `colocate`：共享 GPU 的 Ray actors 通过 CUDA IPC 同步。
- `standalone`：分离部署的 Ray actors 在 GPU 上使用 NCCL，在 NPU 上使用 HCCL。

`auto` 不会推断 `colocate`，因为 driver 无法可靠判断 actor 的实际设备放置。

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

Twinkle 提供三种跨进程检查点引擎实现；`naive` 模式会绕过这些引擎。

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

### IPCCheckpointEngine

适用于模型和 sampler Ray actors 被放置在同一组物理 GPU 上的 CUDA IPC 引擎。NCCL 会拒绝多个
rank 绑定同一张 GPU，因此该拓扑必须通过 CUDA IPC 在进程间映射权重 bucket，而不是跨设备广播。

## 如何选择

- **NCCLCheckpointEngine**: 适用于 GPU 环境,提供最高的传输性能
- **HCCLCheckpointEngine**: 适用于昇腾 NPU 环境
- **IPCCheckpointEngine**: 适用于共享物理 GPU 的 colocated Ray actors
- **不创建引擎 (`naive`)**: 适用于同一进程内的本地 model 和 sampler

> 检查点引擎是 RLHF 训练基础设施的关键组件,确保训练器和采样器使用一致的模型权重。
> 目前的同步分为merge_and_sync=True/False两种情况，为True时将lora合并仅基模并同步，为False时仅同步lora权重。另外，多租户直接附加lora文件到vLLM上，在merge_and_sync=False，或使用多租户时，
> vLLM的启动参数需要开启`enable_lora=True`，在merge_and_sync=True或全参时，该值设置为False.
