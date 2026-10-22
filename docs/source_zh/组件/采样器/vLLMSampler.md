# vLLMSampler

vLLMSampler 使用 vLLM 引擎进行高效推理,支持高吞吐量的批量采样。

## 使用示例

```python
import twinkle
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams, Trajectory
from twinkle import DeviceMesh

twinkle.initialize(mode='local', nproc_per_node=1)
trajectories = [Trajectory(messages=[{'role': 'user', 'content': 'What is 2 + 3?'}])]

# 创建采样器
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    engine_args={'enable_lora': True, 'max_lora_rank': 16},
    device_mesh=DeviceMesh.from_sizes(dp_size=1),
)

# 使用模型模板编码聊天消息
sampler.set_template('Qwen3_5Template')

# 设置采样参数
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_samples=4,
)

# 进行采样
responses = sampler.sample(
    trajectories,
    sampling_params=params,
    adapter_path='path/to/lora',  # 已保存的 LoRA checkpoint（rank <= max_lora_rank）
)
```

## 特性

- **高性能**: 使用 PagedAttention 和连续批处理实现高吞吐量
- **LoRA 支持**: 支持动态加载和切换 LoRA 适配器
- **多样本生成**: 可以为每个 prompt 生成多个样本
- **Tensor Parallel**: 支持张量并行加速大模型推理

## 远程执行

vLLMSampler 支持 `@remote_class` 装饰器,可以在 Ray 集群中运行:

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.sampler import vLLMSampler

# 初始化 Ray 集群
device_groups = [
    DeviceGroup(name='sampler', ranks=4, device_type='cuda')
]
twinkle.initialize('ray', groups=device_groups)

# 创建远程采样器
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=DeviceMesh.from_sizes(dp_size=4),
    remote_group='sampler'
)

# sample 方法会在 remote worker 中执行
responses = sampler.sample(trajectories, sampling_params=params)
```

## 环境变量

- `TWINKLE_VLLM_IPC_TIMEOUT_S`:
  控制 `vLLMSampler` 与 vLLM worker extension 之间 IPC 通道（ZMQ REQ/REP）的超时时间（秒）。
  默认值为 `300`。该值必须大于 `0`。

> vLLMSampler 在 RLHF 训练中通常与 Actor 模型分离,使用不同的硬件资源,避免推理和训练相互干扰。
