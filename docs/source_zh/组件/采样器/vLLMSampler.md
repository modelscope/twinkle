# vLLMSampler

vLLMSampler 使用 vLLM 引擎进行高效推理,支持高吞吐量的批量采样。

## 使用示例

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams
from twinkle import DeviceMesh

# 创建采样器
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=2, tp_size=2),
    remote_group='sampler_group'
)

# 添加 LoRA
sampler.add_adapter_to_model('my_lora', 'path/to/lora')

# 设置采样参数
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50
)

# 进行采样
response = sampler.sample(
    trajectories,
    sampling_params=params,
    adapter_name='my_lora',
    num_samples=4  # 每个 prompt 生成 4 个样本
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
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=4),
    remote_group='sampler'
)

# sample 方法会在 remote worker 中执行
response = sampler.sample(trajectories, sampling_params=params)
```

> vLLMSampler 在 RLHF 训练中通常与 Actor 模型分离,使用不同的硬件资源,避免推理和训练相互干扰。
