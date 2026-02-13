# TorchSampler

TorchSampler 使用原生 PyTorch 和 transformers 进行推理,适合小规模采样或调试。

## 使用示例

```python
from twinkle.sampler import TorchSampler
from twinkle import DeviceMesh

sampler = TorchSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=1),
)

response = sampler.sample(trajectories, sampling_params=params)
```

## 特性

- **简单易用**: 基于 transformers 的标准接口
- **灵活性高**: 容易定制和扩展
- **内存占用小**: 适合小规模采样

## 适用场景

TorchSampler 特别适合以下场景:

- **调试和开发**: 简单直接,容易调试
- **小规模实验**: 不需要高吞吐量的场景
- **自定义需求**: 需要修改采样逻辑的场景
- **资源受限**: 内存或GPU资源有限的环境

> 对于生产环境或大规模训练,建议使用 [vLLMSampler](vLLMSampler.md) 以获得更好的性能。
