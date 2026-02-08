# TorchSampler

TorchSampler uses native PyTorch and transformers for inference, suitable for small-scale sampling or debugging.

## Usage Example

```python
from twinkle.sampler import TorchSampler
from twinkle import DeviceMesh

sampler = TorchSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=1),
)

response = sampler.sample(trajectories, sampling_params=params)
```

## Features

- **Easy to Use**: Based on transformers' standard interface
- **High Flexibility**: Easy to customize and extend
- **Low Memory Footprint**: Suitable for small-scale sampling

## Use Cases

TorchSampler is particularly suitable for:

- **Debugging and Development**: Simple and straightforward, easy to debug
- **Small-Scale Experiments**: Scenarios that don't require high throughput
- **Custom Requirements**: Scenarios that need to modify sampling logic
- **Resource-Constrained**: Environments with limited memory or GPU resources

> For production environments or large-scale training, it's recommended to use [vLLMSampler](vLLMSampler.md) for better performance.
