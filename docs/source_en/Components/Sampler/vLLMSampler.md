# vLLMSampler

vLLMSampler uses the vLLM engine for efficient inference, supporting high-throughput batch sampling.

## Usage Example

```python
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams
from twinkle import DeviceMesh

# Create sampler
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=2, tp_size=2),
    remote_group='sampler_group'
)

# Add LoRA
sampler.add_adapter_to_model('my_lora', 'path/to/lora')

# Set sampling parameters
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50
)

# Perform sampling
response = sampler.sample(
    trajectories,
    sampling_params=params,
    adapter_name='my_lora',
    num_samples=4  # Generate 4 samples per prompt
)
```

## Features

- **High Performance**: Achieves high throughput using PagedAttention and continuous batching
- **LoRA Support**: Supports dynamic loading and switching of LoRA adapters
- **Multi-Sample Generation**: Can generate multiple samples per prompt
- **Tensor Parallel**: Supports tensor parallelism to accelerate large model inference

## Remote Execution

vLLMSampler supports the `@remote_class` decorator and can run in Ray clusters:

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.sampler import vLLMSampler

# Initialize Ray cluster
device_groups = [
    DeviceGroup(name='sampler', ranks=4, device_type='cuda')
]
twinkle.initialize('ray', groups=device_groups)

# Create remote sampler
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen2.5-7B-Instruct',
    device_mesh=DeviceMesh.from_sizes(dp_size=4),
    remote_group='sampler'
)

# sample method executes in remote worker
response = sampler.sample(trajectories, sampling_params=params)
```

> In RLHF training, vLLMSampler is typically separated from the Actor model, using different hardware resources to avoid interference between inference and training.
