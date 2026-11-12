# vLLMSampler

vLLMSampler uses the vLLM engine for efficient inference, supporting high-throughput batch sampling.

## Usage Example

```python
import twinkle
from twinkle.sampler import vLLMSampler
from twinkle.data_format import SamplingParams, Trajectory
from twinkle import DeviceMesh

twinkle.initialize(mode='local', nproc_per_node=1)
trajectories = [Trajectory(messages=[{'role': 'user', 'content': 'What is 2 + 3?'}])]

# Create sampler
sampler = vLLMSampler(
    model_id='ms://Qwen/Qwen3.5-4B',
    engine_args={'enable_lora': True, 'max_lora_rank': 16},
    device_mesh=DeviceMesh.from_sizes(dp_size=1),
)

# Encode chat messages using the model's template
sampler.set_template('Qwen3_5Template')

# Set sampling parameters
params = SamplingParams(
    max_tokens=512,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    num_samples=4,
)

# Perform sampling
responses = sampler.sample(
    trajectories,
    sampling_params=params,
    adapter_path='path/to/lora',  # Saved LoRA checkpoint (rank <= max_lora_rank)
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
    model_id='ms://Qwen/Qwen3.5-4B',
    device_mesh=DeviceMesh.from_sizes(dp_size=4),
    remote_group='sampler'
)

# sample method executes in remote worker
responses = sampler.sample(trajectories, sampling_params=params)
```

## Environment Variables

- `TWINKLE_VLLM_IPC_TIMEOUT_S`:
  Controls the timeout (in seconds) for the IPC channel (ZMQ REQ/REP)
  between `vLLMSampler` and the vLLM worker extension.
  Default is `300`. This value must be greater than `0`.

> In RLHF training, vLLMSampler is typically separated from the Actor model, using different hardware resources to avoid interference between inference and training.
