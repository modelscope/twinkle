# Sampler

Sampler is a component in Twinkle for generating model outputs, primarily used for sample generation in RLHF training. The sampler supports multiple inference engines, including vLLM and native PyTorch.

## Basic Interface

```python
class Sampler(ABC):
    
    @abstractmethod
    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[SamplingParams] = None,
        adapter_name: str = '',
        *,
        num_samples: int = 1,
    ) -> SampleResponse:
        """Sample from given inputs"""
        ...
    
    def add_adapter_to_model(self, adapter_name: str, config_or_dir, **kwargs):
        """Add LoRA adapter"""
        ...
    
    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """Set template"""
        ...
```

The core method of the sampler is `sample`, which accepts input data and returns generated samples.

## Available Samplers

Twinkle provides two sampler implementations:

### vLLMSampler

vLLMSampler uses the vLLM engine for efficient inference, supporting high-throughput batch sampling.

- High Performance: Uses PagedAttention and continuous batching
- LoRA Support: Supports dynamic loading and switching of LoRA adapters
- Multi-Sample Generation: Can generate multiple samples per prompt
- Tensor Parallel: Supports tensor parallelism to accelerate large model inference

See: [vLLMSampler](vLLMSampler.md)

### TorchSampler

TorchSampler uses native PyTorch and transformers for inference, suitable for small-scale sampling or debugging.

- Easy to Use: Based on transformers' standard interface
- High Flexibility: Easy to customize and extend
- Low Memory Footprint: Suitable for small-scale sampling

See: [TorchSampler](TorchSampler.md)

## How to Choose

- **vLLMSampler**: Suitable for production environments and large-scale training that require high throughput
- **TorchSampler**: Suitable for debugging, small-scale experiments, or custom requirements

> In RLHF training, samplers are typically separated from the Actor model, using different hardware resources to avoid interference between inference and training.
