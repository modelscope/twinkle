# Sampler

Sampler is a component in Twinkle for generating model outputs, primarily used for sample generation in RLHF training. The current sampler implementation uses vLLM.

## vLLMSampler Sampling Interface

The concrete `vLLMSampler` exposes the following sampling interface:

```python
class vLLMSampler:

    def sample(
        self,
        inputs: Union[InputFeature, List[InputFeature], Trajectory, List[Trajectory]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_path: Optional[str] = None,
        *,
        return_encoded: bool = False,
        use_base_model: bool = False,
    ) -> List[SampleResponse]:
        """Sample from given inputs"""
        ...

    def set_template(self, template_cls: Union[Template, Type[Template], str], **kwargs):
        """Set template"""
        ...
```

The core method is `sample`, which returns a list of `SampleResponse` objects. Set `SamplingParams(num_samples=N)` to generate multiple sequences per prompt; each response contains `sequences`.

## Available Samplers

Twinkle provides the `vLLMSampler` implementation:

### vLLMSampler

vLLMSampler uses the vLLM engine for efficient inference, supporting high-throughput batch sampling.

- High Performance: Uses PagedAttention and continuous batching
- LoRA Support: Supports dynamic loading and switching of LoRA adapters
- Multi-Sample Generation: Can generate multiple samples per prompt
- Tensor Parallel: Supports tensor parallelism to accelerate large model inference

See: [vLLMSampler](vLLMSampler.md)

Server configurations use `sampler_type: vllm` for standard sampling or `vllm_async` for async RL. `mock` is for tests. The former [TorchSampler](TorchSampler.md) example is unavailable.

> In RLHF training, samplers are typically separated from the Actor model, using different hardware resources to avoid interference between inference and training.
