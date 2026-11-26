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

Each normal candidate's `new_input_feature` is the complete trajectory: messages, tools, user_data, original media and numerical encoding fields. `SampledSequence` still carries tokens, logprobs, decoded text, stop_reason, routed_experts and sampling_masks; there is no separate `new_state`. `max_tokens=0` scores the prompt without adding an assistant message.

The materialized HTTP client opts into `project_request_fields=True`: the response identifies unchanged request fields for the client to restore without a tokenizer. An unchanged message history is restored locally; the response sends only its generated extension. Callers that do not negotiate projection retain the existing complete-history response. Changed histories and newly encoded numerical fields remain in the response. Explicit `token_only=True` returns tokens and statistics; `HTTPSampler` uses its local template to append the actual tokens without retokenizing the reply. Server-side DataPlane storage keeps the complete trajectory independently of HTTP projection. This reduces response duplication, not per-turn media uploads.

## Available Samplers

Twinkle provides the `vLLMSampler` implementation:

### vLLMSampler

vLLMSampler uses the vLLM engine for efficient inference, supporting high-throughput batch sampling.

- High Performance: Uses PagedAttention and continuous batching
- LoRA Support: Supports dynamic loading and switching of LoRA adapters
- Multi-Sample Generation: Can generate multiple samples per prompt
- Tensor Parallel: Supports tensor parallelism to accelerate large model inference

See: [vLLMSampler](vLLMSampler.md)

Use `sampler_type: vllm` for standard sampling, including the {ref}`client-side streaming GRPO examples <client-async-grpo>`. The client-side `StreamingDriver` schedules rollouts and training. `mock` is for tests. The former [TorchSampler](TorchSampler.md) example is unavailable.

> In RLHF training, samplers are typically separated from the Actor model, using different hardware resources to avoid interference between inference and training.
