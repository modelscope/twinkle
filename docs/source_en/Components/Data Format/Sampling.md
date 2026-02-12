# Sampling Output

Sampling output is a data format used to represent input parameters and return results of the sampling process.

## SamplingParams

Sampling parameters are used to control the model's sampling behavior.

```python
@dataclass
class SamplingParams:
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[str, Sequence[str], Sequence[int], None] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    repetition_penalty: float = 1.0
```

- max_tokens: Maximum number of tokens to generate
- seed: Random seed
- stop: Stop sequences, can be a string, sequence of strings, or sequence of token ids
- temperature: Temperature parameter controlling sampling randomness. 0 means greedy sampling
- top_k: Top-K sampling parameter, -1 means not used
- top_p: Top-P (nucleus) sampling parameter
- repetition_penalty: Repetition penalty coefficient

### Conversion Methods

SamplingParams provides conversion methods to adapt to different inference engines:

```python
# Convert to vLLM's SamplingParams
vllm_params = params.to_vllm(num_samples=4, logprobs=True, prompt_logprobs=0)

# Convert to transformers' generate parameters
gen_kwargs = params.to_transformers(tokenizer=tokenizer)
```

## SampleResponse

Sample response is the result data structure returned by the sampler.

```python
@dataclass
class SampleResponse:
    trajectories: List[Trajectory]
    logprobs: Optional[List[List[float]]] = None
    prompt_logprobs: Optional[List[List[float]]] = None
    stop_reason: Optional[List[StopReason]] = None
```

- trajectories: List of generated trajectories
- logprobs: Log probabilities of generated tokens
- prompt_logprobs: Log probabilities of prompt tokens
- stop_reason: Stop reason, can be "length" (reached max length) or "stop" (encountered stop sequence)

Usage example:

```python
from twinkle.data_format import SamplingParams, SampleResponse
from twinkle.sampler import vLLMSampler

sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
params = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9)
response: SampleResponse = sampler.sample(trajectories, sampling_params=params, num_samples=4)

# Access generated trajectories
for traj in response.trajectories:
    print(traj.messages)
```
