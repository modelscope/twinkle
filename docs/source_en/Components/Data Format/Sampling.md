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
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    num_samples: int = 1
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
params = SamplingParams(num_samples=4, logprobs=1, prompt_logprobs=0)
vllm_params = params.to_vllm()

# Convert to transformers' generate parameters
gen_kwargs = params.to_transformers(tokenizer=tokenizer)
```

## SampleResponse

`sampler.sample(...)` returns one `SampleResponse` per input prompt. Each response contains `sequences`, with one `SampledSequence` per generated completion:

```python
@dataclass
class SampledSequence:
    stop_reason: StopReason
    tokens: List[int]
    logprobs: Optional[List[List[Tuple[int, float]]]] = None
    decoded: str = None
    new_input_feature: InputFeature = None

@dataclass
class SampleResponse:
    sequences: Sequence[SampledSequence]
    prompt_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[List[Optional[float]]] = None
    topk_prompt_logprobs: Optional[List[Optional[List[Tuple[int, float]]]]] = None
```

`seq.tokens` contains generated token IDs; `seq.logprobs` contains `(token_id, logprob)` candidates for each token when requested. `seq.stop_reason` is `length`, `stop`, `abort`, or `error`. When using `vLLMSampler` with a configured template, `seq.new_input_feature` contains the prompt and completion features for training.

```python
from twinkle.data_format import SamplingParams

# Configure sampler and its template first; see the vLLMSampler example.
params = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9, num_samples=4, logprobs=1)
responses = sampler.sample(trajectories, sampling_params=params)
for response in responses:
    for seq in response.sequences:
        print(sampler.decode_response(seq.tokens))
```

See [vLLMSampler](../Sampler/vLLMSampler.md) for sampler initialization.
