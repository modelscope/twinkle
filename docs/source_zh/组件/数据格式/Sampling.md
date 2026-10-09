# 采样输出

采样输出是用于表示采样过程的输入参数和返回结果的数据格式。

## SamplingParams

采样参数用于控制模型的采样行为。

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

- max_tokens: 生成的最大 token 数量
- seed: 随机种子
- stop: 停止序列,可以是字符串、字符串序列或 token id 序列
- temperature: 温度参数,控制采样的随机性。0 表示贪心采样
- top_k: Top-K 采样参数,-1 表示不使用
- top_p: Top-P (nucleus) 采样参数
- repetition_penalty: 重复惩罚系数

### 转换方法

SamplingParams 提供了转换方法来适配不同的推理引擎:

```python
# 转换为 vLLM 的 SamplingParams
params = SamplingParams(num_samples=4, logprobs=1, prompt_logprobs=0)
vllm_params = params.to_vllm()

# 转换为 transformers 的 generate 参数
gen_kwargs = params.to_transformers(tokenizer=tokenizer)
```

## SampleResponse

`sampler.sample(...)` 为每个输入 prompt 返回一个 `SampleResponse`。每个响应的 `sequences` 包含多个生成结果，每个结果是 `SampledSequence`：

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

`seq.tokens` 是生成的 token ID；请求 logprobs 后，`seq.logprobs` 保存每个 token 的 `(token_id, logprob)` 候选列表。`seq.stop_reason` 为 `length`、`stop`、`abort` 或 `error`。设置 `return_encoded=True` 后，`seq.new_input_feature` 包含可用于训练的输入特征。

```python
from twinkle.data_format import SamplingParams

# 先配置 sampler 及其模板，参见 vLLMSampler 示例。
params = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9, num_samples=4, logprobs=1)
responses = sampler.sample(trajectories, sampling_params=params, return_encoded=True)
for response in responses:
    for seq in response.sequences:
        print(sampler.decode_response(seq.tokens))
```

采样器初始化见 [vLLMSampler](../采样器/vLLMSampler.md)。
