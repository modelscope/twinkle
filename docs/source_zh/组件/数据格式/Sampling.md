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
vllm_params = params.to_vllm(num_samples=4, logprobs=True, prompt_logprobs=0)

# 转换为 transformers 的 generate 参数
gen_kwargs = params.to_transformers(tokenizer=tokenizer)
```

## SampleResponse

采样响应是采样器返回的结果数据结构。

```python
@dataclass
class SampleResponse:
    trajectories: List[Trajectory]
    logprobs: Optional[List[List[float]]] = None
    prompt_logprobs: Optional[List[List[float]]] = None
    stop_reason: Optional[List[StopReason]] = None
```

- trajectories: 采样生成的轨迹列表
- logprobs: 生成 token 的对数概率
- prompt_logprobs: prompt token 的对数概率
- stop_reason: 停止原因,可以是 "length" (达到最大长度) 或 "stop" (遇到停止序列)

使用示例:

```python
from twinkle.data_format import SamplingParams, SampleResponse
from twinkle.sampler import vLLMSampler

sampler = vLLMSampler(model_id='Qwen/Qwen2.5-7B-Instruct')
params = SamplingParams(max_tokens=512, temperature=0.7, top_p=0.9)
response: SampleResponse = sampler.sample(trajectories, sampling_params=params, num_samples=4)

# 访问生成的轨迹
for traj in response.trajectories:
    print(traj.messages)
```
