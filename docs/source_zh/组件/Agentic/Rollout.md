# 多轮 Rollout

Rollout 模块提供统一的多轮对话引擎 `MultiTurnRollout`，每轮 assistant 可由本地 sampler、OpenAI 兼容 API，或在两者间动态选择的 callback 生成。

## Rollout 基类

```python
from abc import ABC, abstractmethod
from twinkle.data_format import Trajectory

class Rollout(ABC):

    @abstractmethod
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        raise NotImplementedError()
```

所有 rollout 接受轨迹列表并返回相同数量的轨迹，附带额外字段（`messages`、`turns`、`stop_reason`、`truncated`）。

## MultiTurnRollout

统一的多轮 rollout 引擎，支持本地 sampler、外部 API 和逐轮后端选择。每条轨迹在线程池中独立执行。

### 每轮循环

1. 将每个轨迹编码为带生成提示的 `InputFeature`
2. 调用 `response_callback(...)`，从 sampler 或 API 获取一个 `SampledSequence`
3. 检查终止条件：`stop_reason == 'length'`、无工具调用、或达到最大轮次
4. 通过 `ToolManager` 分发工具调用，追加工具响应
5. 计算桥接 token（工具轮次 + 生成提示），设置 `labels = -100`
6. 重复直到所有轨迹完成

```python
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle.data_format.sampling import SamplingParams

rollout = MultiTurnRollout(
    sampler=vllm_sampler,
    template=template,
    tool_manager=tool_manager,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=4096),
    max_turns=6,
    max_trajectory_tokens=8192,
    trace_dir='rollout_traces/',
)

# 运行 rollout
results = rollout(trajectories)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `sampler` | Sampler | 本地 sampler；两个后端同时存在时默认使用它。 |
| `api` | `API` | 可选的外部生成 API。 |
| `template` | `Template` | 必传；用于编码所有后端的输出。 |
| `response_callback` | `Callable` | 可选的逐轮后端选择器，返回 `SampledSequence`。 |
| `api_appended_as` | `str` | API 轮为 `demonstration`（仅 SFT）或 `context`（不训练）。 |
| `api_kwargs` | `Dict` | 传给每次 API 调用的请求字段。 |
| `tool_manager` | `ToolManager` | 工具分发器。也可以按调用传入。 |
| `sampling_params` | `SamplingParams` | 默认采样参数。 |
| `max_turns` | `int` | 每个轨迹的最大轮次（默认：6）。 |
| `max_trajectory_tokens` | `int` | 最大总 token 长度；超出则截断轨迹。 |
| `trace_dir` | `str` | 每轨迹 JSON 跟踪文件的目录。 |
| `trace_callback` | `Callable` | 决定是否存储轨迹跟踪。 |
| `success_callback` | `Callable` | 决定文件名前缀（`ok-` 或 `fail-`）。 |

### 输出字段

每个输出轨迹字典包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `List[Dict]` | 包含工具轮次的完整对话。 |
| `input_ids` | `List[int]` | 完整序列的 token ID。 |
| `labels` | `List[int]` | 训练标签（非可训练 token 为 `-100`）。 |
| `completion_mask` | `List[int]` | 由 policy 生成且具有 rollout log probability 的位置。 |
| `turns` | `int` | 执行的轮次数。 |
| `stop_reason` | `str` | `'stop'` / `'length'` |
| `truncated` | `bool` | 轨迹是否被截断（而非自行结束）：生成触及 `max_tokens`（`stop_reason='length'`）、达到轮次上限，或被长度上限丢弃。 |
| `logprobs` | `List` | 每 token 的对数概率（如有）。 |

### Ray 远程支持

`MultiTurnRollout` 使用 `@remote_class()` 装饰器，支持作为 Ray actor 透明部署：

```python
# rollout 可以作为 Ray 远程 actor 运行
rollout_actor = MultiTurnRollout.remote(sampler=sampler, template=template, ...)
results = ray.get(rollout_actor.__call__.remote(trajectories))
```

## API 与混合后端 Rollout

纯 API 模式使用同一个类，并仍需传入本地 template，以便将外部回复编码成训练侧一致的 token：

```python
from twinkle_agentic.protocol.openai import OpenAI
from twinkle_agentic.rollout import MultiTurnRollout

api = OpenAI(model='qwen3.5-32b', base_url='http://localhost:8000/v1', concurrency=8)
rollout = MultiTurnRollout(
    api,
    template=template,
    tool_manager=tool_manager,
    sampling_params=SamplingParams(temperature=0.7),
    max_turns=6,
    trace_dir='api_traces/',
)
results = rollout(trajectories)
```

同时传入 `sampler` 和 `api` 时，默认使用 sampler。传入 `response_callback` 可逐轮选择后端；callback 会收到两个后端，并必须返回一个 `SampledSequence`。API 轮没有 rollout log probability，因此 `api_appended_as='demonstration'` 会让它参与 SFT 但跳过 GRPO；使用 `'context'` 可让它完全不参与训练。

### 停止原因

| 原因 | 说明 |
|------|------|
| `stop` | 助手回复未包含工具调用（自然结束）。 |
| `length` | 生成达到 token 上限。 |
| `max_turns` | 达到工具轮次上限且没有 follow-up。 |
| `generation_error` | 外部端点未能返回有效响应。 |
