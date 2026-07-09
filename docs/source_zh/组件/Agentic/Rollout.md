# 多轮 Rollout

Rollout 模块提供了用于 Agentic RLHF 训练的多轮对话 rollout 引擎。包含两种实现：用于批量 vLLM 采样的 `MultiTurnRollout` 和用于 OpenAI 兼容 API 端点的 `APIMultiTurnRollout`。

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

批量多轮 rollout 引擎，使用 vLLM 采样器进行生成。每轮中所有活跃轨迹通过单次批量采样调用并行处理，最大化吞吐量。

### 每轮循环

1. 将每个轨迹编码为带生成提示的 `InputFeature`
2. 批量调用 `sampler.sample(active_pifs)` —— 所有活跃轨迹并行
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
| `sampler` | Sampler | 用于批量生成的 vLLM 采样器实例。 |
| `template` | `Template` | 用于编码/解码的聊天模板。 |
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
| `turns` | `int` | 执行的轮次数。 |
| `stop_reason` | `str` | `'stop'` / `'length'` |
| `truncated` | `bool` | 轨迹是否被截断。 |
| `logprobs` | `List` | 每 token 的对数概率（如有）。 |

### Ray 远程支持

`MultiTurnRollout` 使用 `@remote_class()` 装饰器，支持作为 Ray actor 透明部署：

```python
# rollout 可以作为 Ray 远程 actor 运行
rollout_actor = MultiTurnRollout.remote(sampler=sampler, template=template, ...)
results = ray.get(rollout_actor.__call__.remote(trajectories))
```

## APIMultiTurnRollout

通过 OpenAI 兼容 chat-completions API 进行多轮 rollout。每个轨迹在线程池中独立运行，实现网络并发。

```python
from twinkle_agentic.rollout.api_multi_turn import APIMultiTurnRollout
from twinkle_agentic.protocol.openai import OpenAI

api = OpenAI(model='qwen3.5-32b', base_url='http://localhost:8000/v1')

rollout = APIMultiTurnRollout(
    api=api,
    tool_manager=tool_manager,
    sampling_params=SamplingParams(temperature=0.7),
    max_turns=6,
    concurrency=8,
    trace_dir='api_traces/',
)

results = rollout(trajectories)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `api` | `OpenAI` | OpenAI 兼容 API 客户端。 |
| `tool_manager` | `ToolManager` | 工具分发器（单个或按轨迹的列表）。 |
| `sampling_params` | `SamplingParams` | 默认采样参数。 |
| `max_turns` | `int` | 每轨迹最大轮次（默认：6）。 |
| `concurrency` | `int` | 并行 API 调用的线程池大小（默认：8）。 |
| `extra_body` | `Dict` | API 请求中附加的额外字段。 |
| `trace_dir` | `str` | 跟踪文件目录。 |

### 停止原因

| 原因 | 说明 |
|------|------|
| `stop` | 助手回复未包含工具调用（自然结束）。 |
| `length` | API 返回 `finish_reason='length'`（token 限制）。 |
| `max_turns` | 达到 `max_turns` 限制。 |
| `api_error` | API 调用或工具执行抛出异常。 |

## 选择建议

| 特性 | MultiTurnRollout | APIMultiTurnRollout |
|------|-----------------|---------------------|
| **后端** | vLLM 采样器（本地 GPU） | OpenAI 兼容 API |
| **训练集成** | 生成 `input_ids` / `labels` 用于 GRPO | 仅消息（用于数据收集） |
| **批处理** | GPU 级别批量并行 | 网络级别线程并发 |
| **用例** | 在线 RLHF 训练循环 | 离线数据生成 / 评估 |
