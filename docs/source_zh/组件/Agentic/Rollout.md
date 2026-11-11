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

## 流式交接与权重发布（v5）

`twinkle_agentic.async_rl.StreamingDriver` 负责准入、版本记账和已完成记录的交接。训练配方负责 optimizer batch 大小和发布时机，通过自己的回调请求发布；driver 不直接传输权重。

`buffer_depth` 限制在途轨迹容量，`max_rows_per_pull` 独立限制每次训练交接大小。例如容量 128、取数上限 64，可以预留下一批生成工作，同时让配方在每次更新后发布权重。只扩大容量而不限制取数，可能把多个已完成批次合并到一次交接。

- `group_pull=True`：上限必须是 `lcm(need_rows, num_generations)` 的正整数倍。完整且完成评分的组保持连续，未完成或等待评分的组继续留在缓冲区。
- `group_pull=False`：上限必须是 `need_rows` 的正整数倍。
- `max_rows_per_pull=None`：保留原有的不设上限、按完整批次取数的行为。
- 自定义 `assembly_ready` 自行负责选择策略，不能同时传入该上限。

该上限约束正常完整批次的交接，有限数据流的尾批处理仍是独立路径，行为不变。容量和取数上限不会放宽策略版本校验。

存在活跃 partial rollout 时，原地发布必须先暂停生成，完成权重同步后再恢复。`abort_all_inflight` 只发出中断，不关闭重试入口，不能单独作为安全发布屏障。暂停或传输失败必须向上传播，不能无条件恢复生成。自适应 `ClusterAdapter` 继续拥有其完整发布事务。

中断、暂停、恢复和生成状态查询显式标记为 `control=True`。即使 sampler handle 被序列化并转发到 rollout actor、缺少原始 worker-pool 注册表，控制指令仍能越过等待生成结束的屏障。普通生命周期和权重操作继续保留屏障。
