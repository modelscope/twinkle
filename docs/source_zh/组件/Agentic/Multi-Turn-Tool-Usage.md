# 多轮工具使用指南

本指南介绍如何在 Twinkle 中设置和运行带工具调用的多轮 Agentic rollout。

## 架构概览

Agentic rollout 管线由四个核心组件组成：

- **Tool** —— 实现特定能力（搜索、代码执行等）
- **ToolManager** —— 注册工具并分发 LLM 工具调用
- **Env**（可选）—— RL 环境，通过 `EnvTool` 暴露工具
- **Rollout** —— 驱动多轮对话循环

## 快速开始：基于 API 的 Rollout

使用 OpenAI 兼容 API 运行多轮工具使用 rollout 的最简方式：

```python
from twinkle_agentic.protocol.openai import OpenAI
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.rollout.api_multi_turn import APIMultiTurnRollout
from twinkle.data_format.sampling import SamplingParams

# 1. 定义工具
class WeatherTool(Tool):
    def __call__(self, tool_name, arguments):
        city = arguments.get('city', '未知')
        return f'{city}的天气：晴，25°C。'

    def tool_info(self):
        return {
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': '获取城市的当前天气。',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {'type': 'string', 'description': '城市名称。'},
                    },
                    'required': ['city'],
                },
            },
        }

# 2. 设置 ToolManager
manager = ToolManager([WeatherTool()])

# 3. 创建 API 客户端
api = OpenAI(model='qwen3.5-32b', base_url='http://localhost:8000/v1')

# 4. 创建 rollout
rollout = APIMultiTurnRollout(
    api=api,
    tool_manager=manager,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=2048),
    max_turns=6,
    concurrency=8,
)

# 5. 准备轨迹
trajectories = [
    {
        'messages': [
            {'role': 'user', 'content': '北京今天天气怎么样？'},
        ],
    },
]

# 6. 运行 rollout
results = rollout(trajectories)
for r in results:
    print(f"轮次: {r['turns']}, 停止原因: {r['stop_reason']}")
    for msg in r['messages']:
        print(f"  [{msg['role']}] {msg.get('content', '')[:100]}")
```

## 训练集成：基于 vLLM 的 Rollout

用于 RLHF 训练时，使用 `MultiTurnRollout`，它会生成 `input_ids` 和 `labels`：

```python
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle.data_format.sampling import SamplingParams

rollout = MultiTurnRollout(
    sampler=vllm_sampler,           # vLLMSampler 实例
    template=template,               # 聊天模板
    tool_manager=manager,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=4096),
    max_turns=6,
    max_trajectory_tokens=8192,
    trace_dir='rollout_traces/',
)

# 在 GRPO 训练循环中
results = rollout(batch_trajectories)
# results 包含 input_ids、labels、logprobs 用于训练
```

## 将环境用作工具

将 RL 环境桥接到工具管线中：

```python
from twinkle_agentic.envs.base import Env, StepResult
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager

# 定义环境
class CodeEnv(Env):
    def step(self, tool_name, arguments):
        code = arguments.get('code', '')
        # 在沙箱中执行代码
        result = execute_in_sandbox(code)
        return StepResult(observation=result, reward=1.0, done=False)

    def tools(self):
        return [{
            'type': 'function',
            'function': {
                'name': 'run_python',
                'description': '执行 Python 代码。',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'code': {'type': 'string'},
                    },
                    'required': ['code'],
                },
            },
        }]

# 桥接 Env -> Tool -> ToolManager
env = CodeEnv()
env_tools = EnvTool.from_env(env)
manager = ToolManager(env_tools)

# 照常在 rollout 中使用 manager
rollout = APIMultiTurnRollout(api=api, tool_manager=manager, max_turns=10)
```

## 使用 OpenEnv 环境

连接远程 OpenEnv WebSocket 服务器：

```python
from twinkle_agentic.envs.openenv import OpenEnv
from twinkle_agentic.envs.env_tool import EnvTool

env = OpenEnv(
    base_url='http://localhost:8000',
    env_cls='coding_env.CodingEnv',
    tool_schema=[{
        'type': 'function',
        'function': {
            'name': 'submit',
            'description': '提交代码解决方案。',
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {'type': 'string'},
                },
            },
        },
    }],
)

env.reset()
env_tools = EnvTool.from_env(env)
manager = ToolManager(env_tools)
```

## 每轨迹独立 ToolManager

当每个轨迹需要独立工具集时（例如，轨迹绑定的状态）：

```python
# 创建每轨迹的 manager
managers = []
for traj in trajectories:
    env = create_env_for(traj)
    env_tools = EnvTool.from_env(env)
    managers.append(ToolManager(env_tools))

# 传入列表（与轨迹 1:1 对齐）
results = rollout(trajectories, tool_manager=managers)
```

## 跟踪调试

两种 rollout 实现都支持跟踪文件输出用于调试：

```python
rollout = APIMultiTurnRollout(
    api=api,
    tool_manager=manager,
    trace_dir='traces/',
    trace_callback=lambda t: t['turns'] > 1,    # 仅存储多轮对话
    success_callback=lambda t: t.get('stop_reason') == 'stop',
)
```

跟踪文件以 `{step}-{ok|fail}-{id}.json` 格式保存，包含完整对话和元数据。
