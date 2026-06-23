# 执行环境（Envs）

Envs 模块提供了用于 Agentic 训练的 RL 执行环境抽象。环境可以在多轮 rollout 中交互式参与，也可以批量评估已完成的轨迹。

## Env 基类

```python
from twinkle_agentic.envs.base import Env, StepResult

class Env(ABC):

    def reset(self, trajectory=None) -> StepResult:
        """重置环境，开始新一轮。"""

    @abstractmethod
    def step(self, tool_name: str, arguments: dict) -> StepResult:
        """执行单个动作，返回观测 + 奖励 + 完成标志。"""

    def tools(self) -> List[ToolInfo]:
        """返回此环境中可用的工具定义。"""

    def evaluate(self, trajectories, **kwargs) -> List[float]:
        """批量评估已完成的轨迹，返回奖励列表。"""

    def close(self) -> None:
        """释放资源。"""
```

### StepResult

```python
@dataclass
class StepResult:
    observation: str = ''    # 动作执行后的环境观测
    reward: float = 0.0      # 此步骤的标量奖励
    done: bool = False        # 是否终止
    info: Dict[str, Any] = field(default_factory=dict)  # 额外元数据
```

### 两种使用模式

1. **交互模式**（多轮 rollout）—— 逐步执行：

```python
env = MyEnv()
env.reset(trajectory)
result = env.step('search', {'query': 'Python'})
# ... 重复直到 result.done
```

2. **批量评估模式** —— 评估已完成的轨迹：

```python
rewards = env.evaluate(completed_trajectories)
```

## EnvTool

`EnvTool` 将 `Env` 包装为 `Tool`，连接环境与 `ToolManager` 和 `MultiTurnRollout`。

```python
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager

env = MyEnv()

# 为环境中定义的每个工具创建一个 EnvTool
env_tools = EnvTool.from_env(env)

# 注册到 ToolManager
manager = ToolManager(env_tools)
```

### 核心特性

| 特性 | 说明 |
|------|------|
| `from_env(env)` | 工厂方法：为 `env.tools()` 中的每个工具创建一个 `EnvTool`。 |
| `last_result` | 存储最近一次 `StepResult` 供调用方检查。 |
| `done` | 属性：最后一步是否终止了回合。 |
| `episode_reward` | 属性：来自 `info['episode_reward']` 的累计奖励。 |

### 手动构造

```python
env_tool = EnvTool(
    env=my_env,
    tool_name='execute_code',
    description='在沙箱中执行 Python 代码。',
    parameters={
        'type': 'object',
        'properties': {
            'code': {'type': 'string', 'description': '要执行的 Python 代码。'},
        },
        'required': ['code'],
    },
)
```

## OpenEnv

`OpenEnv` 将基于 WebSocket 的 [OpenEnv](https://github.com/OpenEnv) 环境服务器适配为同步的 Twinkle `Env`。

```python
from twinkle_agentic.envs.openenv import OpenEnv

env = OpenEnv(
    base_url='http://localhost:8000',
    env_cls='coding_env.CodingEnv',      # 可选的类型化客户端
    env_kwargs={'message_timeout_s': 30},
    tool_schema=[...],                    # 可选的工具定义
)
```

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `base_url` | `str` | 运行中的 OpenEnv 服务器 URL。 |
| `env_cls` | `str` 或 class | 类型化客户端的点分导入路径或类。`None` 使用 `GenericEnvClient`。 |
| `env_kwargs` | `Dict` | 传递给客户端构造函数的额外参数。 |
| `tool_schema` | `List[ToolInfo]` | 通过 `tools()` 暴露的工具定义。 |
| `action_mapper` | `Callable` | 自定义函数，将 `(tool_name, args)` 映射为发送给服务器的动作字典。 |

### 与 Rollout 集成使用

```python
from twinkle_agentic.envs.openenv import OpenEnv
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.rollout.api_multi_turn import APIMultiTurnRollout

# 设置环境
env = OpenEnv(base_url='http://localhost:8000', tool_schema=[...])
env.reset()

# 桥接到 ToolManager
env_tools = EnvTool.from_env(env)
manager = ToolManager(env_tools)

# 在 rollout 中使用
rollout = APIMultiTurnRollout(api=api, tool_manager=manager, max_turns=10)
results = rollout(trajectories)
```

### 实现自定义环境

```python
from twinkle_agentic.envs.base import Env, StepResult

class CodeExecutionEnv(Env):

    def reset(self, trajectory=None):
        self._sandbox = create_sandbox()
        return StepResult(observation='沙箱已就绪。')

    def step(self, tool_name, arguments):
        code = arguments.get('code', '')
        output = self._sandbox.run(code)
        return StepResult(
            observation=output,
            reward=1.0 if 'error' not in output.lower() else 0.0,
            done=False,
        )

    def tools(self):
        return [{
            'type': 'function',
            'function': {
                'name': 'execute_code',
                'description': '运行 Python 代码。',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'code': {'type': 'string'},
                    },
                },
            },
        }]

    def close(self):
        self._sandbox.cleanup()
```
