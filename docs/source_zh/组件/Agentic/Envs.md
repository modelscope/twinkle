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
| `episode_reward` | 属性：来自 `info['episode_reward']` 的累计奖励，缺失时回退到最后一步的 `reward`。 |

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

## OpenEnv：两种接入模式

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) 的环境包同时提供「Environment 实现」和「EnvClient 客户端」，因此 Twinkle 提供两个适配器，对应两种截然不同的部署形态：

| | **嵌入式** `OpenEnv` | **服务端** `OpenEnvClient` |
|---|---|---|
| 环境运行位置 | 训练进程内（直接实例化 Environment） | 独立的 OpenEnv 服务进程/容器 |
| 通信 | 无（本地函数调用） | WebSocket 长连接（一个连接 = 一个 session） |
| 隔离性 | 无，与训练进程共享内存和 GPU 节点 | 进程级/容器级，可部署在完全独立的机器 |
| 依赖 | 环境包需装在训练节点 | 环境包只需装在环境节点 |
| 扩展方式 | `EnvPool` 按 Ray worker 分片 | 服务端自身的并发 session + 多副本 |
| 适用场景 | 纯计算型轻量环境（棋类、文本游戏） | 代码执行、需要隔离或需独立扩缩容的环境 |

> **不要**把 `OpenEnvClient` 放进 `EnvPool`：session 的生命周期在服务端，用 Ray 再分片一次不会带来任何收益，只会多一层 RPC。

### 模式一：嵌入式 `OpenEnv`

绕过 OpenEnv 的 FastAPI 服务，在训练进程内直接构造 Environment，零网络开销。

```python
from twinkle_agentic.envs.openenv import OpenEnv

env = OpenEnv(
    env_name='openspiel_env',                  # 环境包名，自动发现 Environment / Action 类
    env_kwargs={'game_name': 'blackjack'},     # 传给 Environment 构造函数
)
result = env.reset()
result = env.step('play', {'action': 'hit'})
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `env_name` | `str` | OpenEnv 环境包名（如 `'coding_env'`）。自动从 `<env_name>.server` 中发现 `*Environment` 类，并从包的 `__all__` 中发现 `*Action` 类。 |
| `env_cls` | `str` 或 class | 显式指定 Environment 类（`'module:ClassName'`），与 `env_name` 二选一。 |
| `env_kwargs` | `Dict` | 传给 Environment 构造函数的参数。 |
| `action_cls` | `str` 或 class | 显式指定 Action 类；省略时从 `env_name` 自动发现。 |
| `action_mapper` | `Callable` | `(tool_name, arguments) -> action`。默认把工具参数直接作为 Action 的字段。 |

需注意嵌入式 `OpenEnv` 并未实现 `tools()`，因此继承基类的空列表，`EnvTool.from_env(env)` 会走回退分支，只生成一个通用的 `env_action` 工具。当模型需要看到真实的动作名时，应给 `EnvTool` 传入显式 schema，或子类化重载 `tools()`。

### 模式二：服务端 `OpenEnvClient`

先在环境节点上把 OpenEnv 环境跑成一个普通 HTTP/WebSocket 服务（不需要 Docker）：

```bash
pip install openenv
pip install -e /path/to/OpenEnv/envs/coding_env
uvicorn coding_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

然后在训练侧为每条轨迹创建一个客户端。每个实例都持有自己的 WebSocket session，服务端为它维护一个独立的 Environment 实例：

```python
from twinkle_agentic.envs.openenv import OpenEnvClient

env = OpenEnvClient(
    env_name='coding_env',                 # 自动发现 EnvClient 子类 + Action 类
    base_url='http://10.0.0.5:8000',       # 也可用 OPENENV_BASE_URL 环境变量
    message_timeout_s=120,                 # 环境要跑长耗时代码时调大
)
env.reset()
result = env.step('run_python', {'code': 'print(1 + 1)'})
print(result.observation)                  # '2'
env.close()
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `env_name` | `str` | OpenEnv 环境包名，从中自动发现 `EnvClient` 子类与 `*Action` 类。 |
| `env_cls` | `str` 或 class | 显式指定客户端类（`'module:ClassName'`），与 `env_name` 二选一。 |
| `base_url` | `str` | 服务地址，`http(s)://` 或 `ws(s)://` 均可（自动转换）。缺省读取 `OPENENV_BASE_URL`。也可以填负载均衡器地址。 |
| `action_cls` | `str` 或 class | Action 类；省略时自动发现。 |
| `action_mapper` | `Callable` | `(tool_name, arguments) -> action`，返回 Action 实例或字段字典。 |
| `tools` | `List[ToolInfo]` | 暴露给模型的工具 schema。默认是单个 `run_python(code)`，与 OpenEnv 代码类环境对齐。 |
| `reset_kwargs` | `Dict` | 转发给服务端 `reset()` 的参数（如 `repl_env` 的 `task_prompt` / `expected_answer`）。也可以按 episode 修改 `env.reset_kwargs` 属性。 |
| `connect_timeout_s` | `float` | WebSocket 连接超时，默认 10s。 |
| `message_timeout_s` | `float` | 单条消息超时，默认 120s。 |
| `client_kwargs` | `Dict` | 传给 OpenEnv 客户端构造函数的额外参数。 |

补充能力：

- `register_tool(tool_info, handler)`：注册一个**在客户端本地执行**的工具，不发往服务端。典型用途是 `submit_solution` 这类记账工具——把模型的答案存到 env 上，供训练循环后续打分。同名工具会覆盖默认工具。
- `execute(action)`：直接发送动作并返回服务端**原始** `StepResult`，可以读取 `exit_code` 等结构化字段。训练循环用它在同一个 session 里追加执行单元测试。
- `episode_reward` / `last_result` / `client`：累计奖励、上一次原始结果、底层同步客户端。

**容量与并发**：`OpenEnvClient` 内部调用 OpenEnv 客户端的 `.sync()`，每个实例拥有独立的后台事件循环，因此可以放在线程池里并发 reset/step。但服务端必须能容纳全部并发 session：环境类需声明 `SUPPORTS_CONCURRENT_SESSIONS = True`，且 `create_app(..., max_concurrent_envs=N)` 要够大，否则多出的连接会被拒绝。由于该上限按 worker 进程生效，总容量为 `workers x max_concurrent_envs`。OpenEnv 自带的 `coding_env` 将 `SUPPORTS_CONCURRENT_SESSIONS` 留在保守的默认值，需要子类化后打开（否则 `create_app` 会在 `max_concurrent_envs > 1` 时抛出 `ConcurrencyConfigurationError`）；`cookbook/rl/envs/openenv_server/server_app.py` 给出了完整写法。

### 与 Rollout 集成使用

两种模式的下游用法一致：

```python
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.rollout import MultiTurnRollout

env.reset()

# 桥接到 ToolManager
env_tools = EnvTool.from_env(env)
manager = ToolManager(env_tools)

# 在 rollout 中使用
rollout = MultiTurnRollout(api=api, template=template, tool_manager=manager, max_turns=10)
results = rollout(trajectories)
```

端到端的多轮 GRPO 训练示例见[Agentic RL 部署与训练](../../使用指引/Agentic%20RL部署与训练.md)。

## AgentEnv：Firecracker microVM 沙箱

`AgentEnv` 通过 E2B SDK 操作远端沙箱，既可连接 **E2B 官方云服务**，也可连接自建的 [AgentENV](https://github.com/kvcache-ai/AgentENV)。官方服务不要求你部署 AgentENV gateway，也不要求客户端具备 KVM。`reset()` 创建独立沙箱，`step()` 执行工具，`close()` 释放沙箱；命令、文件和子进程都在远端执行。

与 OpenEnv 适配器的差异：

| | `OpenEnvClient` | `AgentEnv` |
|---|---|---|
| 隔离 | 取决于环境服务的部署 | 由沙箱 provider 提供 |
| 执行器 | 取决于具体环境，如游戏或代码执行器 | 沙箱中的真实进程 |
| 单环境内存 | 取决于环境实现 | 取决于 provider / 模板配置 |
| 文件 / pip / 子进程 | 取决于环境实现和暴露的工具 | SDK 提供这些能力 |
| 传输 | 长连接 WebSocket | HTTP（E2B SDK）|
| 前置条件 | 一个 OpenEnv 服务 | 官方 E2B 账户和密钥，或自建 AgentENV 服务和模板 |

OpenEnv 的 `coding_env` 底层是 smolagents 的 `LocalPythonExecutor`，**一个 AST 解释器，而非操作系统级沙箱**。它对 `decorator_list` 没有任何处理，**装饰器会被静默忽略**——`@patch` 不生效、测试不报错，reward 产出一个形式正常的错误数值。这类隐形错误比崩溃难以定位。它适用于约束「仅允许 import 白名单」，不适用于执行对抗性代码。当测试依赖装饰器，或模型需要写文件、装包、开子进程时，使用 `AgentEnv`。

### 连接 E2B 官方服务

在仓库根目录选择 `e2b` 可选依赖，它会一起安装已验证版本的 E2B SDK 和读取 `.env` 所需的 `python-dotenv`：

```bash
pip install -e '.[e2b]'
```

在运行目录的 `.env` 中设置密钥，文件不要提交到 Git：

```dotenv
E2B_API_KEY=your_e2b_api_key
```

使用官方服务时，保持 `E2B_API_URL`、`E2B_SANDBOX_URL`、`E2B_DOMAIN` 未设置，也不要手动传入 `api_url` / `sandbox_url`。让 SDK 选择官方控制面和沙箱地址。

先单独检查“创建 → 执行 → 释放”，这一步不需要 Twinkle 训练服务或 GPU：

```python
from dotenv import load_dotenv
from twinkle_agentic.envs import AgentEnv

load_dotenv('.env')
env = AgentEnv(template='base', workspace='/home/user/twinkle', sandbox_timeout=180)
try:
    env.reset()
    code, output = env.run_script('print("hello from E2B")')
    if code != 0:
        raise RuntimeError(output)
    print(output.strip())
finally:
    env.close()
```

官方 `base` 模板以普通用户运行，示例使用可写的 `/home/user/twinkle`；不要直接沿用自建 AgentENV 示例的 `/workspace`。初始化目录或 `setup_commands` 失败会抛出异常，调用方仍负责关闭已创建的沙箱。

连通后，可运行 [客户端指引中的 E2B Blackjack 多轮训练示例](../../使用指引/服务端和客户端/Twinkle客户端.md)。它为每条轨迹创建一个 sandbox，游戏程序在 sandbox 中运行；训练和策略发布仍由 Twinkle 服务执行。

### 连接自建 AgentENV

这条路径才需要预先部署 AgentENV、构建模板（例如 `aenv pull ubuntu:22.04 --name my-env`），并安装兼容的 E2B SDK。模板名、端点和鉴权以你自己的部署为准；不要将这些配置套用到 E2B 官方云。

```python
from twinkle_agentic.envs import AgentEnv

env = AgentEnv(
    template='my-env',                     # 预先用 aenv CLI 构建的模板
    api_url='http://10.0.0.5:8000',        # server 或 gateway；缺省读 E2B_API_URL
    sandbox_timeout=600,                   # 需大于单个 episode 加测试回放的耗时
)
try:
    env.reset()                            # 启动一个新沙箱，由 scheduler 选节点
    result = env.step('run_command', {'command': 'python -c "print(1 + 1)"'})
    print(result.observation)              # '2'
finally:
    env.close()
```

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `template` | `str` | provider 的模板名/ID，必填。官方示例使用 `base`；自建 AgentENV 使用自行构建的模板。 |
| `api_url` | `str` | 自建 server / gateway 的基础 URL。省略时交给 SDK 读取环境变量或使用官方默认地址。 |
| `api_key` | `str` | 优先使用显式参数，再读 `E2B_API_KEY`。SDK 适配器保留 `'dummy'` 回退，但官方 E2B 必须使用真实有效密钥；自建部署的鉴权由其配置决定。 |
| `sandbox_url` | `str` | 数据面地址，用于沙箱 gateway 与 API 不在同一主机的部署。缺省等于 `api_url`。 |
| `sandbox_timeout` | `int` | 传给 provider 的沙箱超时秒数，默认 300。暂停或终止行为由 provider 决定，不应假设都能自动恢复；程序应主动 `close()`。 |
| `command_timeout` | `int` | 单条命令超时（秒），默认 120。 |
| `setup_commands` | `List[str]` | 每次 `reset` 后执行一次的命令，输出作为 reset 的 observation。 |
| `sandbox_envs` | `Dict[str, str]` | 注入沙箱的环境变量。 |
| `metadata` | `Dict[str, str]` | 沙箱元数据，在 list 接口中可见——适合标记运行名或轨迹 id。 |
| `refresh_timeout` | `bool` | 每次 step 后延长超时，默认 `True`，避免长 episode 中途到期。 |
| `include_default_tools` | `bool` | 是否暴露内置的 `run_command` / `write_file` / `read_file`，默认 `True`。 |

### 注册任务工具

AgentENV 服务端本身不定义工具，仅提供能力原语（任意命令执行、文件读写、端口代理）。因此工具完全是客户端概念，有两种注册方式：

```python
# 1. shell 命令模板，用工具参数格式化
env.register_command_tool(
    {'type': 'function', 'function': {
        'name': 'run_tests',
        'description': '运行任务测试集。',
        'parameters': {'type': 'object',
                       'properties': {'test_file': {'type': 'string'}},
                       'required': ['test_file']}}},
    'cd /workspace && pytest {test_file} -x -q')

# 2. 任意 Python handler：handler(env, arguments) -> str
def _submit(env, arguments):
    env.submitted_code = arguments.get('code', '')
    return 'Solution submitted.'

env.register_tool(submit_schema, _submit)      # 两者均返回 self，可链式调用
```

设置 `include_default_tools=False` 可隐藏内置工具，使动作空间与任务精确对齐，从而保证奖励归因的清晰。注册的工具名与内置工具冲突时会覆盖后者。

其他能力：

- `run_command(arguments)`：公开方法，供自定义 handler 复用。非零退出码**不会抛异常**，stdout、stderr 与退出码会被格式化进 observation，让模型能针对失败做出反应。
- `sandbox` / `sandbox_id`：底层 E2B 句柄（可访问 PTY、文件 watch 等能力）与当前沙箱 id。
- observation 默认截断到 32K 字符，避免失控的命令输出冲爆上下文。

**错误处理与奖励**：基础 `AgentEnv.step` 将常规工具错误作为观察返回，让模型有机会恢复；初始化失败则直接抛出。远端执行是否结束未知时，`ensure_idle()` 拒绝复用资源。任务环境还应区分模型的无效操作与基础设施失败：E2B Blackjack 示例会让游戏进程失败中止训练，不会把它当作零奖励。`AgentEnv.evaluate` 默认返回 0；任务可以通过 `StepResult.reward` 提供环境奖励，或另行评分。

**并发**：`AgentEnv` 不使用 `@remote_class`；沙箱的调度由所连接的 provider 服务管理。每条并发轨迹使用独立实例，**不要**放入只服务嵌入式 OpenEnv 的 `EnvPool`。`reset()` 是同步网络调用，初始化耗时应计入控制线程开销；不要把它计作模型生成时间。

容量取决于调度器的在途/缓冲预算及 provider 的并发、内存和费用额度。E2B 流式示例默认预算为 `BUFFER_GROUPS × NUM_GENERATIONS = 12` 个 sandbox；自建 AgentENV 则还需核对模板的内存配置。自建部署步骤见[Agentic RL 部署与训练](../../使用指引/Agentic%20RL部署与训练.md)。

## EnvPool：分布式环境池

`EnvPool` 是一个 `@remote_class`，把 `pool_size` 个**嵌入式** `OpenEnv` 实例按 Ray worker 分片。每个 worker 管理 `ceil(pool_size / world_size)` 个槽位（最后一个分片会被裁到 `pool_size`，因此可能更少），`reset`/`step` 通过 `remote_function` 自动路由到持有该槽位的 worker。

```python
from twinkle_agentic.envs.openenv import EnvPool

pool = EnvPool(
    pool_size=64,
    device_mesh=mesh,
    env_kwargs={'env_name': 'openspiel_env', 'env_kwargs': {'game_name': 'blackjack'}},
)

# 每个槽位包装成一个标准 Env，可直接交给 EnvTool / ToolManager
envs = pool.get_adapters(64)
env_tools = EnvTool.from_env(envs[0])
```

| 方法 | 说明 |
|------|------|
| `reset(idx)` / `step(idx, tool_name, arguments)` | 操作单个槽位。 |
| `reset_batch(indices)` / `step_batch(indices, tool_names, arguments_list)` | 批量操作，一次 RPC 覆盖多个槽位，按 `indices` 顺序返回。 |
| `get_adapters(n)` | 在 driver 侧把前 `n` 个槽位包装为 `EnvPoolAdapter`（标准 `Env`）。`n > pool_size` 时抛异常。 |
| `close()` | 关闭全部环境。 |

`EnvPoolAdapter` 实现标准 `Env` 接口，把 `reset`/`step` 代理到对应 worker；`step` 出错时返回 `done=True` 并把错误写入 `info['error']`，避免单个环境异常拖垮整批 rollout。它自身的 `close()` 是空操作——资源需通过池的 `close()` 释放。

## 实现自定义环境

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
