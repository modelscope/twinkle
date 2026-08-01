# Environments (Envs)

The Envs module provides an RL execution environment abstraction for agentic training. Environments can participate in multi-turn rollouts interactively or evaluate completed trajectories in batch.

## Env Base Class

```python
from twinkle_agentic.envs.base import Env, StepResult

class Env(ABC):

    def reset(self, trajectory=None) -> StepResult:
        """Reset for a new episode."""

    @abstractmethod
    def step(self, tool_name: str, arguments: dict) -> StepResult:
        """Execute a single action, return observation + reward + done."""

    def tools(self) -> List[ToolInfo]:
        """Return tool definitions available in this environment."""

    def evaluate(self, trajectories, **kwargs) -> List[float]:
        """Batch-evaluate completed trajectories, return rewards."""

    def close(self) -> None:
        """Release resources."""
```

### StepResult

```python
@dataclass
class StepResult:
    observation: str = ''    # Environment observation after the action
    reward: float = 0.0      # Scalar reward for this step
    done: bool = False        # Whether the episode is terminated
    info: Dict[str, Any] = field(default_factory=dict)  # Extra metadata
```

### Two Usage Modes

1. **Interactive mode** (multi-turn rollout) — step-by-step execution:

```python
env = MyEnv()
env.reset(trajectory)
result = env.step('search', {'query': 'Python'})
# ... repeat until result.done
```

2. **Batch evaluation mode** — evaluate completed trajectories:

```python
rewards = env.evaluate(completed_trajectories)
```

## EnvTool

`EnvTool` wraps an `Env` as a `Tool`, bridging the environment with `ToolManager` and `MultiTurnRollout`.

```python
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager

env = MyEnv()

# Create one EnvTool per tool defined in the environment
env_tools = EnvTool.from_env(env)

# Register into ToolManager
manager = ToolManager(env_tools)
```

### Key Features

| Feature | Description |
|---------|-------------|
| `from_env(env)` | Factory: creates one `EnvTool` per tool in `env.tools()`. |
| `last_result` | Stores the most recent `StepResult` for inspection. |
| `done` | Property: whether the last step terminated the episode. |
| `episode_reward` | Property: cumulative reward from `info['episode_reward']`. |

### Manual Construction

```python
env_tool = EnvTool(
    env=my_env,
    tool_name='execute_code',
    description='Execute Python code in a sandbox.',
    parameters={
        'type': 'object',
        'properties': {
            'code': {'type': 'string', 'description': 'Python code to execute.'},
        },
        'required': ['code'],
    },
)
```

## OpenEnv: Two Integration Modes

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment package ships both an `Environment` implementation and an `EnvClient`, so Twinkle provides two adapters for two very different deployment shapes:

| | **Embedded** `OpenEnv` | **Server** `OpenEnvClient` |
|---|---|---|
| Where the env runs | Inside the training process (Environment instantiated directly) | A separate OpenEnv service process / container |
| Transport | None (local function calls) | Persistent WebSocket (one connection = one session) |
| Isolation | None; shares memory and the GPU node with training | Process / container level; can live on a completely separate machine |
| Dependencies | Env package must be installed on the training node | Env package only needed on the environment node |
| Scaling | `EnvPool` shards across Ray workers | The server's own concurrent sessions + replicas |
| Best for | Pure-compute lightweight envs (board games, text games) | Code execution, anything needing isolation or independent scaling |

> Do **not** wrap `OpenEnvClient` in `EnvPool`: the session's lifetime is owned by the server, so sharding it again through Ray buys nothing and only adds an RPC hop.

### Mode 1: Embedded `OpenEnv`

Bypasses OpenEnv's FastAPI server and constructs the Environment in-process — zero network overhead.

```python
from twinkle_agentic.envs.openenv import OpenEnv

env = OpenEnv(
    env_name='openspiel_env',                  # Package name; Environment / Action classes auto-discovered
    env_kwargs={'game_name': 'blackjack'},     # Passed to the Environment constructor
)
result = env.reset()
result = env.step('play', {'action': 'hit'})
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `env_name` | `str` | OpenEnv package name (e.g. `'coding_env'`). The `*Environment` class is discovered from `<env_name>.server` and the `*Action` class from the package's `__all__`. |
| `env_cls` | `str` or class | Explicit Environment class (`'module:ClassName'`), used instead of `env_name`. |
| `env_kwargs` | `Dict` | Kwargs for the Environment constructor. |
| `action_cls` | `str` or class | Explicit Action class; auto-discovered from `env_name` when omitted. |
| `action_mapper` | `Callable` | `(tool_name, arguments) -> action`. Defaults to passing the tool arguments as Action fields. |

### Mode 2: Server `OpenEnvClient`

First, run the OpenEnv environment as an ordinary HTTP/WebSocket service on the environment host (no Docker required):

```bash
pip install openenv
pip install -e /path/to/OpenEnv/envs/coding_env
uvicorn coding_env.server.app:app --host 0.0.0.0 --port 8000 --workers 4
```

Then create one client per trajectory on the training side. Each instance owns its own WebSocket session, and the server keeps a dedicated Environment instance for it:

```python
from twinkle_agentic.envs.openenv import OpenEnvClient

env = OpenEnvClient(
    env_name='coding_env',                 # EnvClient subclass + Action class auto-discovered
    base_url='http://10.0.0.5:8000',       # Or set OPENENV_BASE_URL
    message_timeout_s=120,                 # Raise it when the env runs long-executing code
)
env.reset()
result = env.step('run_python', {'code': 'print(1 + 1)'})
print(result.observation)                  # '2'
env.close()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `env_name` | `str` | OpenEnv package name; the `EnvClient` subclass and `*Action` class are auto-discovered from it. |
| `env_cls` | `str` or class | Explicit client class (`'module:ClassName'`), used instead of `env_name`. |
| `base_url` | `str` | Server address, `http(s)://` or `ws(s)://` (converted automatically). Falls back to `OPENENV_BASE_URL`. A load-balancer address works here too. |
| `action_cls` | `str` or class | Action class; auto-discovered when omitted. |
| `action_mapper` | `Callable` | `(tool_name, arguments) -> action`, returning an Action instance or a dict of fields. |
| `tools` | `List[ToolInfo]` | Tool schemas exposed to the model. Defaults to a single `run_python(code)`, matching OpenEnv's code environments. |
| `reset_kwargs` | `Dict` | Kwargs forwarded to the server's `reset()` (e.g. `task_prompt` / `expected_answer` for `repl_env`). Also settable per episode via the `env.reset_kwargs` attribute. |
| `connect_timeout_s` | `float` | WebSocket connect timeout, default 10s. |
| `message_timeout_s` | `float` | Per-message timeout, default 120s. |
| `client_kwargs` | `Dict` | Extra kwargs for the OpenEnv client constructor. |

Additional capabilities:

- `register_tool(tool_info, handler)`: register a tool handled **locally on the client** instead of being sent to the server. The typical use is a bookkeeping tool such as `submit_solution`, which records the model's answer on the env for the training loop to score later. A tool of the same name shadows the default one.
- `execute(action)`: send an action and return the server's **raw** `StepResult`, so you can read typed fields such as `exit_code`. The training loop uses it to run scoring code inside the same session.
- `episode_reward` / `last_result` / `client`: cumulative reward, last raw result, and the underlying synchronous client.

**Capacity and concurrency**: `OpenEnvClient` calls the OpenEnv client's `.sync()`, giving every instance a dedicated background event loop, so a thread pool of concurrent resets/steps is safe. But the server must have room for every concurrent session: the environment class has to declare `SUPPORTS_CONCURRENT_SESSIONS = True` and `create_app(..., max_concurrent_envs=N)` must be large enough, otherwise extra connections are rejected. OpenEnv's bundled `coding_env` is **single-session** by default (`SUPPORTS_CONCURRENT_SESSIONS = False`) and needs to be subclassed to lift that; `cookbook/rl/openenv_code/server_app.py` shows the full pattern.

### Usage with Rollout

Downstream usage is identical for both modes:

```python
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.rollout.api_multi_turn import APIMultiTurnRollout

env.reset()

# Bridge to ToolManager
env_tools = EnvTool.from_env(env)
manager = ToolManager(env_tools)

# Use in rollout
rollout = APIMultiTurnRollout(api=api, tool_manager=manager, max_turns=10)
results = rollout(trajectories)
```

See [Agentic RL Best Practices](../../Usage%20Guide/Agentic-RL-Best-Practices.md) for an end-to-end multi-turn GRPO example.

## EnvPool: Distributed Environment Pool

`EnvPool` is a `@remote_class` that shards `pool_size` **embedded** `OpenEnv` instances across Ray workers. Each worker manages `pool_size // world_size` slots, and `reset` / `step` are routed to the owning worker automatically via `remote_function`.

```python
from twinkle_agentic.envs.openenv import EnvPool

pool = EnvPool(
    pool_size=64,
    device_mesh=mesh,
    env_kwargs={'env_name': 'openspiel_env', 'env_kwargs': {'game_name': 'blackjack'}},
)

# Each slot becomes a standard Env, ready for EnvTool / ToolManager
envs = pool.get_adapters(64)
env_tools = EnvTool.from_env(envs[0])
```

| Method | Description |
|--------|-------------|
| `reset(idx)` / `step(idx, tool_name, arguments)` | Operate on a single slot. |
| `reset_batch(indices)` / `step_batch(indices, tool_names, arguments_list)` | Batch operations covering many slots in one RPC, returned in `indices` order. |
| `get_adapters(n)` | Wrap the first `n` slots as `EnvPoolAdapter` (a standard `Env`) on the driver side. |
| `close()` | Close all environments. |

`EnvPoolAdapter` implements the standard `Env` interface and proxies `reset` / `step` to the owning worker. On a `step` failure it returns `done=True` with the error in `info['error']`, so one broken environment does not stall the whole rollout batch.

## Implementing a Custom Environment

```python
from twinkle_agentic.envs.base import Env, StepResult

class CodeExecutionEnv(Env):

    def reset(self, trajectory=None):
        self._sandbox = create_sandbox()
        return StepResult(observation='Sandbox ready.')

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
                'description': 'Run Python code.',
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
