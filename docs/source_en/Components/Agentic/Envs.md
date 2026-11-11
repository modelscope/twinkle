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
| `episode_reward` | Property: cumulative reward from `info['episode_reward']`, falling back to the last step's `reward`. |

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

Note that embedded `OpenEnv` does not implement `tools()`, so it inherits the base class's empty list and `EnvTool.from_env(env)` falls back to a single generic `env_action` tool. Pass explicit schemas to `EnvTool` (or subclass and override `tools()`) when the model needs to see the real action names.

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

**Capacity and concurrency**: `OpenEnvClient` calls the OpenEnv client's `.sync()`, giving every instance a dedicated background event loop, so a thread pool of concurrent resets/steps is safe. But the server must have room for every concurrent session: the environment class has to declare `SUPPORTS_CONCURRENT_SESSIONS = True` and `create_app(..., max_concurrent_envs=N)` must be large enough, otherwise extra connections are rejected. Total capacity is `workers x max_concurrent_envs`, since the limit applies per worker process. OpenEnv's bundled `coding_env` leaves `SUPPORTS_CONCURRENT_SESSIONS` at the conservative default and needs to be subclassed to lift it (`create_app` raises `ConcurrencyConfigurationError` for `max_concurrent_envs > 1` otherwise); `cookbook/rl/envs/openenv_server/server_app.py` shows the full pattern.

### Usage with Rollout

Downstream usage is the same for both modes:

```python
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.rollout import MultiTurnRollout

env.reset()

# Bridge to ToolManager
env_tools = EnvTool.from_env(env)
manager = ToolManager(env_tools)

# Use in rollout
rollout = MultiTurnRollout(api=api, template=template, tool_manager=manager, max_turns=10)
results = rollout(trajectories)
```

See [Agentic RL Deployment and Training](../../Usage%20Guide/Agentic-RL-Deployment-and-Training.md) for an end-to-end multi-turn GRPO example.

## AgentEnv: Firecracker microVM Sandboxes

`AgentEnv` uses the E2B SDK to operate remote sandboxes. It supports both **E2B's official cloud service** and a self-hosted [AgentENV](https://github.com/kvcache-ai/AgentENV) deployment. The official service needs no AgentENV gateway or client-side KVM setup. `reset()` creates a separate sandbox, `step()` executes tools, and `close()` releases it. Commands, files and subprocesses run remotely.

Compared with the OpenEnv adapters:

| | `OpenEnvClient` | `AgentEnv` |
|---|---|---|
| Isolation | Depends on the environment-server deployment | Provided by the sandbox provider |
| Executor | Depends on the environment, such as a game or code executor | Real sandbox processes |
| Memory per environment | Depends on the implementation | Depends on the provider / template |
| Files / pip / subprocesses | Depend on the environment and exposed tools | Available through the SDK |
| Transport | Persistent WebSocket | HTTP (E2B SDK) |
| Prerequisites | An OpenEnv service | Official E2B account and key, or a self-hosted AgentENV service and template |

OpenEnv's `coding_env` runs on smolagents' `LocalPythonExecutor`, **an AST interpreter rather than an OS-level sandbox**. It does not handle `decorator_list` at all, so **decorators are silently ignored**: `@patch` has no effect, the test does not error, and the reward comes out as a plausible-looking wrong number. Such silent errors are harder to diagnose than a crash. It is a good fit for enforcing an import allowlist, but not for executing adversarial code. When tests rely on decorators, or the model must write files, install packages, or spawn subprocesses, use `AgentEnv`.

### Connect to official E2B

From the repository root, select the `e2b` extra. It includes the validated E2B SDK and `python-dotenv` for loading `.env`:

```bash
pip install -e '.[e2b]'
```

Put the key in `.env` in the working directory and keep that file out of Git:

```dotenv
E2B_API_KEY=your_e2b_api_key
```

Leave `E2B_API_URL`, `E2B_SANDBOX_URL` and `E2B_DOMAIN` unset. Also omit the `api_url` / `sandbox_url` arguments so the SDK selects the official control and sandbox endpoints.

First check create → execute → release independently. This needs neither a Twinkle training server nor a GPU:

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

The official `base` template uses an unprivileged user. Use a writable directory such as `/home/user/twinkle` rather than copying `/workspace` from a self-hosted AgentENV example. Directory initialization or `setup_commands` failures raise; the caller remains responsible for closing an already-created sandbox.

Once connected, follow the [E2B Blackjack training example in the client guide](../../Usage%20Guide/Server%20and%20Client/Twinkle-Client.md). Each trajectory gets a separate sandbox running the game; training and policy publication still run on the Twinkle server.

### Connect to self-hosted AgentENV

This route requires deploying AgentENV, building a template (for example, `aenv pull ubuntu:22.04 --name my-env`) and installing a compatible E2B SDK. Use the template, endpoints and authentication configured for your deployment; they are separate from the official E2B cloud settings.

```python
from twinkle_agentic.envs import AgentEnv

env = AgentEnv(
    template='my-env',                     # Template built beforehand with the aenv CLI
    api_url='http://10.0.0.5:8000',        # Server or gateway; falls back to E2B_API_URL
    sandbox_timeout=600,                   # Must outlast one episode plus any test replay
)
try:
    env.reset()                            # Boots a fresh sandbox; the scheduler picks the node
    result = env.step('run_command', {'command': 'python -c "print(1 + 1)"'})
    print(result.observation)              # '2'
finally:
    env.close()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `template` | `str` | Required provider template name/ID. The official example uses `base`; self-hosted AgentENV uses your own template. |
| `api_url` | `str` | Self-hosted server / gateway URL. When omitted, the SDK uses its environment settings or official default. |
| `api_key` | `str` | Explicit argument, then `E2B_API_KEY`. The adapter retains a `'dummy'` fallback, but official E2B requires a valid real key; self-hosted authentication depends on the deployment. |
| `sandbox_url` | `str` | The data plane, for a deployment whose sandbox gateway answers on a different host than the API. Defaults to `api_url`. |
| `sandbox_timeout` | `int` | Sandbox timeout passed to the provider, in seconds; default 300. Pause/termination behavior depends on the provider. Do not assume automatic recovery; explicitly call `close()`. |
| `command_timeout` | `int` | Per-command timeout in seconds, default 120. |
| `setup_commands` | `List[str]` | Commands run once after each `reset`; their output becomes the reset observation. |
| `sandbox_envs` | `Dict[str, str]` | Environment variables injected into the sandbox. |
| `metadata` | `Dict[str, str]` | Sandbox metadata, visible in the list APIs — useful for tagging a run name or trajectory id. |
| `refresh_timeout` | `bool` | Extend the timeout after every step, default `True`, to avoid expiry during long episodes. |
| `include_default_tools` | `bool` | Expose the built-in `run_command` / `write_file` / `read_file`, default `True`. |

### Registering task tools

The AgentENV server defines no tools of its own; it only provides capability primitives (arbitrary command execution, file I/O, port proxying). Tools are therefore a purely client-side concept, registered in one of two ways:

```python
# 1. Shell command template, formatted with the tool arguments
env.register_command_tool(
    {'type': 'function', 'function': {
        'name': 'run_tests',
        'description': 'Run the task test suite.',
        'parameters': {'type': 'object',
                       'properties': {'test_file': {'type': 'string'}},
                       'required': ['test_file']}}},
    'cd /workspace && pytest {test_file} -x -q')

# 2. Arbitrary Python handler: handler(env, arguments) -> str
def _submit(env, arguments):
    env.submitted_code = arguments.get('code', '')
    return 'Solution submitted.'

env.register_tool(submit_schema, _submit)      # Both return self, so calls can be chained
```

Setting `include_default_tools=False` hides the built-ins so the action space matches the task exactly, which keeps reward attribution clean. A registered tool whose name collides with a built-in overrides it.

Additional capabilities:

- `run_command(arguments)`: public, so custom handlers can reuse it. A non-zero exit code is **not** raised; stdout, stderr, and the exit code are formatted into the observation so the model can react to the failure.
- `sandbox` / `sandbox_id`: the underlying E2B handle (giving access to PTY, file watching, and similar) and the current sandbox id.
- Observations are truncated to 32K characters to keep a runaway command from blowing up the context.

**Error handling and rewards**: base `AgentEnv.step` returns ordinary tool errors as observations so the model can recover; initialization failures raise. `ensure_idle()` rejects resource reuse when remote completion is unknown. Task environments should also distinguish invalid model actions from infrastructure failures: E2B Blackjack aborts training on a failed game process rather than assigning a zero reward. `AgentEnv.evaluate` defaults to zero; tasks may provide environment rewards through `StepResult.reward` or use a separate scorer.

**Concurrency**: `AgentEnv` is not a `@remote_class`; sandbox scheduling belongs to the connected provider. Give each concurrent trajectory its own instance. Do **not** put it in `EnvPool`, which serves embedded OpenEnv instances. `reset()` is a synchronous network call; account for initialization time as control-thread overhead rather than model-generation time.

Capacity depends on the scheduler's in-flight/buffer budget and the provider's concurrency, memory and billing quotas. The E2B streaming example defaults to `BUFFER_GROUPS × NUM_GENERATIONS = 12` sandboxes. Self-hosted AgentENV also requires checking template memory settings; see [Agentic RL Deployment and Training](../../Usage%20Guide/Agentic-RL-Deployment-and-Training.md).

## EnvPool: Distributed Environment Pool

`EnvPool` is a `@remote_class` that shards `pool_size` **embedded** `OpenEnv` instances across Ray workers. Each worker manages `ceil(pool_size / world_size)` slots (the last shard is clipped to `pool_size`, so it may hold fewer), and `reset` / `step` are routed to the owning worker automatically via `remote_function`.

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
| `get_adapters(n)` | Wrap the first `n` slots as `EnvPoolAdapter` (a standard `Env`) on the driver side. Raises if `n > pool_size`. |
| `close()` | Close all environments. |

`EnvPoolAdapter` implements the standard `Env` interface and proxies `reset` / `step` to the owning worker. On a `step` failure it returns `done=True` with the error in `info['error']`, so one broken environment does not stall the whole rollout batch. Its own `close()` is a no-op — release resources through the pool's `close()`.

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
