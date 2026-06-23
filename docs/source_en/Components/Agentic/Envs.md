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

## OpenEnv

`OpenEnv` adapts an [OpenEnv](https://github.com/OpenEnv) WebSocket-based environment server as a synchronous Twinkle `Env`.

```python
from twinkle_agentic.envs.openenv import OpenEnv

env = OpenEnv(
    base_url='http://localhost:8000',
    env_cls='coding_env.CodingEnv',      # Optional typed client
    env_kwargs={'message_timeout_s': 30},
    tool_schema=[...],                    # Optional tool definitions
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `base_url` | `str` | URL of the running OpenEnv server. |
| `env_cls` | `str` or class | Dotted import path or class for a typed client. `None` uses `GenericEnvClient`. |
| `env_kwargs` | `Dict` | Extra kwargs for the client constructor. |
| `tool_schema` | `List[ToolInfo]` | Tool definitions exposed via `tools()`. |
| `action_mapper` | `Callable` | Custom function to map `(tool_name, args)` to the action dict sent to the server. |

### Usage with Rollout

```python
from twinkle_agentic.envs.openenv import OpenEnv
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.rollout.api_multi_turn import APIMultiTurnRollout

# Set up environment
env = OpenEnv(base_url='http://localhost:8000', tool_schema=[...])
env.reset()

# Bridge to ToolManager
env_tools = EnvTool.from_env(env)
manager = ToolManager(env_tools)

# Use in rollout
rollout = APIMultiTurnRollout(api=api, tool_manager=manager, max_turns=10)
results = rollout(trajectories)
```

### Implementing a Custom Environment

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
