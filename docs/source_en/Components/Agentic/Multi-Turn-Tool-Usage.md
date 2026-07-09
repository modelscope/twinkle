# Multi-Turn Tool Usage Guide

This guide shows how to set up and run multi-turn agentic rollouts with tool use in Twinkle.

## Architecture Overview

The agentic rollout pipeline consists of four key components:

- **Tool** — implements a specific capability (search, code execution, etc.)
- **ToolManager** — registers tools and dispatches LLM tool calls
- **Env** (optional) — RL environment that exposes tools via `EnvTool`
- **Rollout** — drives the multi-turn conversation loop

## Quick Start: API-based Rollout

The simplest way to run a multi-turn tool-use rollout using an OpenAI-compatible API:

```python
from twinkle_agentic.protocol.openai import OpenAI
from twinkle_agentic.tools.base import Tool
from twinkle_agentic.tools.tool_manager import ToolManager
from twinkle_agentic.rollout.api_multi_turn import APIMultiTurnRollout
from twinkle.data_format.sampling import SamplingParams

# 1. Define tools
class WeatherTool(Tool):
    def __call__(self, tool_name, arguments):
        city = arguments.get('city', 'unknown')
        return f'The weather in {city} is sunny, 25°C.'

    def tool_info(self):
        return {
            'type': 'function',
            'function': {
                'name': 'get_weather',
                'description': 'Get the current weather for a city.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'city': {'type': 'string', 'description': 'City name.'},
                    },
                    'required': ['city'],
                },
            },
        }

# 2. Set up ToolManager
manager = ToolManager([WeatherTool()])

# 3. Create API client
api = OpenAI(model='qwen3.5-32b', base_url='http://localhost:8000/v1')

# 4. Create rollout
rollout = APIMultiTurnRollout(
    api=api,
    tool_manager=manager,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=2048),
    max_turns=6,
    concurrency=8,
)

# 5. Prepare trajectories
trajectories = [
    {
        'messages': [
            {'role': 'user', 'content': "What's the weather like in Beijing?"},
        ],
    },
]

# 6. Run rollout
results = rollout(trajectories)
for r in results:
    print(f"Turns: {r['turns']}, Stop: {r['stop_reason']}")
    for msg in r['messages']:
        print(f"  [{msg['role']}] {msg.get('content', '')[:100]}")
```

## Training Integration: vLLM-based Rollout

For RLHF training, use `MultiTurnRollout` which produces `input_ids` and `labels`:

```python
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout
from twinkle.data_format.sampling import SamplingParams

rollout = MultiTurnRollout(
    sampler=vllm_sampler,           # vLLMSampler instance
    template=template,               # Chat template
    tool_manager=manager,
    sampling_params=SamplingParams(temperature=0.7, max_tokens=4096),
    max_turns=6,
    max_trajectory_tokens=8192,
    trace_dir='rollout_traces/',
)

# In GRPO training loop
results = rollout(batch_trajectories)
# results contain input_ids, labels, logprobs for training
```

## Using Environments as Tools

Bridge an RL environment into the tool pipeline:

```python
from twinkle_agentic.envs.base import Env, StepResult
from twinkle_agentic.envs.env_tool import EnvTool
from twinkle_agentic.tools.tool_manager import ToolManager

# Define environment
class CodeEnv(Env):
    def step(self, tool_name, arguments):
        code = arguments.get('code', '')
        # Execute code in sandbox
        result = execute_in_sandbox(code)
        return StepResult(observation=result, reward=1.0, done=False)

    def tools(self):
        return [{
            'type': 'function',
            'function': {
                'name': 'run_python',
                'description': 'Execute Python code.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'code': {'type': 'string'},
                    },
                    'required': ['code'],
                },
            },
        }]

# Bridge Env -> Tool -> ToolManager
env = CodeEnv()
env_tools = EnvTool.from_env(env)
manager = ToolManager(env_tools)

# Use manager in rollout as usual
rollout = APIMultiTurnRollout(api=api, tool_manager=manager, max_turns=10)
```

## Using OpenEnv Environments

Connect to a remote OpenEnv WebSocket server:

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
            'description': 'Submit code solution.',
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

## Per-Trajectory Tool Managers

For scenarios where each trajectory needs its own tool set (e.g., trajectory-bound state):

```python
# Create per-trajectory managers
managers = []
for traj in trajectories:
    env = create_env_for(traj)
    env_tools = EnvTool.from_env(env)
    managers.append(ToolManager(env_tools))

# Pass as a list (aligned 1:1 with trajectories)
results = rollout(trajectories, tool_manager=managers)
```

## Trace Debugging

Both rollout implementations support trace dumps for debugging:

```python
rollout = APIMultiTurnRollout(
    api=api,
    tool_manager=manager,
    trace_dir='traces/',
    trace_callback=lambda t: t['turns'] > 1,    # Only store multi-turn
    success_callback=lambda t: t.get('stop_reason') == 'stop',
)
```

Trace files are saved as `{step}-{ok|fail}-{id}.json` with the full conversation and metadata.
