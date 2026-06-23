# Multi-Turn Rollout

The Rollout module provides multi-turn conversation rollout engines for agentic RLHF training. Two implementations are available: `MultiTurnRollout` for batched vLLM sampling and `APIMultiTurnRollout` for OpenAI-compatible API endpoints.

## Rollout Base Class

```python
from abc import ABC, abstractmethod
from twinkle.data_format import Trajectory

class Rollout(ABC):

    @abstractmethod
    def __call__(self, trajectories: List[Trajectory], **kwargs) -> List[Trajectory]:
        raise NotImplementedError()
```

All rollouts accept a list of trajectories and return the same number of trajectories with additional fields (`messages`, `turns`, `stop_reason`, `truncated`).

## MultiTurnRollout

Batched multi-turn rollout engine that uses a vLLM sampler for generation. All active trajectories are sampled in a single batched call per turn for maximum throughput.

### Per-turn Loop

1. Encode each trajectory into an `InputFeature` with a generation prompt
2. Batch `sampler.sample(active_pifs)` — all live trajectories in parallel
3. Check termination: `stop_reason == 'length'`, no tool calls, or max turns reached
4. Dispatch tools via `ToolManager`, append tool responses
5. Compute bridge tokens (tool turns + generation prompt) with `labels = -100`
6. Repeat until all trajectories are done

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

# Run rollout
results = rollout(trajectories)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `sampler` | Sampler | vLLM sampler instance for batched generation. |
| `template` | `Template` | Chat template for encoding/decoding. |
| `tool_manager` | `ToolManager` | Tool dispatcher. Can also be passed per-call. |
| `sampling_params` | `SamplingParams` | Default sampling parameters. |
| `max_turns` | `int` | Maximum number of turns per trajectory (default: 6). |
| `max_trajectory_tokens` | `int` | Max total token length; exceeding truncates the trajectory. |
| `trace_dir` | `str` | Directory for per-trajectory JSON trace dumps. |
| `trace_callback` | `Callable` | Decides whether to store a trajectory trace. |
| `success_callback` | `Callable` | Decides filename prefix (`ok-` vs `fail-`). |

### Output Fields

Each output trajectory dict includes:

| Field | Type | Description |
|-------|------|-------------|
| `messages` | `List[Dict]` | Full conversation including tool turns. |
| `input_ids` | `List[int]` | Token IDs of the full sequence. |
| `labels` | `List[int]` | Training labels (`-100` for non-trainable tokens). |
| `turns` | `int` | Number of turns performed. |
| `stop_reason` | `str` | `'stop'` / `'length'` |
| `truncated` | `bool` | Whether the trajectory was truncated. |
| `logprobs` | `List` | Per-token log probabilities (if available). |

### Ray Remote Support

`MultiTurnRollout` is decorated with `@remote_class()`, enabling transparent deployment as a Ray actor:

```python
# The rollout can run as a Ray remote actor
rollout_actor = MultiTurnRollout.remote(sampler=sampler, template=template, ...)
results = ray.get(rollout_actor.__call__.remote(trajectories))
```

## APIMultiTurnRollout

Multi-turn rollout over an OpenAI-compatible chat-completions API. Each trajectory runs independently in a thread pool for network concurrency.

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

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `api` | `OpenAI` | OpenAI-compatible API client. |
| `tool_manager` | `ToolManager` | Tool dispatcher (single or per-trajectory list). |
| `sampling_params` | `SamplingParams` | Default sampling parameters. |
| `max_turns` | `int` | Maximum turns per trajectory (default: 6). |
| `concurrency` | `int` | Thread pool size for parallel API calls (default: 8). |
| `extra_body` | `Dict` | Extra fields to include in API requests. |
| `trace_dir` | `str` | Directory for trace dumps. |

### Stop Reasons

| Reason | Description |
|--------|-------------|
| `stop` | Assistant responded without tool calls (natural end). |
| `length` | API returned `finish_reason='length'` (token limit). |
| `max_turns` | Reached `max_turns` limit. |
| `api_error` | API call or tool execution raised an exception. |

## Choosing Between Rollouts

| Feature | MultiTurnRollout | APIMultiTurnRollout |
|---------|-----------------|---------------------|
| **Backend** | vLLM sampler (local GPU) | OpenAI-compatible API |
| **Training integration** | Produces `input_ids` / `labels` for GRPO | Messages only (for data collection) |
| **Batching** | GPU-level batch parallelism | Network-level thread concurrency |
| **Use case** | Online RLHF training loop | Offline data generation / evaluation |
