# Multi-Turn Rollout

The Rollout module provides one multi-turn conversation engine for agentic RLHF training. `MultiTurnRollout` can generate each assistant turn with a local sampler, an OpenAI-compatible API, or a callback that chooses between them.

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

Multi-turn rollout engine supporting local samplers, external APIs, and per-turn backend selection. Each trajectory runs independently in the rollout thread pool.

### Per-turn Loop

1. Encode each trajectory into an `InputFeature` with a generation prompt
2. Call `response_callback(...)` to obtain one `SampledSequence` from the sampler or API
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
| `sampler` | Sampler | Local sampler. Used by default when both backends exist. |
| `api` | `API` | Optional external generation API. |
| `template` | `Template` | Required local chat template for encoding every backend's output. |
| `response_callback` | `Callable` | Optional per-turn backend selector returning `SampledSequence`. |
| `api_appended_as` | `str` | API turns are `demonstration` (SFT only) or `context` (no loss). |
| `api_kwargs` | `Dict` | Request fields forwarded to each API call. |
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
| `completion_mask` | `List[int]` | Policy-generated positions that carry rollout log probabilities. |
| `turns` | `int` | Number of turns performed. |
| `stop_reason` | `str` | `'stop'` / `'length'` |
| `truncated` | `bool` | Whether the trajectory was cut off rather than concluding on its own: generation hit `max_tokens` (`stop_reason='length'`), the turn limit was reached, or a length cap dropped it. |
| `logprobs` | `List` | Per-token log probabilities (if available). |

### Ray Remote Support

`MultiTurnRollout` is decorated with `@remote_class()`, enabling transparent deployment as a Ray actor:

```python
# The rollout can run as a Ray remote actor
rollout_actor = MultiTurnRollout.remote(sampler=sampler, template=template, ...)
results = ray.get(rollout_actor.__call__.remote(trajectories))
```

## API and Mixed-Backend Rollouts

API-only rollout uses the same class and still requires the local template that tokenizes external replies:

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

When both `sampler` and `api` are supplied, the default is the sampler. Pass `response_callback` to choose per turn; it receives both backends and must return one `SampledSequence`. API turns have no rollout log probabilities, so `api_appended_as='demonstration'` includes them in SFT but excludes them from GRPO. Use `'context'` to exclude them from both.

### Stop Reasons

| Reason | Description |
|--------|-------------|
| `stop` | Assistant responded without tool calls (natural end). |
| `length` | Generation reached its token limit. |
| `max_turns` | Reached the tool-turn limit without a follow-up. |
| `generation_error` | The external endpoint failed before returning a valid response. |

## Streaming handoffs and weight publication (v5)

`twinkle_agentic.async_rl.StreamingDriver` owns admission, version accounting and completed-record handoffs. The training recipe owns optimizer batch size and publication cadence, requesting a publication with its own callback. The driver does not transfer weights itself.

`buffer_depth` limits in-flight trajectories; `max_rows_per_pull` independently bounds each training handoff. For example, capacity 128 with a pull limit of 64 leaves a batch available for overlap while the recipe trains and publishes after each update. Increasing capacity without bounding the pull can combine completed batches into one handoff.

- `group_pull=True`: the limit must be a positive integer multiple of `lcm(need_rows, num_generations)`. Whole, scored groups remain contiguous; incomplete or pending-score groups stay buffered.
- `group_pull=False`: the limit must be a positive integer multiple of `need_rows`.
- `max_rows_per_pull=None`: retain the existing uncapped whole-batch behavior.
- A custom `assembly_ready` owns its own selection policy and cannot be combined with this limit.

The limit is an upper bound on normal full-batch pulls. Existing finite-stream tail handling is separate and unchanged. Capacity and pull size do not relax policy-version validation.

For in-place weight publication with active partial rollouts, the caller must pause generation, complete weight synchronization, and only then resume generation. `abort_all_inflight` issues interruptions but does not close the retry gate; it is not a safe publication barrier by itself. A failed pause or transfer must propagate without unconditionally resuming generation. An adaptive `ClusterAdapter` owns its existing full publication transaction.

Abort, pause, resume and generation-status methods are explicitly marked `control=True`. This lets them bypass the generation-drain barrier even through a sampler handle serialized into a rollout actor, where the original worker-pool registry is absent. Ordinary lifecycle and weight operations retain their barriers.
