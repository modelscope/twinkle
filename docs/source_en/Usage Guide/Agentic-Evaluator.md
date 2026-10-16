# Agentic Evaluator

Install evaluation support separately:

```bash
pip install 'twinkle-kit[eval]'
```

`twinkle_agentic.evaluator.Evaluator` is a single-use facade for EvalScope's native runner. EvalScope owns datasets, agent loops, tools, judges, caches, metrics, and reports; Twinkle only adapts the candidate model boundary.

## Protocol API

```python
from twinkle_agentic.evaluator import Evaluator
from twinkle_agentic.protocol.openai import OpenAI

reports = Evaluator(
    api=OpenAI(model='qwen-plus', api_key='...', base_url='https://example.com/v1'),
    datasets=['gsm8k', 'bfcl_v4'],
    task_config={'generation_config': {'temperature': 0.0}},
).run()
```

## Sampler

```python
from twinkle_agentic.evaluator import Evaluator

reports = Evaluator(
    sampler=sampler,
    datasets=['gsm8k'],
    template=template,
    sampler_kwargs={'adapter_uri': 'twinkle://my-adapter'},
    task_config={'limit': 100, 'generation_config': {'max_tokens': 2048}},
).run()
```

The sampler path micro-batches compatible EvalScope requests. `template.parse_tool_call()` is required for tool benchmarks unless the sampler returns structured assistant tool calls. The HTTP sampler accepts either Twinkle `SamplingParams` or a mapping.

## Capability boundary

Text/chat, function-calling, and EvalScope native AgentLoop tasks are supported. Image, audio, video, streaming, non-native EvalScope backends, and nested Twinkle rollout loops are not. Explicit generation options are mapped exactly or rejected before evaluation begins; Twinkle does not approximate unsupported parameters.

`run()` returns EvalScope's `dict[str, Report]` unchanged. `resolved_task_config` exposes the constructed EvalScope configuration, and `output_dir` becomes available after EvalScope resolves the run directory. See EvalScope `TaskConfig` for pass-through fields and install its benchmark-specific extras separately when needed.
