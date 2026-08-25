# Agentic 评测

评测能力是可选依赖：

```bash
pip install 'twinkle-kit[eval]'
```

`twinkle_agentic.evaluator.Evaluator` 是 EvalScope Native runner 的一次性轻量封装。数据集、AgentLoop、工具、Judge、缓存、指标和报告仍由 EvalScope 负责，Twinkle 只适配被评测模型。

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
reports = Evaluator(
    sampler=sampler,
    datasets=['gsm8k'],
    template=template,
    sampler_kwargs={'adapter_uri': 'twinkle://my-adapter'},
    task_config={'limit': 100, 'generation_config': {'max_tokens': 2048}},
).run()
```

Sampler 路径会将兼容的 EvalScope 请求微批处理。工具评测要求 sampler 返回结构化 tool calls，或提供 `template.parse_tool_call()`。HTTP sampler 同时接受 Twinkle `SamplingParams` 和字典。

## 能力边界

支持文本/对话、函数调用和 EvalScope Native AgentLoop；不支持多模态、流式、EvalScope 非 Native backend，以及在 adapter 内嵌套 Twinkle rollout。显式 generation 参数必须精确映射，否则会在评测前报错，不会静默近似或忽略。

`run()` 原样返回 EvalScope 的 `dict[str, Report]`。`resolved_task_config` 可获取实际构建的配置，EvalScope 创建输出目录后可通过 `output_dir` 获取。其余透传字段请参考 EvalScope `TaskConfig`；benchmark 的额外依赖仍应按 EvalScope 文档单独安装。
