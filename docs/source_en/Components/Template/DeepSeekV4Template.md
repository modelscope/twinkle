# DeepSeek-V4 Template

The `DeepseekV4Template` provides native support for DeepSeek V4's custom chat template encoding, including its unique thinking mode, tool-call protocol, and multi-token special tokens.

## Usage

```python
from twinkle.template import DeepseekV4Template

template = DeepseekV4Template(
    model_id='deepseek-ai/DeepSeek-V4',
    enable_thinking=True,
)
```

## Features

- **Custom tokenizer wrapper**: Overrides `apply_chat_template` with DeepSeek V4's encoding protocol
- **Thinking mode**: Supports `thinking` / `chat` modes with configurable reasoning effort
- **Tool calls**: Native DSML (DeepSeek Markup Language) tool-call encoding
- **Multi-token EOS**: Handles DeepSeek V4's multi-character special tokens

## Thinking Modes

```python
# Enable deep thinking (reasoning mode)
template = DeepseekV4Template(model_id='...', enable_thinking=True)

# Control reasoning effort
# 'max' or 'high' enables extended reasoning budget
template.encode(messages, reasoning_effort='max')
```

## Tool Call Support

DeepSeek V4 uses its own DSML protocol for structured function calling:

```python
messages = [
    {'role': 'user', 'content': 'What is the weather in Shanghai?'},
]
tools = [
    {'type': 'function', 'function': {'name': 'get_weather', 'parameters': {...}}}
]

features = template.encode(messages, tools=tools)
```

## Key Differences from Base Template

| Feature | Base Template | DeepseekV4Template |
|:--------|:-------------|:-------------------|
| Chat template | HuggingFace native | Custom DSML encoding |
| Thinking | `<think>` tags | Native thinking mode toggle |
| Tool calls | Hermes/Qwen format | DSML tool blocks |
| EOS handling | Single token | Multi-token special markers |
