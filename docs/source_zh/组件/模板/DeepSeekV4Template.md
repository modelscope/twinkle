# DeepSeek-V4 模板

`DeepseekV4Template` 为 DeepSeek V4 提供原生支持，包括其独特的思考模式、工具调用协议和多 token 特殊标记。

## 使用方法

```python
from twinkle.template import DeepseekV4Template

template = DeepseekV4Template(
    model_id='deepseek-ai/DeepSeek-V4',
    enable_thinking=True,
)
```

## 特性

- **自定义 tokenizer 包装**：用 DeepSeek V4 的编码协议覆盖 `apply_chat_template`
- **思考模式**：支持 `thinking` / `chat` 模式切换
- **工具调用**：原生 DSML 工具调用编码
- **多 token EOS**：处理 DeepSeek V4 的多字符特殊标记

## 与基础模板的区别

| 特性 | 基础模板 | DeepseekV4Template |
|:-----|:---------|:-------------------|
| Chat 模板 | HuggingFace 原生 | 自定义 DSML 编码 |
| 思考模式 | `<think>` 标签 | 原生思考模式开关 |
| 工具调用 | Hermes/Qwen 格式 | DSML 工具块 |
| EOS 处理 | 单 token | 多 token 特殊标记 |
