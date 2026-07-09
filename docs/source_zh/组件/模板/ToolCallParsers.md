# 工具调用解析器

Twinkle 的模板系统包含模块化的工具调用解析框架，用于训练具有函数调用能力的模型。

## 架构

```
ToolCallRegistry
├── HermesQwenParser  — Hermes/Qwen 风格 <tool_call>...</tool_call>
├── ReActParser       — ReAct Thought/Action/Observation
├── ClineParser       — Cline XML 工具调用
└── VCPParser         — VCP 协议
```

## ToolCallParser 接口

```python
from twinkle.template.tools import ToolCallParser

class ToolCallParser(ABC):
    name: str = ''

    def detect(self, text: str) -> bool:
        """检查文本是否包含此格式的标记"""

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """提取 OpenAI 格式的工具调用"""

    def clean(self, text: str) -> str:
        """去除标记，返回纯内容"""
```

## ToolCallRegistry

注册表自动发现解析器并路由检测：

```python
from twinkle.template.tools import ToolCallRegistry

# 检测补全使用了哪种格式
parser = ToolCallRegistry.detect_first(completion_text)
if parser:
    tool_calls = parser.parse(completion_text)
```

## 内置解析器

| 解析器 | 格式说明 |
|:-------|:---------|
| HermesQwenParser | `<tool_call>{"name": "...", "arguments": {...}}</tool_call>` |
| ReActParser | Thought/Action/Action Input/Observation |
| ClineParser | Cline XML 结构化参数 |
| VCPParser | Visual Code Protocol |
