# 工具与 ToolManager

Tools 模块提供了抽象工具接口和中央工具分发器（`ToolManager`），用于 Agentic 多轮 rollout。工具遵循 OpenAI function-calling schema，与 LLM 工具调用能力无缝集成。

## Tool 基类

```python
from abc import ABC, abstractmethod
from twinkle.data_format import Tool as ToolInfo

class Tool(ABC):

    @abstractmethod
    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """执行工具并返回字符串结果。"""
        raise NotImplementedError

    @abstractmethod
    def tool_info(self) -> ToolInfo:
        """返回 OpenAI 兼容的工具 schema。"""
        raise NotImplementedError
```

### 实现自定义工具

```python
from twinkle_agentic.tools.base import Tool

class SearchTool(Tool):

    def __call__(self, tool_name: str, arguments: dict) -> str:
        query = arguments.get('query', '')
        # 执行搜索逻辑
        return f'搜索结果：{query}'

    def tool_info(self):
        return {
            'type': 'function',
            'function': {
                'name': 'search',
                'description': '搜索网络信息。',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'query': {
                            'type': 'string',
                            'description': '搜索查询。',
                        },
                    },
                    'required': ['query'],
                },
            },
        }
```

## ToolManager

`ToolManager` 是工具的注册中心和分发器。它解析 LLM 结构化输出中的工具调用，并路由到正确的工具实现。

```python
from twinkle_agentic.tools.tool_manager import ToolManager

# 通过 Tool 实例列表初始化
manager = ToolManager([search_tool, calculator_tool])

# 或通过字典初始化
manager = ToolManager({'search': search_tool, 'calc': calculator_tool})

# 或动态注册
manager = ToolManager()
manager.register(search_tool)
manager.register(calculator_tool)
```

### 核心方法

| 方法 | 说明 |
|------|------|
| `register(tool)` | 注册工具（名称从 `tool_info()` 提取）。 |
| `unregister(name)` | 按名称移除工具。 |
| `names()` | 列出所有已注册的工具名称。 |
| `copy()` | 创建管理器的浅拷贝。 |
| `tool_infos()` | 返回所有工具 schema 列表（用于 API 请求）。 |
| `__call__(tool_call)` | 分发工具调用并返回结果字符串。 |

### 分发工具调用

`ToolManager` 接受 OpenAI 格式的工具调用字典：

```python
tool_call = {
    'id': 'call_1',
    'type': 'function',
    'function': {
        'name': 'search',
        'arguments': '{"query": "Python 教程"}',
    },
}

result = manager(tool_call)
# result: '搜索结果：Python 教程'
```

**错误处理：** 如果工具名未知、参数是无效 JSON 或工具抛出异常，`ToolManager` 返回描述性错误字符串而不是抛出异常——这保证了 rollout 循环的持续运行。

### 与 Rollout 集成

```python
from twinkle_agentic.rollout.multi_turn import MultiTurnRollout

rollout = MultiTurnRollout(
    sampler=sampler,
    template=template,
    tool_manager=manager,  # 传入工具管理器
    max_turns=6,
)
```

Rollout 引擎对模型生成的每个工具调用执行 `manager(tool_call)`，并将结果作为 `{'role': 'tool', 'content': result}` 消息追加。
