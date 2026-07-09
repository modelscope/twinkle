# Tool Call Parsers

Twinkle's template system includes a modular tool-call parsing framework for training models with function calling capabilities.

## Architecture

```
ToolCallRegistry
├── HermesQwenParser  — Hermes/Qwen style <tool_call>...</tool_call>
├── ReActParser       — ReAct Thought/Action/Observation
├── ClineParser       — Cline XML-based tool calls
└── VCPParser         — VCP protocol
```

## ToolCallParser Interface

```python
from twinkle.template.tools import ToolCallParser

class ToolCallParser(ABC):
    name: str = ''
    open_marker: str | None = None
    close_marker: str | None = None

    def detect(self, text: str) -> bool:
        """Check if text contains this format's markup."""
        ...

    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Extract tool calls in OpenAI format."""
        ...

    def clean(self, text: str) -> str:
        """Strip markup, return plain content."""
        ...
```

## ToolCallRegistry

The registry auto-discovers parsers and routes detection:

```python
from twinkle.template.tools import ToolCallRegistry

# Detect which format a completion uses
parser = ToolCallRegistry.detect_first(completion_text)
if parser:
    tool_calls = parser.parse(completion_text)
    clean_text = parser.clean(completion_text)
```

## Built-in Parsers

### HermesQwenParser

Parses Hermes/Qwen-style function calls:

```xml
<tool_call>
{"name": "get_weather", "arguments": {"city": "Shanghai"}}
</tool_call>
```

### ReActParser

Parses ReAct-style reasoning traces:

```
Thought: I need to check the weather
Action: get_weather
Action Input: {"city": "Shanghai"}
Observation: ...
```

### ClineParser

Parses Cline XML-based tool invocations with structured parameters.

### VCPParser

Parses VCP (Visual Code Protocol) tool calls.

## Usage in Training

Tool call parsers integrate with the Template during preprocessing:

```python
from twinkle.template import Template

template = Template(
    model_id='ms://Qwen/Qwen3.5-4B',
    enable_thinking=True,
)

# Template automatically uses ToolCallRegistry for
# tool-call aware tokenization during encoding
features = template.encode(messages, tools=tool_definitions)
```
