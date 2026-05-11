# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import Any, Dict, List, Literal, Union

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class FunctionCall(TypedDict, total=False):
    """Inner ``function`` payload of a tool call.

    ``arguments`` should be a ``dict`` at chat-template render time (the
    template iterates with ``arguments|items``); at dispatch time the
    :class:`ToolManager` also accepts a JSON string for backward-compat
    with callers that build ToolCalls from raw API JSON.
    """
    name: str
    arguments: Union[str, Dict[str, Any]]


class ToolCall(TypedDict, total=False):
    """A single tool invocation emitted by the assistant, OpenAI shape.

    Example:
        >>> {
        >>>     "type": "function",
        >>>     "function": {
        >>>         "name": "weather",
        >>>         "arguments": {"city": "Beijing"},
        >>>     },
        >>> }
    """
    id: str
    type: Literal['function']
    function: FunctionCall


class FunctionSpec(TypedDict, total=False):
    """Inner ``function`` payload of a tool definition."""
    name: str
    description: str
    parameters: Union[str, Dict[str, Any]]


class Tool(TypedDict, total=False):
    """Tool definition advertised to the LLM, OpenAI shape.

    Example:
        >>> {
        >>>     "type": "function",
        >>>     "function": {
        >>>         "name": "ocr_tool",
        >>>         "description": "A tool to transfer image to text.",
        >>>         "parameters": {"image_path": "The input image path."},
        >>>     },
        >>> }
    """
    type: Literal['function']
    function: FunctionSpec


class Message(TypedDict, total=False):
    """The single round message of the LLM.

    Example:
        >>> {"role": "system", "content": "You are a helpful assistant, which ..."}
        >>> {"role": "user", "content": "What is the weather of Beijing today?"}
        >>> {"role": "assistant", "content": "I need to call the weather api.",
        ...  "tool_calls": [{"type": "function",
        ...                  "function": {"name": "weather",
        ...                               "arguments": {"city": "Beijing"}}}]}
        >>> {"role": "tool", "content": "Sunny", "tool_call_id": "call_1"}
        >>> {"role": "assistant", "content": "The weather of Beijing is sunny."}
    """ # noqa
    role: Literal['system', 'user', 'assistant', 'tool']
    type: str
    content: Union[str, List[Dict[str, str]]]
    tool_calls: List[ToolCall]
    tool_call_id: str
    reasoning_content: str
