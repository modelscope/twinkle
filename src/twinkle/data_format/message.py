# Copyright (c) ModelScope Contributors. All rights reserved.
import sys
from typing import Literal, Union, List, Dict, Optional, Any
if sys.version_info <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict


class ToolCall(TypedDict, total=False):
    """The information of the tool called by the LLM.

    Args:
        tool_name: The name of the tool.
        arguments: Json string. The arguments of the tool.
    """
    tool_name: str
    arguments: str


class Tool(TypedDict, total=False):
    """The information of the tool given to the LLM.

    Args:
        tool_name: The name of the tool.
        description: The description of the tool.
        parameters: Json string. The argument info of the tool.

    Example:
        >>> {
        >>>     "tool_name": "ocr_tool",
        >>>     "description": "A tool to transfer image to text.",
        >>>     "parameters": "{\\"image_path\\": \\"The input image path.\\"}"
        >>> }
    """
    tool_name: str
    description: str
    parameters: str


class Message(TypedDict, total=False):
    """The single round message of the LLM.

    Args:
        role: The role of the message.
            Available values:
                - system: The instruction information of the LLM, optional. If it exists, it should be the first round of the messages.
                - user: The user information given to the LLM.
                - assistant: The assistant information returned by the LLM.
                - tool_calls: The tool calling requirements of the LLM.
                - tool_call_id: The tool call id of the LLM.
                - reasoning_content: The reasoning content of the LLM, usually
        content: The content of the message.
        tool_calls: The tool calling requirements of the LLM.
        reasoning_content: The reasoning content of the LLM, usually generated with a pair <think></think> labels, which is the model thinking content.

    Example:
        >>> {"role": "system", "content": "You are a helpful assistant, which ..."}
        >>> {"role": "user", "content": "What is the weather of Beijing today?"}
        >>> {"role": "assistant", "content": "I need to call the weather api.", "tool_calls": [{"tool_name": "weather", "arguments": "{\\"city\\": \\"Beijing\\"}"}]}
        >>> {"role": "tool", "content": "Sunny"}
        >>> {"role": "assistant", "content": "The weather of Beijing is sunny."}
    """ # noqa
    role: Literal['system', 'user', 'assistant', 'tool']
    type: str
    content: Union[str, List[Dict[str, str]]]
    tool_calls: List[ToolCall]
    reasoning_content: str
    images: Optional[List[Union[str, Any]]]
    videos: Optional[List[Union[str, Any]]]
    audios: Optional[List[Union[str, Any]]]
