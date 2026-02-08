# Message

A message represents a single round of information in a model conversation. The message definition is:

```python

class ToolCall(TypedDict, total=False):
    tool_name: str
    arguments: str
    
class Message(TypedDict, total=False):
    role: Literal['system', 'user', 'assistant', 'tool']
    type: str
    content: Union[str, List[Dict[str, str]]]
    tool_calls: List[ToolCall]
    reasoning_content: str
    images: Optional[List[Union[str, Any]]]
    videos: Optional[List[Union[str, Any]]]
    audios: Optional[List[Union[str, Any]]]
```

Essentially, `Message` is a Dict. It contains several fields, with the following being strongly relevant to developers:

- role: Message type, including four types: 'system', 'user', 'assistant', 'tool'.
  - system: System instruction message, only appears in the 0th message
  - user: User input message
  - assistant: Model reply message
  - tool: Tool call result, similar to user message input to the model
- content: Message body, if it contains multimodal information, then placeholders are needed:
  - <image>: Image placeholder
  - <video>: Video placeholder
  - <audio>: Audio placeholder

```text
<image>The image shows a grassland with three rabbits on it.
```

- tool_calls: Tool call list, information output by the model to the user, usually parsed from the content corresponding to assistant.
  - The ToolCall structure contains two fields: tool_name and arguments, which are the tool name and parameters respectively. arguments is a json-string that can be parsed into a valid json string.

- images: Original image information contained in the message
- videos: Original video information contained in the message
- audios: Original audio information contained in the message
