# 消息

消息代表了模型对话的单轮信息。消息的定义为：

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

本质上，`Message`是一个Dict。里面包含了若干字段，和开发者强相关的有：

- role: 消息类型，包含了'system', 'user', 'assistant', 'tool'四类。
  - system: 系统指令消息，仅在第0个消息中出现
  - user: 用户输入消息
  - assistant: 模型回复的消息
  - tool: 工具调用结果，类似user消息输入给模型
- content: 消息正文，如果包含多模态信息，则需要有占位符：
  - <image>: 图片占位符
  - <video>: 视频占位符
  - <audio>: 音频占位符

```text
<image>图片中是一片草地，上面有三只兔子。
```

- tool_calls: 工具调用列表，为模型输出给用户的信息，通常在assistant对应的content中解析出来。
  - ToolCall 的结构中包含tool_name和arguments两个字段，分别是工具名称和参数。arguments是一个json-string，可以被解析为合法json字符串。

- images: 消息中包含的原图片信息
- videos: 消息中包含的原视频信息
- audios: 消息中包含的原音频信息
