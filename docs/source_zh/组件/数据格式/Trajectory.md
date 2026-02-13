# 轨迹

数据集ETL之后输入Template的原始数据结构是`Trajectory`(轨迹)。这是一个符合AgenticRL的命名方法，主要代表了模型多轮对话的实际表现。

```python
class Trajectory(TypedDict, total=False):
    messages: List[Message]
    extend_message: List[Tuple[str, List[Message]]]
    tools: List[Tool]
```

- messages: Message消息的列表，代表模型实际进行的多轮对话，通常是`user`和`assistant`交替出现。
- extend_message: 在DPO、PPO等训练中通常需要不可用轨迹，或低分轨迹，该轨迹会放在extend_message中
- tools: 模型在本次调用中的所有可用工具列表

Trajectory是twinkle中所有数据集预处理输出，模板输入的标准接口。格式转换为由原始数据集转换为Trajectory，再到InputFeature。
