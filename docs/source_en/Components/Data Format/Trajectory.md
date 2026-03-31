# Trajectory

The raw data structure input to Template after dataset ETL is `Trajectory` (trajectory). This is a naming method that conforms to AgenticRL, mainly representing the actual performance of the model's multi-turn conversation.

```python
class Trajectory(TypedDict, total=False):
    messages: List[Message]
    tools: List[Tool]
    user_data: List[Tuple[str, Any]]
```

- messages: A list of Message messages, representing the multi-turn conversations actually conducted by the model, usually alternating between `user` and `assistant`.
- tools: A list of all available tools for the model in this call
- user_data: User-defined data, such as labels in KTO training

For preference alignment training like DPO, preprocessors return `{'positive': List[Trajectory], 'negative': List[Trajectory]}` format.

Trajectory is the standard interface for all dataset preprocessing outputs and template inputs in Twinkle. The format conversion goes from the original dataset to Trajectory, and then to InputFeature.
