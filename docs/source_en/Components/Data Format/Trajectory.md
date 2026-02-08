# Trajectory

The raw data structure input to Template after dataset ETL is `Trajectory` (trajectory). This is a naming method that conforms to AgenticRL, mainly representing the actual performance of the model's multi-turn conversation.

```python
class Trajectory(TypedDict, total=False):
    messages: List[Message]
    extend_message: List[Tuple[str, List[Message]]]
    tools: List[Tool]
```

- messages: A list of Message messages, representing the multi-turn conversations actually conducted by the model, usually alternating between `user` and `assistant`.
- extend_message: In training such as DPO and PPO, unusable trajectories or low-score trajectories are usually needed, which will be placed in extend_message
- tools: A list of all available tools for the model in this call

Trajectory is the standard interface for all dataset preprocessing outputs and template inputs in Twinkle. The format conversion goes from the original dataset to Trajectory, and then to InputFeature.
