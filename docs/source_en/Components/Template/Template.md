# Template

The template is a key component for converting Trajectory to InputFeature.

```python
class Template:

    def __init__(self,
                 model_id: str,
                 use_chat_template: bool = True,
                 max_length: Optional[int] = 8192,
                 truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise',
                 default_system: Optional[str] = None):
        ...

    def encode(self, trajectory: Trajectory, add_generation_prompt: bool = False) -> InputFeature:
        # Encode a single sample
        ...

    def batch_encode(self, trajectories: Union[Dict[str, Any], List[Trajectory]]) -> List[InputFeature]:
        # Batch encode samples
        ...

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        # Encode one sample and return the original sample
        # Generally used to check data reasonableness in RL algorithms like GRPO
        ...

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        # Batch check samples
        ...

    def decode(self, token_ids: List[int], **kwargs) -> str:
        # Decode sample
        ...

    def batch_decode(self, token_ids: List[List[int]], **kwargs) -> List[str]:
        # Batch decode samples
        ...
```

- model_id: Model id containing tokenizer or processor
- use_chat_template: Whether to use chat_template. If not used, it is generally a pre-training scenario
- max_length: Maximum length of a single sample
- truncation_strategy: How to handle the sample if it exceeds the maximum length
  - raise: Throw an exception. Generally used for very precise dataset scenarios
  - left: Remove tokens on the left to conform to max_length
  - right: Remove tokens on the right to conform to max_length
  - default_system: If the dataset does not have a system, use the default system

> Template does not support using functions as replacements because it needs to support many functions internally. If you need to write a new Template, please inherit the `Template` class.
> Generally speaking, using the Template base class is sufficient for pure text models. In the base class, we use tokenizer.apply_chat_template to encode the model, which is universal for general pure text models.
