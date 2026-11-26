# Template

The template encodes a trajectory while retaining its complete dict payload. `InputFeature` is a model-input type hint, not a field projection: messages, tools, user_data and original images/videos/audios remain alongside numerical tensors. Non-model fields are filtered by the model's input processor.

`encode_prompt` returns the same complete dict. Multimodal trajectories carry serving `engine_input_ids` separately from training `input_ids`; text-only trajectories reuse `input_ids` when `engine_input_ids` is absent. `append_completion` appends actual generated tokens and returns a new trajectory with a structured assistant message; it does not re-encode the sampled history.

```python
class Template:

    def __init__(self,
                 model_id: str,
                 use_chat_template: bool = True,
                 max_length: Optional[int] = None,
                 truncation_strategy: Literal['raise', 'left', 'right', 'split', 'delete'] = 'raise',
                 default_system: Optional[str] = None):
        ...

    def encode(self, sample=None, add_generation_prompt=None, *, intent=None,
               engine='vllm', return_length=False, **kwargs):
        ...

    def batch_encode(self, trajectories, add_generation_prompt=None, *, intent=None,
                     engine='vllm', return_length=False, **kwargs):
        ...

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        # Encode one sample and return the original sample
        # Generally used to check data reasonableness in RL algorithms like GRPO
        ...

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        # Batch check samples
        ...

    def decode(self, token_ids, *, context=None, **kwargs):
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
  - split: Split the oversized sample into multiple max_length chunks (not supported for multimodal, LazyDataset, or IterablePackingDataset)
  - delete: Single-sample encode raises MaxLengthError; batch_encode drops that sample
- default_system: If the dataset does not have a system message, use the default system

## Shared interface

`swift.dev.template.Template` inherits this base while retaining swift's factories and model-family implementations. Default/False encoding uses TRAIN; True uses INFER. Explicit intents are shared by both libraries. INFER/ENGINE/SCORE omit training labels; `encode_prompt(..., add_generation_prompt=False)` uses SCORE. Causal training labels shift left once, with a final -100; loss_scale/completion_mask shift with them when present. A nonempty `user_data.key_rounds` selects assistant message indices after system-message normalization.

List batches return a flat list; columnar batches return equally sized columns. Single encode can return a list under split; encode_prompt rejects split results. Left/right retain protected media tokens and slice their sequence companions together. Multimodal and paired samples reject split. Append never truncates recorded tokens. Without a decode context, decode returns text; with one, it returns the shared DecodedResponse.

Components accept template instances, subclasses, or registered names through set_template. HTTP sends only a registered name and construction settings, using the existing TemplateSpec for swift. Register SwiftTemplate through the existing swift worker startup hook before remote use; an unregistered name raises an error instead of falling back.

> Template does not support using functions as replacements because it needs to support many functions internally. If you need to write a new Template, please inherit the `Template` class.
> Generally speaking, using the Template base class is sufficient for pure text models. In the base class, we use tokenizer.apply_chat_template to encode the model, which is universal for general pure text models.

# Template mapping

Currently, the model-template mapping is simple:

- Template class: Supported in all pure text LLMs.
- DeepseekV4Template class: For DeepSeek V4, rewrites the chat template encoding logic, `encode_messages` is built into twinkle.
- Qwen3_5Template class: For Qwen3.5 MLLMs.
# Template

The template encodes a trajectory while retaining its complete dict payload. `InputFeature` is a model-input type hint, not a field projection: messages, tools, user_data and original images/videos/audios remain alongside numerical tensors. Non-model fields are filtered by the model's input processor.

`encode_prompt` returns the same complete dict. Multimodal trajectories carry serving `engine_input_ids` separately from training `input_ids`; text-only trajectories reuse `input_ids` when `engine_input_ids` is absent. `append_completion` appends actual generated tokens and returns a new trajectory with a structured assistant message; it does not re-encode the sampled history.

```python
class Template:

    def __init__(self,
                 model_id: str,
                 use_chat_template: bool = True,
                 max_length: Optional[int] = None,
                 truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise',
                 default_system: Optional[str] = None):
        ...

    def encode(self, sample=None, add_generation_prompt=None, *, intent=None,
               engine='vllm', return_length=False, **kwargs):
        ...

    def batch_encode(self, trajectories, add_generation_prompt=None, *, intent=None,
                     engine='vllm', return_length=False, **kwargs):
        ...

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        # Encode one sample and return the original sample
        # Generally used to check data reasonableness in RL algorithms like GRPO
        ...

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        # Batch check samples
        ...

    def decode(self, token_ids, *, context=None, **kwargs):
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

# Template mapping

Currently, the model-template mapping is simple:

- Template class：Supported in all pure text LLMs.
- Qwen3_5Template class: For Qwen3.5 MLLMs.
