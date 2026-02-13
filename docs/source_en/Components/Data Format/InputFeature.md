# Model Input

The class used by Twinkle to represent model input is `InputFeature`, which is adapted to model structures such as transformers/megatron.

```python
InputType = Union[List[List[int]], List[int], np.ndarray, Any]

class InputFeature(TypedDict, total=False):
    # Text-related fields
    input_ids: InputType
    attention_mask: InputType
    position_ids: InputType
    labels: InputType
```

InputFeature is essentially a Dict. Its input comes from the output of the `Template` component.

- input_ids: Token list after List[Messages] is nested with a template
- attention_mask: Attention mask
- position_ids: Position encoding for sample distinction
- labels: Training labels, which have already undergone a one-token left shift

In the case of packing or padding_free, fields such as input_ids are concatenated from lists of multiple samples.
In multimodal scenarios, InputFeature contains other multimodal fields.

InputFeature is the standard interface for all template outputs and model inputs in Twinkle.
