# InputProcessor

InputProcessor carries the data preparation process for different tasks.

```python
class InputProcessor:

    def __init__(self, device_mesh: Optional[DeviceMesh] = None,
                 padding_free: bool = False,
                 framework: Literal['transformers', 'megatron'] = 'transformers',
                 **kwargs):
        ...

    def __call__(self, inputs: Union[InputFeature, List[InputFeature]], **kwargs) -> Union[InputFeature, List[InputFeature]]:
        # Overall processing entry point
        ...
    
    def prepare_inputs(self, inputs: Union[List[InputFeature], InputFeature], **kwargs) -> List[InputFeature]:
        # Move to cuda device
        ...

    def pad_cp(self, inputs: List[InputFeature], **kwargs) ->List[InputFeature]:
        # Handle cp
        ...

    def split_cp(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        # Handle cp
        ...

    def collate_fn(self, inputs: List[InputFeature], micro_batch_size: Optional[int] = None,
                   variable_seq_lengths=False, **kwargs) -> List[InputFeature]:
        # data_collator
        ...
```

- device_mesh: Used to split cp. If there is no cp, the device_mesh parameter can be omitted.
- padding_free: Whether to concatenate multiple samples into one. This function is similar to PackingDataset, but PackingDataset makes the length of each batch basically consistent, while padding_free only considers concatenation within this batch.
  - Using PackingDataset will automatically trigger padding_free in InputProcessor
- framework: Supports transformers and megatron. Different model architectures return slightly different model inputs

> Twinkle places collate_fn in InputProcessor because different tasks (sft/grpo, etc.) have different input requirements. Currently, InputProcessor is executed on the model side by default, because this decouples DataLoader and the model.
> Because collate_fn is related to the running task, Megatron's micro_batch_size and other information, if run in DataLoader, it will cause DataLoader to be unable to become an independent component, and its logic will also become complex.

InputProcessor implements the __call__ method, so you can use your own function to complete your own task data preparation process:

```python
def my_processor(inputs: Union[InputFeature, List[InputFeature]]) -> Union[InputFeature, List[InputFeature]]:
    return ...

model.set_processor(my_processor)
# Or
dataloader.set_processor(my_processor)
```
