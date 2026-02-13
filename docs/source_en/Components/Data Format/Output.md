# Model Output

Detailed type definition for model output.

## OutputType

OutputType defines the data types supported by model output:

```python
OutputType = Union[np.ndarray, 'torch.Tensor', List[Any]]
```

Supports NumPy arrays, PyTorch tensors, or lists of any type.

## ModelOutput

ModelOutput is the standard class used by Twinkle to represent model output. This class is adapted for model structures such as transformers/megatron.

```python
class ModelOutput(TypedDict, total=False):
    logits: OutputType
    loss: OutputType
```

ModelOutput is essentially a Dict. Its fields come from the model's output and loss calculation.

- logits: Generally [BatchSize * SequenceLength * VocabSize] size, used with labels to calculate loss
- loss: Actual residual

ModelOutput is the standard interface for all model outputs in Twinkle.

Usage example:

```python
from twinkle.data_format import ModelOutput

# In the model's forward method
def forward(self, inputs):
    ...
    return ModelOutput(
        logits=logits,
        loss=loss
    )
```

> Note: ModelOutput is defined using TypedDict, meaning it's a regular dict at runtime but provides type hints during type checking.
