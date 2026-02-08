# Model Output

The class used by Twinkle to represent model output is `ModelOutput`, which is adapted to model structures such as transformers/megatron.

```python
class ModelOutput(TypedDict, total=False):
    logits: OutputType
    loss: OutputType
```

ModelOutput is essentially a Dict. Its fields come from the model's output and loss calculation.

- logits: Generally [BatchSize * SequenceLength * VocabSize] size, used with labels to calculate loss
- loss: Actual residual

ModelOutput is the standard interface for all model outputs in Twinkle.
