# Chunked Cross Entropy

A memory-efficient variant of cross-entropy loss that processes the vocabulary dimension in chunks to reduce peak GPU memory usage.

```python
from twinkle.loss import ChunkedCrossEntropyLoss

loss_fn = ChunkedCrossEntropyLoss(
    chunk_size=1024,           # vocabulary chunk size
    reduction='mean',
)

model.set_loss(loss_fn)
```

**Parameters:**
- `chunk_size`: Number of vocabulary tokens to process per chunk (default: 1024)
- `reduction`: Reduction mode — `sum`, `mean`, or `none`

The implementation uses a custom autograd function that splits the logit-to-loss computation into chunks along the vocabulary dimension. This avoids materializing the full `[batch*seq_len, vocab_size]` probability tensor, significantly reducing memory for large vocabularies.

> Useful when training with large vocabulary models where standard cross-entropy causes OOM errors.
