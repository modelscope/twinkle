# Streaming Dataset

Streaming datasets are used to load datasets in a streaming manner, generally used for ultra-large-scale datasets or multimodal datasets to save memory usage. Streaming datasets have no index and length, and can only be accessed through iterators.

Twinkle's streaming dataset methods are the same as `Dataset`. However, since it does not provide `__getitem__` and `__len__` methods, streaming datasets need to use `next` for access:

```python
from twinkle.dataset import IterableDataset, DatasetMeta

dataset = IterableDataset(DatasetMeta(...))
trajectory = next(dataset)
```

Streaming datasets also have the `@remote_class` decorator and can run in Ray workers.
