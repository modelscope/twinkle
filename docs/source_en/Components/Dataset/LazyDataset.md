# Lazy Loading Dataset

LazyDataset is a variant of `Dataset` that defers expensive operations (preprocessing, encoding) to `__getitem__` time, preventing OOM for large or multimodal datasets.

## Key Differences from Dataset

| Operation | Dataset | LazyDataset |
|-----------|---------|-------------|
| `map` | Executes immediately on all data | Records the operation, applies per-item in `__getitem__` |
| `filter` | Executes immediately | Executes immediately (same as Dataset, index mapping required) |
| `mix_dataset` | Merges datasets immediately | Records strategy, resolves indices lazily |
| `encode` | Encodes all data immediately | Records flag, encodes per-item in `__getitem__` |

## Lazy Map

When you call `map`, LazyDataset records the preprocessing function instead of applying it eagerly:

```python
from twinkle.dataset import LazyDataset, DatasetMeta

dataset = LazyDataset(DatasetMeta(dataset_id='ms://xxx/xxx'))
dataset.add_dataset(DatasetMeta(dataset_id='ms://yyy/yyy'))

# Per-dataset preprocessing (before mix)
dataset.map(preprocess_fn_a, dataset_meta=DatasetMeta(dataset_id='ms://xxx/xxx'))
dataset.map(preprocess_fn_b, dataset_meta=DatasetMeta(dataset_id='ms://yyy/yyy'))

dataset.mix_dataset()

# Global preprocessing (after mix, applies to all items)
dataset.map(global_preprocess_fn)
```

- **Before mix**: `map` is recorded per-dataset, so different datasets can have different preprocessing pipelines.
- **After mix**: `map` is recorded globally and applies to all items regardless of source dataset.
- All map operations are applied lazily in `__getitem__` in the order they were registered.

## Lazy Mix

`mix_dataset` supports two strategies:

```python
dataset.mix_dataset(interleave=True)   # Round-robin interleaving (default)
dataset.mix_dataset(interleave=False)  # Concatenation
```

- **Interleave**: Items cycle through datasets in round-robin order. Shorter datasets wrap around.
- **Concatenate**: Items are accessed sequentially — all of dataset A, then all of dataset B.

## Lazy Encode

Calling `encode` only marks the dataset for encoding. The actual `template.encode()` call happens inside `__getitem__`:

```python
dataset.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=512)
dataset.encode()
```

> Note: `truncation_strategy='split'` is not supported in LazyDataset because splitting may produce multiple outputs from a single item.

## Eager Filter

Unlike other operations, `filter` executes immediately because it needs to build the index mapping of valid items upfront:

```python
dataset.filter(filter_fn, dataset_meta=DatasetMeta(dataset_id='ms://xxx/xxx'))
```

## Remote Execution

LazyDataset has the `@remote_class` decorator and can run in Ray workers, just like `Dataset`.
