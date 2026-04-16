# 懒加载数据集

LazyDataset 是 `Dataset` 的变体，它将预处理、编码等开销较大的操作推迟到 `__getitem__` 时执行，从而避免大规模或多模态数据集的内存溢出问题。

## 与 Dataset 的关键差异

| 操作 | Dataset | LazyDataset |
|------|---------|-------------|
| `map` | 立即对所有数据执行 | 记录操作，在 `__getitem__` 中逐条执行 |
| `filter` | 立即执行 | 立即执行（与 Dataset 相同，需要构建索引映射） |
| `mix_dataset` | 立即合并数据集 | 记录策略，延迟解析索引 |
| `encode` | 立即编码所有数据 | 记录标志，在 `__getitem__` 中逐条编码 |

## 懒加载 Map

调用 `map` 时，LazyDataset 会记录预处理函数而非立即执行：

```python
from twinkle.dataset import LazyDataset, DatasetMeta

dataset = LazyDataset(DatasetMeta(dataset_id='ms://xxx/xxx'))
dataset.add_dataset(DatasetMeta(dataset_id='ms://yyy/yyy'))

# 按数据集的预处理（混合前）
dataset.map(preprocess_fn_a, dataset_meta=DatasetMeta(dataset_id='ms://xxx/xxx'))
dataset.map(preprocess_fn_b, dataset_meta=DatasetMeta(dataset_id='ms://yyy/yyy'))

dataset.mix_dataset()

# 全局预处理（混合后，对所有数据生效）
dataset.map(global_preprocess_fn)
```

- **混合前**：`map` 按数据集记录，不同数据集可以有不同的预处理流程。
- **混合后**：`map` 全局记录，对所有数据统一生效。
- 所有 map 操作在 `__getitem__` 中按注册顺序依次执行。

## 懒加载 Mix

`mix_dataset` 支持两种策略：

```python
dataset.mix_dataset(interleave=True)   # 轮询交错（默认）
dataset.mix_dataset(interleave=False)  # 顺序拼接
```

- **交错**：按轮询顺序从各数据集中取数据，较短的数据集会循环。
- **拼接**：按顺序访问——先取完数据集 A 的全部数据，再取数据集 B。

## 懒加载 Encode

调用 `encode` 仅标记需要编码，实际的 `template.encode()` 在 `__getitem__` 中执行：

```python
dataset.set_template('Qwen3_5Template', model_id='ms://Qwen/Qwen3.5-4B', max_length=512)
dataset.encode()
```

> 注意：LazyDataset 不支持 `truncation_strategy='split'`，因为分割可能从单条数据产生多条输出。

## 即时 Filter

与其他操作不同，`filter` 会立即执行，因为它需要预先构建有效数据项的索引映射：

```python
dataset.filter(filter_fn, dataset_meta=DatasetMeta(dataset_id='ms://xxx/xxx'))
```

## 远程执行

LazyDataset 拥有 `@remote_class` 装饰器，可以在 Ray Worker 中运行，与 `Dataset` 一致。
