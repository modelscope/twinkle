# 基本数据集组件

## 数据集标识

开源社区数据集可以由三个字段定义：

- 数据集名称：代表了数据集id，例如`swift/self-cognition`。
- 子数据集名称：一个数据集可能包含了多个子数据集，而且每个子数据集格式可能不同。
- 子数据集分片：常见分片有train/test等，用于训练、验证等。

使用Hugging Face社区的datasets库可以看到一个加载数据集的例子：

```python
from datasets import load_dataset
train_data = load_dataset("glue", "mrpc", split="train")
```

在twinkle的数据集输入中，使用`DatasetMeta`类来表达输入数据格式。该类包含：

```python
@dataclass
class DatasetMeta:
    dataset_id: str
    subset_name: str = 'default'
    split: str = 'train'
    data_slice: Iterable = None
```

前三个字段分别对应了数据集名称、子数据集名称、split，第四个字段`data_slice`是需要选择的数据范围，例如：

```python
dataset_meta = DatasetMeta(..., data_slice=range(100))
```

使用该类时开发者无需担心data_slice越界。twinkle会针对数据集长度进行重复取样。

> 注意：data_slice对流式数据集是没有效果的。


