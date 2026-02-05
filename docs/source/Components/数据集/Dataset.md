# 基本数据集组件

## DatasetMeta

开源社区的数据集可以由三个字段定义：

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

## Dataset

twinkle的Dataset是实际数据集的浅封装，包含了下载、加载、混合、预处理、encode等操作。

1. 数据集的加载

```python
from twinkle.dataset import Dataset, DatasetMeta

dataset = Dataset(DatasetMeta(dataset_id='ms://swift/self-cognition', data_slice=range(1500)))
```
数据集的`ms://`前缀代表了从ModelScope社区下载，如果替换为`hf://`会从Hugging Face社区下载。如果没有前缀则默认从Hugging Face社区下载。你也可以传递一个本地路径：

```python
from twinkle.dataset import Dataset, DatasetMeta

dataset = Dataset(DatasetMeta(dataset_id='my/custom/dataset.jsonl', data_slice=range(1500)))
```

2. 设置template

Template组件是负责字符串/图片多模态原始数据转换为模型输入token的组件。数据集可以设置一个Template来完成`encode`过程。

```python
dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-7B-Instruct', max_length=512)
```

set_template方法支持传入`kwargs`（例如例子中的`max_length`），作为`Template`的构造参数使用。

3. 增加数据集

```python
dataset.add_dataset(DatasetMeta(dataset_id='ms://xxx/xxx', data_slice=range(1000)))
```

`add_dataset`可以在已有数据集基础上增加其他数据集，并在后续调用`mix_dataset`将它们混合起来。

4. 预处理数据

预处理数据（ETL）过程是数据清洗和标准化的重要流程。例如：

```json
{
  "query": "some query here",
  "response": "some response with extra info",
}
```

这个原始数据中，response可能包含了不规范的信息，在开始训练前需要对response进行过滤和改变


