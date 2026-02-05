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

这个原始数据中，response可能包含了不规范的信息，在开始训练前需要对response进行过滤和修复，并更换为twinkle标准的格式。于是可以编写一个方法处理对应的数据：

```python
from twinkle.data_format import Trajectory, Message
from twinkle.dataset import DatasetMeta
def preprocess_row(row):
    query = row['query']
    response = row['response']
    if not query or not response:
        return None
    # Fix response
    response = _do_some_fix_on_response(response)
    return Trajectory(
        messages=[
            Message(role='user', content=query),
            Message(role='assistant', content=response)
        ]
    )

dataset.map(preprocess_row, dataset_meta=DatasetMeta(dataset_id='ms://xxx/xxx'))
```

> 提示:
> 1. 目前Dataset的map接口不支持`batched=True`方式
> 2. 如果某个row有问题，返回None，dataset.map会自动过滤空行
> 3. 不同的数据集预处理方式可能不同，因此需要额外传递`dataset_meta`参数。如果没有调用过`add_dataset`方法，即Dataset中只有一个数据集的时候，本参数可以省略

同理，Dataset提供了filter方法：
```python
def filter_row(row):
    if ...:
        return False
    else:
        return True

dataset.filter(filter_row, dataset_meta=DatasetMeta(dataset_id='ms://xxx/xxx'))
```

5. 混合数据集

当你在Dataset中增加了多个数据集之后，需要使用`mix_dataset`来混合它们。

```python
dataset.mix_dataset()
```

6编码数据集

数据集在输入模型前，一定会经过分词和编码过程转换为token。这个过程通常由`tokenizer`组件完成。但在现在大模型训练过程中，一般不会直接使用tokenizer，这是因为模型的训练需要额外的字段准备，仅进行tokenizer.encode过程不足以完成。
在twinkle中，编码数据集由Template组件来完成。上面已经讲述了如何设置Template，下面可以直接进行encode：

```python
dataset.encode()
```

> 1. Dataset的`map` `encode` `filter`等方法均使用`datasets`的`map`方式进行，因此在对应方法的kwargs中均可以使用对应的参数
> 2. `load_from_cache_file`参数默认为False，因为该参数设置为True时会引发一些数据集改变但训练仍然使用缓存的头痛问题。如果你的数据集较大而且更新不频繁，可以直接置为True
> 3. encode不需要指定`DatasetMeta`，因为预处理过后所有数据集格式都是相同的

6. 获取数据

同普通数据集一样，twinkle的`Dataset`可以通过索引来使用数据。

```python
trajectory = dataset[0]
length = len(dataset)
```

7. 远程运行支持

`Dataset`类标记了`@remote_class`装饰器，因此可以在ray中运行：

```python
dataset = Dataset(..., remote_group='actor_group')
# 下面的方法会运行在ray worker上
dataset.map(...)
```

> 整体数据集的使用流程是：
> 1. 构造数据集，如果需要ray worker中运行则传入remote_group参数
> 2. 设置template
> 3. 预处理数据
> 4. 如果增加了多个数据集，混合数据
> 5. encode数据




