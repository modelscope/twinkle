# 流式数据集

流式数据集用于将数据集按照流的方式加载，一般用于超大规模数据集或者多模态数据集上用以节省内存使用。流式数据集没有索引和长度，只能通过迭代器访问。

twinkle的流式数据集和`Dataset`的方法都是相同的。但由于不提供`__getitem__`和`__len__`方法，因此流式数据集的使用需要使用`next`:

```python
from twinkle.dataset import IterableDataset, DatasetMeta

dataset = IterableDataset(DatasetMeta(...))
trajectory = next(dataset)
```

流式数据集也有`@remote_class`装饰器，可以在ray的worker中运行。



