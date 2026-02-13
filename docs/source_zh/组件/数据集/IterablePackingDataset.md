# 流式固定长度装箱数据集

`IterablePackingDataset`和`PackingDataset`一样，同样用于数据集的自动拼接装箱。不同的是`IterablePackingDataset`适配于大数据集或多模态场景下的流式读取。

本数据集同样需要额外调用`pack_dataset()`来开启装箱过程。
```python
dataset.pack_dataset()
```

本数据集也有`@remote_class`装饰器，可以在ray的worker中运行。


