# DataLoader

DataLoader 是 PyTorch 中用于加载处理后的数据集，并提供数据给模型的组件。该组件的工作流程为：

传入数据集 -> 构建 sampler 和 batch_sampler -> 索引数据 -> 调用 sampler 拿到索引 -> 从 dataset 中取出一个 batch -> 进行 collate_fn 操作 -> 吐出数据

DataLoader 的整体工作方式类似于：

```python
for data in dataloader:
    ...
```

可以看出 dataloader 包含 `__iter__` 方法，返回一个迭代器出来。在 DDP、TP、Ulysses 等不同训练条件下，由于每个 rank 取出的数据不同，因此一般 sampler 有多种实现，较为复杂。

在 Twinkle 中，我们采取了一个非常简单直接的方案，将 `DeviceMesh` 传递给 DataLoader，由于 DeviceMesh 中包含了集群结构，因此 DeviceMesh 可以给出所有 rank 需要的数据分片。
因此我们额外开发了 `DeviceMeshSampler` 和 `DeviceMeshFetcher`，分别用于普通数据集和流式数据集两类的取样工作。
另外，由于 LazyDataset 的存在，导致数据集实际取出数据时可能包含了无效数据或者抛出异常，因此提供了 `RetrySampler` 来进行跳过和重试。

DataLoader 的使用非常简单：

```python
dataloader = DataLoader(dataset)
for data in dataloader:
    ...
```
在 torchrun 条件下，由于整体同构，因此全局只需要一个 device_mesh，这个参数无需通过 DataLoader 的构造传入，infra 模块会自动分析并传入。

DataLoader 也支持在 Ray 模式下工作：
```python

def create_dataset():
    dataset = Dataset(...)
    dataset.map(...)
    dataset.encode(...)
    return dataset

dataloader = DataLoader(create_dataset, device_mesh=actor_device_mesh, remote_group='actor')
for data in dataloader:
    ...
```

DataLoader 的 dataset 参数可以传入一个 Callable 来返回一个 Dataset，这样可以做到数据集的构建代码放在 driver 中，但实际的构建在 Dataloader 的 worker 中，防止了跨进程的 pickle，提高速度。
dataloader 的 `@remote_class` 装饰器的执行范围也是 `first`，这意味着它只会有一个 worker 用来取出数据。

> 开发者无需担心 dataloader 返回的 data 占用 driver 内存，data 通常是一个引用句柄，到了需要使用的 worker 才会实际传递并解包。
> Dataloader 默认不设置任何的 collate_fn，而是将这个过程交由模型处理。