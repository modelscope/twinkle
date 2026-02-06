from cookbook.grpo.lora_gpu import actor_device_mesh

# DataLoader

DataLoader是PyTorch中用于加载处理后的数据集，并提供数据给模型的组件。该组件的工作流程为：

传入数据集 -> 构建sampler和batch_sampler-> 索引数据 -> 调用sampler拿到索引 -> 从dataset中取出一个batch -> 进行collate_fn操作 -> 吐出数据

DataLoader的整体工作方式类似于：

```python
for data in dataloader:
    ...
```

可以看出dataloader包含`__iter__`方法，返回一个迭代器出来。在DDP、TP、Ulysses等不同训练条件下，由于每个rank取出的数据不同，因此一般sampler有多种实现，较为复杂。

在twinkle中，我们采取了一个非常简单直接的方案，将`DeviceMesh`传递给DataLoader，由于DeviceMesh中包含了集群结构，因此DeviceMesh可以给出所有rank需要的数据分片。
因此我们额外开发了`DeviceMeshSampler`和`DeviceMeshFetcher`，分别用于普通数据集和流式数据集两类的取样工作。
另外，由于LazyDataset的存在，导致数据集实际取出数据时可能包含了无效数据或者抛出异常，因此提供了`RetrySampler`来进行跳过和重试。

DataLoader的使用非常简单：

```python
dataloader = DataLoader(dataset)
for data in dataloader:
    ...
```
在torchrun条件下，由于整体同构，因此全局只需要一个device_mesh，这个参数无需通过DataLoader的构造传入，infra模块会自动分析并传入。

DataLoader也支持在ray模式下工作：
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

DataLoader的dataset参数可以传入一个Callable来返回一个Dataset，这样可以做到数据集的构建代码放在driver中，但实际的构建在Dataloader的worker中，防止了跨进程的pickle，提高速度。
dataloader的`@remote_class`装饰器的执行范围也是`first`，这意味着它只会有一个worker用来取出数据。

> 开发者无须担心dataloader返回的data占用driver内存，data通常是一个引用句柄，到了需要使用的worker才会实际传递并解包。