# DataLoader

DataLoader is a component in PyTorch used to load processed datasets and provide data to the model. The workflow of this component is:

Input dataset -> Build sampler and batch_sampler -> Index data -> Call sampler to get indices -> Extract a batch from dataset -> Perform collate_fn operation -> Output data

The overall working method of DataLoader is similar to:

```python
for data in dataloader:
    ...
```

As you can see, dataloader contains the `__iter__` method, returning an iterator. Under different training conditions such as DDP, TP, Ulysses, etc., since each rank extracts different data, samplers generally have multiple implementations and are relatively complex.

In Twinkle, we adopted a very simple and direct approach by passing `DeviceMesh` to the DataLoader. Since DeviceMesh contains the cluster structure, DeviceMesh can provide the data shards needed by all ranks.
Therefore, we additionally developed `DeviceMeshSampler` and `DeviceMeshFetcher`, which are used for sampling work of ordinary datasets and streaming datasets respectively.
Additionally, due to the existence of LazyDataset, when the dataset actually extracts data, it may contain invalid data or throw exceptions, so we provide `RetrySampler` for skipping and retrying.

Using DataLoader is very simple:

```python
dataloader = DataLoader(dataset)
for data in dataloader:
    ...
```
Under torchrun conditions, since the overall structure is homogeneous, only one global device_mesh is needed. This parameter does not need to be passed in through the DataLoader constructor; the infra module will automatically analyze and pass it in.

DataLoader also supports working in Ray mode:
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

The dataset parameter of DataLoader can pass in a Callable to return a Dataset. This way, the dataset construction code can be placed in the driver, but the actual construction is in the Dataloader's worker, preventing cross-process pickle and improving speed.
The execution scope of the dataloader's `@remote_class` decorator is also `first`, which means it will only have one worker to extract data.

> Developers don't need to worry about the data returned by dataloader occupying driver memory. Data is usually a reference handle, and it will only be actually transferred and unpacked when it reaches the worker that needs to use it.
> Dataloader does not set any collate_fn by default, but instead hands this process over to the model for handling.
