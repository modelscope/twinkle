# DeviceMesh/DeviceGroup

这两个类用于表达硬件资源分配和网络拓扑，Twinkle 的数据分发和收集也依赖它们。

## DeviceGroup

```python
@dataclass
class DeviceGroup:
    name: str
    ranks: Union[List[int], int]
    device_type: str
    visible_devices: Optional[str] = None  # Optional: explicitly set visible devices (e.g., "8,9")
    gpus_per_worker: int = 1
```

- name: 资源组名
- ranks: 占用硬件列表，如果是CPU资源仅支持int类型
- device_type: 硬件类型，例如 GPU/CPU/NPU 等
- visible_devices: 可见资源列表，用于希望仅使用部分 rank 的硬件的情况
- gpus_per_worker: 每个 worker 占用多少硬件

如果训练 RL，开发者可以构造多个这样的组，并将对应的模型、采样器分配进入其中。

## DeviceMesh

DeviceMesh 承载了组件拓扑、分布式并行信息，这个类会在组件内传递，用于数据分发和数据收集。

```python
@dataclass
class DeviceMesh:
    ...

    @staticmethod
    def from_sizes(*, world_size: int = 1, dp_size: int = 1, fsdp_size: int = None, tp_size: int = None,
                   pp_size: int = None, ulysses_size: int = None, cp_size: int = None, ep_size: int = None,
                   etp_size: int = None,vpp_size: int = None, device_type: str = 'cuda', sequence_parallel: bool = False) -> "DeviceMesh":
        ...
```

推荐使用 `from_sizes` 来构造它。

我们举一个例子：

```python
sampler_device_mesh = DeviceMesh.from_sizes(dp_size=4)
actor_device_mesh = DeviceMesh.from_sizes(dp_size=2, pp_size=2, tp_size=2)

dataloader = DataLoader(...)
sampler = vLLMSampler(..., device_mesh=sampler_device_mesh, remote_group=...)
actor = MegatronModel(..., device_mesh=actor_device_mesh, remote_group=...)

for data in dataloader:
    sampler_output = sampler.sample(data)
    model_output = actor.forward(sampler_output)
```

我们以上面的伪代码来分析数据传递情况。

dataloader 取出数据 -> 按照 dp_size=4 分发给 sampler -> 按照 dp_size=4 收集数据 -> 按照 dp_size=2 分发给模型 -> 按照 dp_size=2 收集输出

通过 DeviceMesh，可以将数据流平顺地在各个 group 和组件之间流转起来。

数据的分发判断由 DeviceMesh 的 `get_slice` 方法执行：

```python
batch[device_mesh.get_slice(len(batch))]
```

get_slice 会根据当前 rank，计算出当前 worker 属于哪个 dp 组，并获取对应的数据。该过程发生在 DataLoader 的 DeviceMeshSampler 中，同样发生在 remote_class 的 dispatch 和 collect 中。
