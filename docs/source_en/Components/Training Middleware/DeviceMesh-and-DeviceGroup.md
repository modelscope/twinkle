# DeviceMesh/DeviceGroup

These two classes are used to express hardware resource allocation and network topology. Twinkle's data distribution and collection also depend on them.

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

- name: Resource group name
- ranks: Occupied hardware list, only supports int type for CPU resources
- device_type: Hardware type, such as GPU/CPU/NPU, etc.
- visible_devices: Visible resource list, used when you only want to use part of the rank's hardware
- gpus_per_worker: How much hardware each worker occupies

If training RL, developers can construct multiple such groups and assign corresponding models and samplers into them.

## DeviceMesh

DeviceMesh carries component topology and distributed parallel information. This class is passed within components for data distribution and data collection.

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

It is recommended to use `from_sizes` to construct it.

Let's give an example:

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

We analyze the data transfer situation using the pseudo-code above.

dataloader fetches data -> distributes to sampler according to dp_size=4 -> collects data according to dp_size=4 -> distributes to model according to dp_size=2 -> collects output according to dp_size=2

Through DeviceMesh, data flow can be smoothly transferred between various groups and components.

Data distribution judgment is performed by the `get_slice` method of DeviceMesh:

```python
batch[device_mesh.get_slice(len(batch))]
```

get_slice calculates which dp group the current worker belongs to based on the current rank and obtains the corresponding data. This process occurs in the DeviceMeshSampler of DataLoader, and also in the dispatch and collect of remote_class.
