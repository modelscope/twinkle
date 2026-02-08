# RemoteClass

All components in Twinkle that support use in Ray and HTTP are decorated with `@remote_class` and `@remote_function`. This decorator intercepts the construction of the class and, in Ray mode, converts the class construction to worker execution.

```python
from twinkle import remote_class, remote_function

@remote_class(execute='first')
class MyComponent:
    
    def __init__(self, **kwargs):
        ...
    
    @remote_function(dispatch='slice_dp', collect='first')
    def func(self, *args, **kwargs):
        ...
        return ...
```

Developers only need to write the above code to transfer the `MyComponent` class to worker execution. Among them:

- remote_class: Marks the class as needing remote execution. If Twinkle initialization is set to `local` mode, or if the class construction does not pass in a `remote_group` setting, or if `remote_group` is the current worker, the class will be constructed within the process.
- remote_function: Marks a method of a class marked with `remote_class` as executable in Ray. Its input and output will be compressed and passed by Ray.

Calling `MyComponent`:

```python
import twinkle
from twinkle import DeviceGroup

device_groups = [
    DeviceGroup(
        name='default',
        ranks=4,
        device_type='cuda',
    )
]

twinkle.initialize('ray', groups=device_groups)

_my_component = MyComponent(remote_group='default')
_my_component.func(...)
```

In this way, we wrote a `MyComponent` and constructed a group called `default` using 4 GPUs in the Ray cluster, and constructed `MyComponent` in that group.

Parameters when remote_class decorates a class:

- execute: Supports first/all. first will only be created on the 0th device of the group, generally used for the construction of Dataset and DataLoader. all will be constructed on all devices.

Parameters when remote_function decorates a method:

- dispatch: How to distribute input data. Supports four types: slice/all/slice_dp/function. slice will evenly distribute list input (non-list will be fully distributed), all performs full distribution, slice_dp will split and distribute the input data according to the dp group of device_mesh to ensure the correctness of model input data. The function method supports distributing input data with your own implementation:

```python
def _dispatcher(length, i, args, kwargs):
    # length is the number of workers, i is the current rank, args and kwargs are input data, execute the distribution logic here
    return _args_rank, _kwargs_rank
```

- execute: Supports first/all, execute only on the first worker, or execute on all
- collect: How to collect returned data, supports none/flatten/mean/sum/first/last_pp/function
  - none: Do not process anything
  - flatten: Flatten all worker data to mimic the return structure of single worker execution
  - mean/sum: Return average or cumulative value
  - first: Only return the result of the first worker. Generally used when all workers need input, but the output results are the same
  - last_pp: Return the result of the last pipeline, used for pp parallelism
  - function: Supports custom collection methods

```python
def _collect(all_results: List):
    return ...
```

- sync: Whether to execute synchronously using Ray's method, default is `False`
- lazy_collect: Default is True. In this case, results will not be collected in the driver process, but will be delayed and expanded in the workers that need these results. For specific methods, some methods need to be collected in the driver, such as collecting loss, metrics and other situations with small network load, which can be set to False
