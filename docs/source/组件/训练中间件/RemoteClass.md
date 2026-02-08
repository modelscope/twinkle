# RemoteClass

所有 Twinkle 中支持 Ray 和 HTTP 中使用的组件均通过 `@remote_class` 和 `@remote_function` 进行了装饰。该装饰器会拦截类的构造，在 Ray 模式下，将类的构造转为 worker 执行。

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

开发者只需要编写上述代码，就可以将 `MyComponent` 类转入 worker 执行。其中：

- remote_class: 将类标记为需要远端执行。如果 Twinkle 初始化设置为 `local` 模式，或者该类构造时没有传入 `remote_group` 设置，或者 `remote_group` 为当前 worker，都会在进程内构造该类。
- remote_function: 将某个标记了 `remote_class` 的方法标记为可以在 Ray 中执行。其输入和输出均会被 Ray 压缩传递。

调用 `MyComponent`：

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

通过这种方式，我们编写了一个 `MyComponent`，并在 Ray 集群中使用 4 张卡构造了一个叫 `default` 的组，把 `MyComponent` 构造在了该组中。

remote_class 在装饰类的时候的参数：

- execute: 支持 first/all。first 仅会在该组的第 0 个设备上创建，一般用于 Dataset、DataLoader 的构造，all 会在所有设备上构造。

remote_function 在装饰方法的时候有下面的参数：

- dispatch: 如何分发输入数据。支持 slice/all/slice_dp/函数 四种。slice 会将 list 输入均匀分发（非 list 会全部分发），all 进行全部分发，slice_dp 会将输入数据按照 device_mesh 的 dp 组进行切分分发，来保障模型输入数据的正确性，函数方式支持以自己的实现来分发输入数据：

```python
def _dispatcher(length, i, args, kwargs):
    # length 是 worker 数量，i 是当前 rank，args 和 kwargs 是输入数据，在这里具体执行分发逻辑
    return _args_rank, _kwargs_rank
```

- execute: 支持 first/all，仅在第一个 worker 上执行，还是全部执行
- collect: 如何收集返回的数据，支持 none/flatten/mean/sum/first/last_pp/函数
  - none: 不做任何处理
  - flatten: 将所有 worker 数据进行拉平，模仿单一 worker 执行的返回结构
  - mean/sum: 返回均值或累加值
  - first: 仅返回第一个 worker 的结果。一般用于所有 worker 需要输入，但输出结果相同的情况
  - last_pp: 返回最后一个 pipeline 的结果，用于 pp 并行的情况
  - 函数: 支持自定义收集方法

```python
def _collect(all_results: List):
    return ...
```

- sync: 是否以 Ray 的同步方式执行，默认为 `False`
- lazy_collect: 默认为 True，在这种情况下，会不在 driver 进程中收集结果，而在需要这些结果的 worker 中延迟展开，对于具体方法来说，某些方法需要在 driver 中收集，例如收集 loss、metric 等网络负载不大的情况，可以设置为 False
