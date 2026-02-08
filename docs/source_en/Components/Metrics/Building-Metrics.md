# Building Metrics

Metrics are used to measure the training process and training results. The metric component is part of the customizable components.

```python
class Metric:

    def __init__(self, device_mesh, process_group, **kwargs):
        self.process_group = process_group
        self.device_mesh = device_mesh
    
    # Due to the existence of microbatch, the inputs to Metric may be a List
    def accumulate(self, inputs: 'Union[InputFeature, List[InputFeature]]', outputs: 'ModelOutput'):
        ...

    def calculate(self):
        ...

    def reset(self):
        ...
```

Metrics cannot be passed in through Callable. Because it contains two parts: `accumulate` and `calculate`, and needs to support `reset` to zero out. The device_mesh and process_group belonging to the current dp group are automatically passed in during the construction of the metric for cross-process communication.
Moreover, in the actual implementation, the base class provides a `gather_results` method to assist in collecting input results from various processes.
