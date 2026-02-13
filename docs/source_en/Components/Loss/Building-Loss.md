# Building New Loss

The loss base class in Twinkle is defined as:

```python
class Loss:

    def __call__(self, inputs: InputFeature, outputs: ModelOutput, **kwargs):
        ...
```

The loss input is the model's `InputFeature`, the output is the model's standard `ModelOutput`, and kwargs can be passed in the model's calculate_loss. Since it is a class with a `__call__` method, developers can also use Callable:


```python
def my_loss(inputs: InputFeature, outputs: ModelOutput, extra_data1: int, extra_data2: dict):
    ...
    return loss
```

Use it in the model like this:

```python
model.set_loss(my_loss)
model.calculate_loss(extra_data1=10, extra_data2={})
```

You can also upload the Loss to ModelScope/Hugging Face Hub and dynamically pull it when using:

```python
model.set_loss('ms://my_group/my_loss')
```

Please refer to the plugin documentation for specific details.
