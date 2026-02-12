# MultiLoraMegatronModel

This model inherits from MegatronModel. In addition to providing the same functions, it also provides the ability to run multiple loras in time-sharing, mainly used for multi-tenant training.

```python
class MultiLoraMegatronModel:

    def __init__(self,  # noqa
                 model_id: str,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp16', 'bf16'] = 'bf16',
                 max_loras: int = 5,
                 max_r: int = 32,
                 max_length: int = 8192,
                 **kwargs):
        ...
    
    ...
```

In addition to the same parameters as the base class, this class provides several additional parameters for multi-lora configuration:
- max_loras: Maximum number of loras
- max_r: Maximum lora rank
- max_length: Maximum supported training length

The reason for the existence of max_loras and max_r parameters is that Twinkle's multi-lora technical solution is to add loras to `max_loras` before DDP wrap to prevent later added loras from being unable to accept DDP management.
Because of this, the user's r must be less than or equal to the max_r configuration. During actual training, only part of the lora's rank will be used in the calculation.

MultiLoraMegatronModel supports the `@remote_class` annotation and supports device_mesh, which means it can run in Ray workers.
