# MultiLoraTransformersModel

这个模型继承了TransformersModel，除提供了相同功能外，还提供了分时运行多个lora的能力，主要用于多租户训练。

```python
class MultiLoraTransformersModel:

    def __init__(self,  # noqa
                 model_cls = AutoModelForCausalLM,
                 model_id: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 grad_scaler_config: Dict[str, Any] = None,
                 max_loras: int = 5,
                 max_r: int = 32,
                 max_length: int = 8192,
                 **kwargs):
        ...

    ...
```

除了和基类相同的参数外，本类提供了几个额外参数用于多lora配置：
- max_loras: 最大lora的数量
- max_r: 最大的lora rank
- max_length: 最大的支持训练长度

之所以存在max_loras和max_r参数，是因为twinkle的多lora技术方案是在DDP wrap之前增加lora到`max_loras`个，防止后添加的lora无法接受DDP的管理。
正因如此，用户的r必须要小于等于max_r的配置，在实际训练时仅会使用lora的部分rank参与计算。

MultiLoraTransformersModel支持`@remote_class`注解，并且支持device_mesh，这意味着它可以运行在ray的worker中。

## 租户生命周期

底层使用 `MultiLora` 管理器来处理租户 LoRA 槽位。关键 API：

### acquire_lora

为租户获取一个可用的 LoRA 槽位：

```python
adapter_name = model.multi_lora.acquire_lora('tenant_a', LoraConfig(r=16, lora_alpha=32))
```

- 如果所有槽位已被占用或 `config.r > max_r`，则抛出 `RuntimeError`

### release_lora

释放租户的 LoRA 槽位，权重重置为初始状态：

```python
model.multi_lora.release_lora('tenant_a')
```

### 上下文管理器

使用 `adapter()` 进行作用域激活：

```python
with model.multi_lora.adapter('tenant_a') as name:
    output = model.forward(inputs)
```

### LoraTenant

每个槽位以 `LoraTenant` 数据类追踪：

```python
@dataclass
class LoraTenant:
    index: int                    # 槽位索引 (0..max_loras-1)
    adapter_name: str             # 内部名称（如 "lora_0"）
    config: LoraConfig            # 预分配配置（max_r）
    tenant_adapter_name: str      # 面向用户的租户名（空闲时为 None）
    tenant_config: LoraConfig     # 租户实际配置（空闲时为 None）
```
