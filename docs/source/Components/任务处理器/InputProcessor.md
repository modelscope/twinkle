# InputProcessor

InputProcessor承载了不同任务的数据准备过程。

```python
class InputProcessor:

    def __init__(self, device_mesh: Optional[DeviceMesh] = None,
                 padding_free: bool = False,
                 framework: Literal['transformers', 'megatron'] = 'transformers',
                 **kwargs):
        ...

    def __call__(self, inputs: Union[InputFeature, List[InputFeature]], **kwargs) -> Union[InputFeature, List[InputFeature]]:
        # 整体处理的入口
        ...
    
    def prepare_inputs(self, inputs: Union[List[InputFeature], InputFeature], **kwargs) -> List[InputFeature]:
        # 移动到cuda设备上
        ...

    def pad_cp(self, inputs: List[InputFeature], **kwargs) ->List[InputFeature]:
        # 处理cp
        ...

    def split_cp(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        # 处理cp
        ...

    def collate_fn(self, inputs: List[InputFeature], micro_batch_size: Optional[int] = None,
                   variable_seq_lengths=False, **kwargs) -> List[InputFeature]:
        # data_collator
        ...
```

- device_mesh: 用于切分cp。如果没有cp，device_mesh参数可以不传。
- padding_free: 是否将多个样本拼接为一个，这个功能和PackingDataset比较相似，但PackingDataset会让每个batch长度基本一致，而padding_free仅考虑本batch内部的拼接。
  - 使用PackingDataset会自动在InputProcessor内出发padding_free
- framework: 支持transformers和megatron。不同的模型架构返回的模型输入略有不同

> twinkle将collate_fn放入InputProcessor中，因为不同的任务(sft/grpo等)对输入需求是不同的。目前InputProcessor默认执行在模型端，因为这样可以将DataLoader和模型进行解耦。
> 因为collate_fn和运行任务、megatron的micro_batch_size等信息有关，如果在DataLoader中运行，会导致DataLoader无法独立成为组件，其逻辑也会变得复杂。

InputProcessor实现了__call__方法，因此你可以使用自己的function来完成自己的任务数据准备流程：

```python
def my_processor(inputs: Union[InputFeature, List[InputFeature]]) -> Union[InputFeature, List[InputFeature]]:
    return ...

model.set_processor(my_processor)
# 或者
dataloader.set_processor(my_processor)
```
