# InputProcessor

InputProcessor 承载了不同任务的数据准备过程。

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
        # 移动到 cuda 设备上
        ...

    def pad_cp(self, inputs: List[InputFeature], **kwargs) ->List[InputFeature]:
        # 处理 cp
        ...

    def split_cp(self, inputs: List[Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        # 处理 cp
        ...

    def collate_fn(self, inputs: List[InputFeature], micro_batch_size: Optional[int] = None,
                   variable_seq_lengths=False, **kwargs) -> List[InputFeature]:
        # data_collator
        ...
```

- device_mesh: 用于切分 cp。如果没有 cp，device_mesh 参数可以不传。
- padding_free: 是否将多个样本拼接为一个，这个功能和 PackingDataset 比较相似，但 PackingDataset 会让每个 batch 长度基本一致，而 padding_free 仅考虑本 batch 内部的拼接。
  - 使用 PackingDataset 会自动在 InputProcessor 内触发 padding_free
- framework: 支持 transformers 和 megatron。不同的模型架构返回的模型输入略有不同

> Twinkle 将 collate_fn 放入 InputProcessor 中，因为不同的任务（sft/grpo 等）对输入需求是不同的。目前 InputProcessor 默认执行在模型端，因为这样可以将 DataLoader 和模型进行解耦。
> 因为 collate_fn 和运行任务、Megatron 的 micro_batch_size 等信息有关，如果在 DataLoader 中运行，会导致 DataLoader 无法独立成为组件，其逻辑也会变得复杂。

InputProcessor 实现了 __call__ 方法，因此你可以使用自己的 function 来完成自己的任务数据准备流程：

```python
def my_processor(inputs: Union[InputFeature, List[InputFeature]]) -> Union[InputFeature, List[InputFeature]]:
    return ...

model.set_processor(my_processor)
# 或者
dataloader.set_processor(my_processor)
```
