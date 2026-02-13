# 模型输入

twinkle用于表示模型输入的类是`InputFeature`，该类适配于transformers/megatron等模型结构。

```python
InputType = Union[List[List[int]], List[int], np.ndarray, Any]

class InputFeature(TypedDict, total=False):
    # Text-related fields
    input_ids: InputType
    attention_mask: InputType
    position_ids: InputType
    labels: InputType
```

InputFeature本质上是一个Dict。其输入来自于`Template`组件的输出。

- input_ids: List[Messages]以模板进行嵌套之后的token list
- attention_mask: 注意力掩膜
- position_ids: 用于样本区分的位置编码
- labels: 训练的label，已经进行了一个token的左位移

在packing或padding_free的情况下，input_ids等字段由多个样本的列表拼接而来。
在多模态场景下，InputFeature包含多模态其他字段。

InputFeature是twinkle中所有模板输出、模型输入的标准接口。
