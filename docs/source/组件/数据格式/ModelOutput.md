# 模型输入

twinkle用于表示模型输入的类是`InputFeature`，该类适配于transformers/megatron等模型结构。

```python
class ModelOutput(TypedDict, total=False):
    logits: OutputType
    loss: OutputType
```

ModelOutput本质上是一个Dict。其字段来自于模型的输出和loss计算。

- logits: 一般是[BatchSize * SequenceLength * VocabSize]尺寸，和labels配合计算loss
- loss: 实际残差

ModelOutput是twinkle中所有模型输出的标准接口。
