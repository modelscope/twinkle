# 模型输出

模型输出的详细类型定义。

## OutputType

OutputType 定义了模型输出支持的数据类型:

```python
OutputType = Union[np.ndarray, 'torch.Tensor', List[Any]]
```

支持 NumPy 数组、PyTorch 张量或任意类型的列表。

## ModelOutput

ModelOutput 是 Twinkle 用于表示模型输出的标准类。该类适配于 transformers/megatron 等模型结构。

```python
class ModelOutput(TypedDict, total=False):
    logits: OutputType
    loss: OutputType
```

ModelOutput 本质上是一个 Dict。其字段来自于模型的输出和 loss 计算。

- logits: 一般是 [BatchSize * SequenceLength * VocabSize] 尺寸,和 labels 配合计算 loss
- loss: 实际残差

ModelOutput 是 Twinkle 中所有模型输出的标准接口。

使用示例:

```python
from twinkle.data_format import ModelOutput

# 在模型的 forward 方法中
def forward(self, inputs):
    ...
    return ModelOutput(
        logits=logits,
        loss=loss
    )
```

> 注意:ModelOutput 使用 TypedDict 定义,意味着它在运行时是一个普通的 dict,但在类型检查时会提供类型提示。
