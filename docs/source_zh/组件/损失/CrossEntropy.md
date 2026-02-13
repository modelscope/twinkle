# 交叉熵

交叉熵是模型SFT和PT训练中最常用的一类损失。用于对labels的精确概率拟合。

```python
class CrossEntropyLoss(Loss):

    def __init__(self, **kwargs):
        self.reduction = kwargs.get('reduction', 'mean')

    def __call__(self, inputs, outputs, **kwargs):
        import torch
        logits = outputs['logits'].view(-1, outputs['logits'].shape[-1])
        labels = inputs['labels'].view(-1)
        return torch.nn.CrossEntropyLoss(reduction=self.reduction)(logits, labels)
```

构造中可以传入reduction参数，支持`sum`, `mean`, `none`等（和`torch.nn.CrossEntropyLoss`输入相同）。

> 在Transformers模型中目前使用`sum`。目的是在optimizer.step之前统计有效token数量并在grad层面取单token平均。
