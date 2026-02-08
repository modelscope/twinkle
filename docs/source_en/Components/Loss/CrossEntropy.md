# Cross Entropy

Cross entropy is the most commonly used type of loss in model SFT and PT training. It is used for accurate probability fitting of labels.

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

The reduction parameter can be passed in during construction, supporting `sum`, `mean`, `none`, etc. (same as `torch.nn.CrossEntropyLoss` input).

> Currently using `sum` in Transformers models. The purpose is to count the number of valid tokens before optimizer.step and take the average of single tokens at the grad level.
