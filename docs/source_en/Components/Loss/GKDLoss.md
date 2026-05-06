# GKD Loss

Generalized Knowledge Distillation (GKD) loss uses Jensen-Shannon Divergence for distilling knowledge from a teacher model to a student model.

```python
from twinkle.loss import GKDLoss

loss_fn = GKDLoss(
    teacher_mode='full',  # 'full', 'topk_local', 'topk_remote'
    beta=0.5,             # interpolation weight for JSD
    temperature=1.0,
)

model.set_loss(loss_fn)
```

**Parameters:**
- `teacher_mode`: How teacher logits are obtained
  - `full`: Full vocabulary logits from a local teacher model
  - `topk_local`: Top-k logits from a local teacher with chunked computation for memory efficiency
  - `topk_remote`: Top-k logits from a remote API teacher
- `beta`: Interpolation weight between student and teacher distributions in JSD (0 = pure student, 1 = pure teacher)
- `temperature`: Softmax temperature for both student and teacher distributions

The GKD loss implements chunked computation internally to reduce peak memory usage when working with large vocabularies.

> GKD is useful for training smaller student models that mimic the behavior of larger teacher models, and supports both local and remote teacher setups.
