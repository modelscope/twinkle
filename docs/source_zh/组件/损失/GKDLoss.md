# GKD 损失

广义知识蒸馏（GKD）损失使用 Jensen-Shannon 散度将知识从教师模型蒸馏到学生模型。

```python
from twinkle.loss import GKDLoss

loss_fn = GKDLoss(
    teacher_mode='full',  # 'full', 'topk_local', 'topk_remote'
    beta=0.5,             # JSD 的插值权重
    temperature=1.0,
)

model.set_loss(loss_fn)
```

**参数:**
- `teacher_mode`: 获取教师 logits 的方式
  - `full`: 来自本地教师模型的全词表 logits
  - `topk_local`: 来自本地教师的 top-k logits，使用分块计算以节省内存
  - `topk_remote`: 来自远程 API 教师的 top-k logits
- `beta`: 学生和教师分布在 JSD 中的插值权重（0 = 纯学生，1 = 纯教师）
- `temperature`: 学生和教师分布的 softmax 温度

GKD 损失内部实现了分块计算，以减少处理大词表时的峰值内存使用。

> GKD 适用于训练模仿大型教师模型行为的小型学生模型，同时支持本地和远程教师设置。
