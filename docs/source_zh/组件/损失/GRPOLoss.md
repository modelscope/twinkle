# GRPO 损失

组相对策略优化（GRPO）及其变体实现了带有 PPO 风格裁剪和 KL 正则化的策略梯度损失。

## GRPOLoss

标准 GRPO 损失，带有重要性采样、PPO 裁剪和可选的 KL 惩罚。

```python
from twinkle.loss import GRPOLoss

loss_fn = GRPOLoss(
    clip_range=0.2,
    beta=0.01,        # KL 惩罚系数
)

model.set_loss(loss_fn)
```

**参数:**
- `clip_range`: 重要性权重的 PPO 裁剪范围（默认: 0.2）
- `beta`: KL 散度惩罚系数。设为 0 以禁用 KL 正则化

损失函数同时处理标准批次和打包序列（通过 `position_ids` 检测）。它计算每个 token 的重要性权重，应用 PPO 裁剪，并可选地添加针对参考策略的 KL 惩罚项。

## 变体

Twinkle 提供了多种 GRPO 变体:

### GSPOLoss

序列级重要性采样变体，在序列级别而非 token 级别计算重要性权重。

```python
from twinkle.loss import GSPOLoss
loss_fn = GSPOLoss(clip_range=0.2, beta=0.01)
```

### SAPOLoss

软门控优势策略优化，在优势值上应用 sigmoid 门控来控制优化方向。

```python
from twinkle.loss import SAPOLoss
loss_fn = SAPOLoss(clip_range=0.2, beta=0.01, tau=1.0)
```

### CISPOLoss

裁剪重要性采样策略优化，在与优势值相乘之前对重要性权重进行显式裁剪。

```python
from twinkle.loss import CISPOLoss
loss_fn = CISPOLoss(clip_range=0.2, beta=0.01)
```

### BNPOLoss

批归一化策略优化，在聚合之前对批次内的每 token 损失进行归一化。

```python
from twinkle.loss import BNPOLoss
loss_fn = BNPOLoss(clip_range=0.2, beta=0.01)
```

### DRGRPOLoss

动态比率 GRPO，使用固定分母进行重要性权重计算。

```python
from twinkle.loss import DRGRPOLoss
loss_fn = DRGRPOLoss(clip_range=0.2, beta=0.01)
```

> 所有 GRPO 变体共享相同的打包序列处理、对数概率对齐和 KL 惩罚计算基础流水线。它们的主要区别在于重要性权重和优势值的组合方式。
