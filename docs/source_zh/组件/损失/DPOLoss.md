# DPO 损失

直接偏好优化（DPO）及其变体用于在不需要单独奖励模型的情况下将模型与人类偏好对齐。

## DPOLoss

标准 DPO 损失，支持多种损失类型和可选的无参考模式。

```python
from twinkle.loss import DPOLoss

loss_fn = DPOLoss(
    loss_type='sigmoid',  # 'sigmoid', 'hinge', 'ipo', 'kto'
    beta=0.1,
    sft_weight=0.0,       # 可选的 SFT 正则化权重
    reference_free=False,
)

model.set_loss(loss_fn)
```

**参数:**
- `loss_type`: DPO 变体 — `sigmoid`（默认）, `hinge`, `ipo`, 或 `kto`
- `beta`: 控制偏好强度的温度参数
- `sft_weight`: chosen 响应上额外 SFT 损失的权重
- `reference_free`: 为 `True` 时跳过参考模型的对数概率

损失函数期望批次中 chosen/rejected 样本交替排列。它计算序列级对数概率，优化策略使其偏好 chosen 而非 rejected 响应。

## SimPOLoss

简化偏好优化，通过使用长度归一化的对数概率来消除对参考模型的需求。

```python
from twinkle.loss import SimPOLoss

loss_fn = SimPOLoss(beta=2.0, gamma=1.0)
```

**参数:**
- `beta`: logit 差异的缩放因子
- `gamma`: 添加到偏好差距的 margin 项

## CPOLoss

对比偏好优化，将偏好学习与行为克隆相结合。

```python
from twinkle.loss import CPOLoss

loss_fn = CPOLoss(beta=0.1, cpo_alpha=1.0)
```

**参数:**
- `beta`: 偏好损失的温度
- `cpo_alpha`: chosen 响应上行为克隆（NLL）损失的权重

## ORPOLoss

赔率比偏好优化，在单一损失中统一 SFT 和偏好对齐。

```python
from twinkle.loss import ORPOLoss

loss_fn = ORPOLoss(beta=0.1)
```

该损失将 chosen 响应上的标准 NLL 项与对数赔率比惩罚相结合，推动模型远离 rejected 响应。

> 所有偏好损失都继承自 `PreferenceLossBase` 的共享工具方法，包括对数概率计算、chosen/rejected 拆分和序列级聚合。
