# Embedding 模型训练

Twinkle 支持基于 InfoNCE 损失的对比学习 Embedding 模型训练，内置 in-batch negatives 和跨 rank 聚合。本文介绍如何使用 Twinkle 训练 Embedding 模型。

---

## 概述

Embedding 训练使用以下核心组件：

| 组件 | 职责 |
|:-----|:-----|
| `InfonceLoss` | 对比损失，支持 in-batch negatives |
| `EmbeddingMetric` | 追踪正/负对相似度和损失 |
| `TransformersModel` | 可训练的 Embedding 模型（LoRA 或全参） |
| `InputProcessor` | 将 anchor/positive 对处理为特征 |

### 数据格式

每个训练样本由 **(anchor, positive)** 对组成。在 Embedding 特征张量中：

```
embeddings: [anchor_0, positive_0, anchor_1, positive_1, ...]
labels:     [       1,         0,        1,          0, ...]
```

- `labels=1` 标记新分组的起始位置（anchor）
- `labels=0` 标记组内的 positive/negative

---

## 基础 Embedding 训练

使用 DDP 的最小化 Embedding 训练脚本：

```python
import twinkle
from twinkle import DeviceGroup, DeviceMesh, get_logger
from twinkle.dataloader import DataLoader
from twinkle.loss import InfonceLoss
from twinkle.metric import EmbeddingMetric
from twinkle.model import TransformersModel
from twinkle.processor import InputProcessor
from twinkle.template import Qwen3_5Template

logger = get_logger()

# --- 配置 ---
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
MODEL_GPUS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
TEMPERATURE = 0.07
EMB_MAX_LENGTH = 8192

# --- 初始化 ---
device_groups = [
    DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
]
model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS, groups=device_groups)

# --- 模型 ---
model = TransformersModel(
    model_id=MODEL_ID,
    device_mesh=model_mesh,
    remote_group='model',
    ddp_config={'find_unused_parameters': True},
)
model.set_processor(InputProcessor)
model.set_loss(InfonceLoss, temperature=TEMPERATURE, use_batch=True)
model.set_optimizer(optimizer_cls='AdamW', lr=LEARNING_RATE)
model.set_lr_scheduler(
    scheduler_cls='CosineWarmupScheduler',
    num_warmup_steps=200,
    num_training_steps=total_steps,
)
model.add_metric(EmbeddingMetric, is_training=True)

# --- 模板 ---
template = Qwen3_5Template(
    model_id=MODEL_ID,
    max_length=EMB_MAX_LENGTH,
    enable_thinking=False,
)

# --- 训练循环 ---
for step, batch in enumerate(dataloader):
    # batch: 包含 anchor/positive 对的特征列表
    model.forward_backward(inputs=batch, task='embedding')
    model.clip_grad_and_step(gradient_accumulation_steps=1)

    if step % 10 == 0:
        metric = model.calculate_metric(is_training=True)
        logger.info(f'Step {step}: {metric}')
```

### 关键参数

| 参数 | 推荐值 | 说明 |
|:----|:------|:-----|
| `temperature` | 0.05–0.1 | 越低对比越尖锐；0.07 保持梯度流动直至 cosine > 0.75 |
| `use_batch` | True | 启用跨样本 in-batch negatives 提升效率 |
| `hard_negatives` | None 或 7 | 固定每样本负例数量；None 使用全部 in-batch |
| `find_unused_parameters` | True | Embedding 模型必需（仅最后隐藏状态产生梯度） |

---

## 监控指标

`EmbeddingMetric` 报告关键训练信号：

| 指标 | 含义 |
|:----|:-----|
| `pos_sim` | anchor-positive 平均余弦相似度（目标 > 0.8） |
| `neg_sim` | anchor-negative 平均相似度（目标 < 0.3） |
| `loss` | InfoNCE 损失值 |
| `grad_norm` | 梯度范数 |

健康的训练表现为 `pos_sim` 持续上升、`neg_sim` 稳定或下降。如果 `pos_sim` 过早饱和至 1.0 附近，应降低 temperature。
