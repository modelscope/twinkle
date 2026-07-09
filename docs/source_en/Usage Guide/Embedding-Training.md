# Embedding Training

Twinkle supports contrastive embedding model training with InfoNCE loss, in-batch negatives, and cross-rank gathering. This guide demonstrates how to train embedding models using Twinkle.

---

## Overview

Embedding training in Twinkle uses the following core components:

| Component | Role |
|:----------|:-----|
| `InfonceLoss` | Contrastive loss with in-batch negatives |
| `EmbeddingMetric` | Tracks pos/neg similarity and loss |
| `TransformersModel` | Trainable embedding model (with LoRA or full) |
| `InputProcessor` | Processes anchor/positive pairs into features |

### Data Format

Each training sample consists of **(anchor, positive)** pairs. In the embedding feature tensor:

```
embeddings: [anchor_0, positive_0, anchor_1, positive_1, ...]
labels:     [       1,         0,        1,          0, ...]
```

- `labels=1` marks the start of a new group (anchor)
- `labels=0` marks positives/negatives within the group

---

## Basic Embedding Training

A minimal embedding training script with DDP:

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

# --- Configuration ---
MODEL_ID = 'ms://Qwen/Qwen3.5-4B'
MODEL_GPUS = 4
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
TEMPERATURE = 0.07
EMB_MAX_LENGTH = 8192

# --- Initialize ---
device_groups = [
    DeviceGroup(name='model', ranks=list(range(MODEL_GPUS)), device_type='GPU'),
]
model_mesh = DeviceMesh.from_sizes(world_size=MODEL_GPUS, dp_size=MODEL_GPUS)
twinkle.initialize(mode='ray', nproc_per_node=MODEL_GPUS, groups=device_groups)

# --- Model ---
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

# --- Template ---
template = Qwen3_5Template(
    model_id=MODEL_ID,
    max_length=EMB_MAX_LENGTH,
    enable_thinking=False,
)

# --- Training Loop ---
for step, batch in enumerate(dataloader):
    # batch: list of features with anchor/positive pairs
    model.forward_backward(inputs=batch, task='embedding')
    model.clip_grad_and_step(gradient_accumulation_steps=1)

    if step % 10 == 0:
        metric = model.calculate_metric(is_training=True)
        logger.info(f'Step {step}: {metric}')
```

### Key Parameters

| Parameter | Recommended | Description |
|:----------|:------------|:------------|
| `temperature` | 0.05–0.1 | Lower = sharper contrast. 0.07 keeps gradients flowing until cosine > 0.75 |
| `use_batch` | True | Enables cross-sample in-batch negatives for better efficiency |
| `hard_negatives` | None or 7 | Fix negative count per sample; None uses all in-batch |
| `find_unused_parameters` | True | Required for embedding models (only last hidden state contributes gradients) |

---

## Monitoring

The `EmbeddingMetric` reports key training signals:

| Metric | What it means |
|:-------|:--------------|
| `pos_sim` | Average anchor-positive cosine similarity (target: > 0.8) |
| `neg_sim` | Average anchor-negative similarity (target: < 0.3) |
| `loss` | InfoNCE loss value |
| `grad_norm` | Gradient magnitude |

Healthy training shows `pos_sim` rising and `neg_sim` stable or falling. If `pos_sim` saturates near 1.0, lower the temperature.
