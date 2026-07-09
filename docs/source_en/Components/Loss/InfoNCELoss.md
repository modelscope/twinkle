# InfoNCE Loss

The `InfonceLoss` implements contrastive learning with in-batch negatives and optional cross-rank gathering. It is designed for embedding/retrieval model training.

## Usage

```python
from twinkle.loss import InfonceLoss

loss_fn = InfonceLoss(
    temperature=0.1,
    use_batch=True,           # Enable in-batch negatives
    hard_negatives=7,         # Fix negative count per sample
    mask_fake_negative=True,  # Mask false negatives
    fake_neg_margin=0.1,      # Margin for false negative detection
)

model.set_loss(loss_fn)
```

## Input Format

Each sample is laid out as `anchor(1) + positive(1) + negatives(n)` in a flat embedding tensor. The `inputs['labels']` is a 1-D mask where `1` marks the start of each group.

```
embeddings: [a0, p0, n0_1, n0_2, a1, p1, n1_1, n1_2, ...]
labels:     [ 1,  0,    0,    0,  1,  0,    0,    0, ...]
```

## Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `temperature` | float | 0.1 | Logit scaling factor |
| `use_batch` | bool | True | Use cross-sample in-batch negatives |
| `hard_negatives` | int | None | Fix per-sample negative count (truncate/upsample) |
| `mask_fake_negative` | bool | False | Mask logits > positive + margin |
| `fake_neg_margin` | float | 0.1 | Threshold for false negative masking |
| `include_qq` | bool | False | Add query-query similarity block |
| `include_dd` | bool | False | Add doc-doc similarity block |

## Cross-Rank Gathering

When `use_batch=True` and distributed training is active, embeddings are gathered from all DP ranks to maximize in-batch negative diversity. Only the local shard retains gradients.

## Similarity Blocks

The loss supports three similarity blocks for comprehensive contrastive learning:

- **Q→D (default)**: Query to all documents — primary contrastive signal
- **Q→Q** (`include_qq=True`): Query to all other queries — prevents query collapse
- **D→D** (`include_dd=True`): Document to all other documents — Qwen3-Embedding style

## Example: Embedding Training

```python
from twinkle.loss import InfonceLoss
from twinkle.metric import EmbeddingMetric

# Configure model for embedding
model.set_loss(InfonceLoss(temperature=0.05, use_batch=True, include_qq=True))
model.set_metric(EmbeddingMetric(device_mesh=mesh, process_group=pg))

# Training loop
for batch in dataloader:
    model.forward_backward(batch)
    model.clip_grad_and_step()
```
