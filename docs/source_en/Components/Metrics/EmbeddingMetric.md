# EmbeddingMetric

The `EmbeddingMetric` tracks embedding quality during contrastive (InfoNCE) training. It reports anchor-positive cosine similarity statistics and in-batch negative similarity.

## Usage

```python
from twinkle.metric import EmbeddingMetric

metric = EmbeddingMetric(device_mesh=device_mesh, process_group=process_group)

# During training
metric.accumulate(inputs, outputs)

# At log interval
results = metric.calculate()
# results: {
#   'pos_sim': '0.8523',     # Mean anchor-positive cosine similarity
#   'pos_sim_min': '0.7102', # Min across batch
#   'pos_sim_max': '0.9451', # Max across batch
#   'neg_sim': '0.2134',     # Mean anchor-negative similarity
#   'loss': '0.3412',        # Average InfoNCE loss
#   'grad_norm': '1.234567', # Gradient norm
# }
```

## Reported Metrics

| Metric | Description |
|:-------|:------------|
| `pos_sim` | Mean cosine similarity between anchors and their positives |
| `pos_sim_min` | Minimum anchor-positive similarity in the batch |
| `pos_sim_max` | Maximum anchor-positive similarity in the batch |
| `neg_sim` | Mean similarity between anchors and other positives (in-batch negatives) |
| `loss` | Average contrastive loss value |
| `grad_norm` | Gradient norm (passed via kwargs) |

## Cross-Rank Gathering

`EmbeddingMetric` performs an `all_gather` to compute similarity statistics across all DP ranks, providing a global view of embedding quality even under data-parallel training.

> This metric pairs with `InfonceLoss` for embedding/retrieval training tasks.
