# GRPOMetric

The `GRPOMetric` tracks policy optimization diagnostics during GRPO training, including KL divergence, clipping rates, entropy, and log-probability statistics.

## Usage

```python
from twinkle.metric import GRPOMetric

metric = GRPOMetric(
    device_mesh=device_mesh,
    process_group=process_group,
    epsilon=0.2,          # PPO clip range
    temperature=1.0,      # Sampling temperature for logp rescaling
    top_k_kl=10,          # Track top-K high-KL tokens per step
)

# During training loop
metric.accumulate(inputs, outputs, old_logps=old_logps, advantages=advantages)

# At log interval
results = metric.calculate()
# results: {
#   'train/policy_confidence': 0.85,
#   'train/mean_new_logp': -1.23,
#   'train/mean_old_logp': -1.30,
#   'train/logp_diff_mean': 0.07,
#   'train/approx_kl': 0.003,
#   'train/token_kl_max': 0.15,
#   'train/entropy': 2.1,
#   'train/clip_ratio': 0.02,
#   'train/clip_ratio_low': 0.01,
#   'train/clip_ratio_high': 0.01,
# }
```

## Reported Metrics

| Metric | Description |
|:-------|:------------|
| `train/policy_confidence` | exp(mean_new_logp) — higher means model is more confident |
| `train/mean_new_logp` | Average log-probability of generated tokens under current policy |
| `train/mean_old_logp` | Average log-probability under reference policy |
| `train/logp_diff_mean` | Mean (new - old) log-probability difference |
| `train/approx_kl` | Schulman K3 estimator of KL(old \|\| new) |
| `train/token_kl_max` | Maximum per-token KL across all ranks |
| `train/token_ratio_max` | Maximum importance weight across all ranks |
| `train/entropy` | Average token-level entropy |
| `train/clip_ratio` | Fraction of tokens clipped (low + high) |
| `train/clip_ratio_low` | Fraction clipped below (ratio < 1-ε, negative advantage) |
| `train/clip_ratio_high` | Fraction clipped above (ratio > 1+ε, positive advantage) |

## Variants

- **`GSPOMetric`** — Computes clip rate at sequence level (geometric-mean ratio per sequence)
- **`CISPOMetric`** — Unconditional clip rate (not gated by advantage sign)

## Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `epsilon` | float | 0.2 | Lower clip boundary |
| `epsilon_high` | float | None | Upper clip boundary (defaults to epsilon) |
| `temperature` | float | 1.0 | Rescale logps to T=1 before computing KL |
| `top_k_kl` | int | 0 | If > 0, record top-K high-KL token details |
| `ignore_index` | int | -100 | Label value to mask out |
