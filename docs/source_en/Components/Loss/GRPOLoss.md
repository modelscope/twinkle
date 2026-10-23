# GRPO Loss

Group Relative Policy Optimization (GRPO) and its variants implement policy gradient losses with PPO-style clipping and KL regularization.

## GRPOLoss

The standard GRPO loss with importance sampling, PPO clipping, and optional KL penalty.

```python
from twinkle.loss import GRPOLoss

loss_fn = GRPOLoss(
    epsilon=0.2,
    beta=0.01,        # KL penalty coefficient
)

model.set_loss(loss_fn)
```

**Parameters:**
- `epsilon`: PPO clipping range for importance weights, lower side `1 - epsilon` (default: 0.2)
- `epsilon_high`: upper side `1 + epsilon_high` (default: same as `epsilon`); `CISPOLoss` truncates its IS weight at `1 + epsilon_high`
- `beta`: KL divergence penalty coefficient. Set to 0 to disable KL regularization

The loss handles both standard batches and packed sequences (detected via `position_ids`). It computes per-token importance weights, applies PPO clipping, and optionally adds a KL penalty term against the reference policy.

## Variants

Twinkle provides several GRPO variants:

### GSPOLoss

Sequence-level importance sampling variant that computes importance weights at the sequence level rather than token level.

```python
from twinkle.loss import GSPOLoss
loss_fn = GSPOLoss(epsilon=0.2, beta=0.01)
```

### SAPOLoss

Soft-gated Advantage Policy Optimization applies a sigmoid gate on the advantage to control the optimization direction.

```python
from twinkle.loss import SAPOLoss
loss_fn = SAPOLoss(beta=0.01, tau_pos=1.0, tau_neg=1.0)
```

### CISPOLoss

Clipped Importance Sampling Policy Optimization applies explicit clipping to importance weights before multiplying with advantages.

```python
from twinkle.loss import CISPOLoss
loss_fn = CISPOLoss(epsilon=0.2, epsilon_high=4.0, beta=0.01)
```

### BNPOLoss

Batch-Normalized Policy Optimization normalizes per-token loss across the batch before aggregation.

```python
from twinkle.loss import BNPOLoss
loss_fn = BNPOLoss(epsilon=0.2, beta=0.01)
```

### DRGRPOLoss

Dynamic Ratio GRPO with fixed normalization that uses a fixed denominator for importance weight computation.

```python
from twinkle.loss import DRGRPOLoss
loss_fn = DRGRPOLoss(epsilon=0.2, beta=0.01)
```

> All GRPO variants share the same base pipeline for packed-sequence handling, log-probability alignment, and KL penalty computation. They differ primarily in how importance weights and advantages are combined.
