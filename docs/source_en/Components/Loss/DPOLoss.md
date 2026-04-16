# DPO Loss

Direct Preference Optimization (DPO) and its variants are used for aligning models with human preferences without requiring a separate reward model.

## DPOLoss

The standard DPO loss supports multiple loss types and optional reference-free mode.

```python
from twinkle.loss import DPOLoss

loss_fn = DPOLoss(
    loss_type='sigmoid',  # 'sigmoid', 'hinge', 'ipo', 'kto'
    beta=0.1,
    sft_weight=0.0,       # optional SFT regularization weight
    reference_free=False,
)

model.set_loss(loss_fn)
```

**Parameters:**
- `loss_type`: DPO variant — `sigmoid` (default), `hinge`, `ipo`, or `kto`
- `beta`: Temperature parameter controlling preference strength
- `sft_weight`: Weight for an additional SFT loss term on chosen responses
- `reference_free`: If `True`, skips reference model log-probabilities

The loss expects interleaved chosen/rejected pairs in the batch. It computes sequence-level log-probabilities and optimizes the policy to prefer chosen over rejected responses.

## SimPOLoss

Simplified Preference Optimization that removes the need for a reference model by using length-normalized log-probabilities.

```python
from twinkle.loss import SimPOLoss

loss_fn = SimPOLoss(beta=2.0, gamma=1.0)
```

**Parameters:**
- `beta`: Scaling factor for the logit difference
- `gamma`: Margin term added to preference gap

## CPOLoss

Contrastive Preference Optimization combines preference learning with behavior cloning.

```python
from twinkle.loss import CPOLoss

loss_fn = CPOLoss(beta=0.1, cpo_alpha=1.0)
```

**Parameters:**
- `beta`: Temperature for the preference loss
- `cpo_alpha`: Weight of the behavior cloning (NLL) loss on chosen responses

## ORPOLoss

Odds Ratio Preference Optimization unifies SFT and preference alignment in a single loss.

```python
from twinkle.loss import ORPOLoss

loss_fn = ORPOLoss(beta=0.1)
```

The loss combines a standard NLL term on chosen responses with a log-odds-ratio penalty that pushes the model away from rejected responses.

> All preference losses inherit shared utilities from `PreferenceLossBase`, including log-probability computation, chosen/rejected splitting, and sequence-level aggregation.
