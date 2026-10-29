# Sequence Parallel (SP)

Sequence Parallel splits long sequences across multiple GPUs along the sequence dimension, enabling training with sequence lengths that exceed single-GPU memory. Twinkle implements Ulysses-style sequence parallel with optional derived ring attention.

## Overview

| Concept | Description |
|---------|-------------|
| **SequenceParallelConfig** | Configuration dataclass for SP |
| **SequenceParallelStrategy** | Strategy class that wraps SP lifecycle |
| **SequenceParallel** | Core implementation handling pad/split/gather |

## Configuration

```python
from twinkle.model.transformers.strategy.sequence_parallel import SequenceParallelConfig

config = SequenceParallelConfig(
    enabled=True,           # Enable sequence parallel
    ulysses_size=None,      # Ulysses SP degree (auto-derived from DeviceMesh if None)
)
```

## Usage with DeviceMesh

SP is activated by setting `ulysses_size` in `DeviceMesh.from_sizes()`:

```python
from twinkle.utils import DeviceMesh

# 8 GPUs: 4-way Ulysses SP × 2-way data parallel
device_mesh = DeviceMesh.from_sizes(
    world_size=8,
    dp_size=2,
    ulysses_size=4,
)
```

## How It Works

1. **Pad** — input sequences are padded to a length divisible by SP world size
2. **Split** — padded inputs are evenly split across SP ranks along the sequence dimension
3. **Distributed Attention** — SDPA and FlashAttention 2/3 are patched to perform Ulysses all-to-all communication before/after attention computation
4. **Gather** — collect the present logps, entropies, values and requested logits before the unchanged full-sequence loss

## Supported Attention Backends

| Backend | Status |
|---------|--------|
| FlashAttention 2 / 3 | Ulysses and packed/padding-free paths; requires a compatible installed kernel |
| SDPA | Supported (non-packed batches only) |
| Derived Ring Attention | CUDA FlashAttention 2/3 only (`rp_world_size > 1`); incompatible platforms fail during initialization |

## Qwen3.5 Linear Attention

SP automatically detects Qwen3.5 GatedDeltaNet linear attention layers and applies the `Qwen3_5GatedDeltaNetUlyssesPatch` for correct sequence-parallel behavior on hybrid attention architectures.

## MoE Auxiliary Loss

For MoE models, SP automatically installs a forward hook that gathers router logits across SP ranks before auxiliary loss computation, ensuring correct load-balancing signals.

## Key Constraints

- `num_key_value_heads` must be divisible by `ulysses_size` (for Ulysses) or use ring attention fallback
- Packed/padding-free batches require FlashAttention 2 or 3
- Derived ring attention requires `batch_size == 1` (packed format)
- `torch.distributed` must be initialized

## Isolating per-model SP contexts

Each SequenceParallelStrategy owns its own SequenceParallel context, which stores that model's communication group, ranks, and current batch state, so algorithms that use multiple models, such as DPO, can enable SP training at the same time.
