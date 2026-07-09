# Padding-Free Training

Padding-free (also called "packing") training eliminates wasted computation on padding tokens by concatenating multiple sequences into a single packed batch. Twinkle supports padding-free training for both standard attention and Qwen3.5's GatedDeltaNet linear attention.

## How It Works

Instead of padding all sequences to `max_length`, padding-free packs multiple sequences into one row and uses `position_ids` to track sequence boundaries. This avoids wasted FLOPs on padding tokens.

```
Standard:   [tok tok tok PAD PAD PAD]  [tok tok PAD PAD PAD PAD]
Packed:     [tok tok tok tok tok ...]   ← no padding waste
```

## Usage

Padding-free is enabled via `PackingDataset` or `IterablePackingDataset`:

```python
from twinkle.dataset import PackingDataset

dataset = PackingDataset(
    dataset=base_dataset,
    max_length=8192,
)
```

The dataset automatically packs sequences and generates correct `position_ids` with resets at sequence boundaries.

## GatedDeltaNet Patch (Qwen3.5)

Qwen3.5 uses a hybrid architecture mixing standard attention with GatedDeltaNet linear attention. The native GatedDeltaNet implementation does not reset its linear-attention state at packed sequence boundaries.

`GatedDeltaNetPaddingFreePatch` fixes this by:

1. Patching `Qwen3_5DecoderLayer.forward` to pass `cu_seq_lens_q` (cumulative sequence lengths) to linear attention layers
2. Patching `Qwen3_5GatedDeltaNet.forward` to use flash-linear-attention kernels (`causal_conv1d`, `chunk_gated_delta_rule`) with `cu_seqlens` support

The patch is applied automatically when padding-free is detected on Qwen3.5 models.

### Requirements

- `flash-linear-attention` package must be installed
- Only needed for Qwen3.5 models with GatedDeltaNet layers
- When sequence parallel is enabled, a separate `Qwen3_5GatedDeltaNetUlyssesPatch` is used instead

## Attention Backend Requirements

| Attention Backend | Padding-Free Support |
|-------------------|---------------------|
| FlashAttention2 | Fully supported |
| SDPA | Supported (incompatible with sequence parallel) |
| Eager | Not supported |
