"""Last-token pooling for full sequences and sequence-parallel shards."""
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_reduce


def _packed_last_indices(position_ids: torch.Tensor, total_len: int) -> torch.Tensor:
    """For padding-free batches: per-segment last-token indices into a [1, total] sequence."""
    flat = position_ids.squeeze(0) if position_ids.dim() == 2 else position_ids
    starts = (flat == 0).nonzero(as_tuple=False).squeeze(-1)
    end_anchor = torch.tensor([total_len], device=flat.device, dtype=starts.dtype)
    boundaries = torch.cat([starts, end_anchor])
    return (boundaries[1:] - 1).long()


def pool_last_valid_token(features: torch.Tensor, inputs: dict[str, Any],
                          sp_strategy=None) -> tuple[torch.Tensor, bool]:
    """Gather the last valid token per sequence from ``[B, T, C]`` -> ``[n_seqs, C]``.

    Build a one-hot end-token mask in the un-padded global frame, route it
    through the same pad+split as ``input_ids`` so it aligns with local
    features, pool locally, then ``all_reduce`` only the ``[n_seqs, C]``
    tensor across SP × RP. No feature gather; uniform across
    DP / Ulysses / zigzag-ring / padding-free. Returns ``(pooled, is_packed)``.
    """
    sp_enabled = (
        sp_strategy is not None and getattr(sp_strategy, 'enabled', False)
        and getattr(sp_strategy, 'world_size', 1) > 1)

    if sp_enabled:
        ref_pos = sp_strategy.real_position_ids
    elif inputs.get('position_ids') is not None:
        ref_pos = inputs['position_ids']
    else:
        # Standard padded batch: HF omits position_ids (it derives them internally).
        # Rebuild a position_ids-like reference (valid token -> its index, padding -> -1)
        # so the last-valid-token pooling below, which keys off ``ref_pos >= 0`` for
        # validity and a single ``== 0`` per row for packing detection, still works.
        T = features.shape[1]
        arange = torch.arange(T, device=features.device)
        am = inputs.get('attention_mask')
        if am is None:
            ref_pos = arange.expand(features.shape[0], T)
        else:
            if am.dim() >= 3:
                am = am.reshape(am.shape[0], -1)[:, :T]
            ref_pos = torch.where(am.bool(), arange, arange.new_tensor(-1))
    if ref_pos.dim() == 3:
        ref_pos = ref_pos[0]
    # Accept both flash-attn (cu_seq_lens_q) and Megatron (cu_seqlens_q) conventions.
    cu_seq_lens_q = inputs.get('cu_seq_lens_q', inputs.get('cu_seqlens_q'))

    is_packed = (
        features.shape[0] == 1 and (cu_seq_lens_q is not None or int((ref_pos.reshape(-1) == 0).sum()) > 1))

    device, dtype = features.device, features.dtype
    T_real = ref_pos.shape[-1]

    if is_packed:
        if torch.is_tensor(cu_seq_lens_q) and cu_seq_lens_q.numel() >= 2:
            end_idx = (cu_seq_lens_q[1:].long() - 1).to(device)
        else:
            end_idx = _packed_last_indices(ref_pos, T_real).to(device)
        n_seqs = end_idx.shape[0]
        batch_idx = torch.zeros_like(end_idx)
        target_idx = torch.arange(n_seqs, device=device)
        B_mask, n_targets = 1, n_seqs
    else:
        n_seqs = ref_pos.shape[0]
        # clamp_min(0) prevents negative indexing when a row is fully padded (sum == 0).
        positions = torch.arange(T_real, device=ref_pos.device).expand_as(ref_pos)
        end_idx = positions.masked_fill(ref_pos < 0, -1).amax(-1).clamp_min(0)
        batch_idx = torch.arange(n_seqs, device=device)
        target_idx = torch.zeros(n_seqs, dtype=torch.long, device=device)
        B_mask, n_targets = n_seqs, 1

    mask = torch.zeros(B_mask, T_real, n_targets, dtype=dtype, device=device)
    mask[batch_idx, end_idx, target_idx] = 1.0

    if sp_enabled:
        # Route mask through the same pad+split as input_ids to align with local features.
        rp = sp_strategy.real_position_ids
        rp_padded = sp_strategy.pad(rp, padding_value=-1, position_ids=rp, dim=-1)
        mask = sp_strategy.pad(mask, padding_value=0, position_ids=rp, dim=1)
        mask = sp_strategy.split(mask, dim=1, position_ids=rp_padded)

    assert mask.shape[1] == features.shape[1], (
        f'mask seq-dim {mask.shape[1]} != features seq-dim {features.shape[1]}; '
        'sp_strategy.split contract broken or T_real mismatched.')

    pooled = torch.einsum('bth,btn->bnh', features, mask)
    pooled = pooled.squeeze(0) if is_packed else pooled.squeeze(1)

    if sp_enabled and dist.is_available() and dist.is_initialized():
        for grp_attr, size_attr in (('_sp_group', 'sp_world_size'), ('_rp_group', 'rp_world_size')):
            grp = getattr(sp_strategy, grp_attr, None)
            if grp is not None and getattr(sp_strategy, size_attr, 1) > 1:
                pooled = all_reduce(pooled, op=dist.ReduceOp.SUM, group=grp)

    return pooled, is_packed
