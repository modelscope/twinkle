# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shard snapshots and HF delta export based on verl."""
import torch

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def shard_delta_indices(
    local_new: torch.Tensor,
    local_snap: torch.Tensor,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Byte-diff a local shard against its snapshot; return (global_positions, values).

    Positions are int64 indices into the *full flattened parameter* (offset + local index).
    Dtype-agnostic, bytewise (view-as-int), no arithmetic -- matches ``bytewise_diff_mask``.
    """
    es = local_new.element_size()
    int_dtype = _DTYPE_INT.get(es)
    if int_dtype is None:
        raise ValueError(f"unsupported element size {es}")
    mask = local_new.view(int_dtype) != local_snap.view(int_dtype)
    local_idx = mask.nonzero(as_tuple=False).view(-1)
    values = local_new[local_idx]
    global_idx = local_idx.to(torch.int64) + offset
    return global_idx, values


def _hf_entry_identity(name, spec, place, lidx, lval):
    """Identity profile: the param IS its own single slot (weight name == HF name)
    -- translate the shard-local delta to within-param coordinates. int32
    positions: the wire is int32 anyway and the engine asserts the range."""
    from .spec import translate_flat_indices

    gidx = (translate_flat_indices(lidx, place) if lidx.numel() else lidx).to(torch.int32)
    counts = torch.zeros(1, dtype=torch.int64)
    counts[0] = int(gidx.numel())
    return [(name, tuple(spec.full_shape))], str(lval.dtype).replace("torch.", ""), counts, gidx, lval


def hf_delta_export(gen, snaps: dict, entry_fn):
    """STEADY export: wrap a raw ``(name, local_shard, spec)`` exporter into final
    HF-coordinate delta entries ``(slots, dtype_str, counts, hf_idx, hf_val,
    gather_group)`` -- diff against the pinned snapshot, refresh it, then hand the
    shard-local delta to ``entry_fn(name, spec, place, lidx, lval)``, the engine's
    per-param entry builder (FSDP: identity only; veomni adds the EP converter).
    The delta engine consumes these entries verbatim (batch -> gather -> wire); no
    spec, no placement and no conversion cross the boundary. Requires a prior seed
    pass."""
    from .spec import derive_dtensor_placement

    for name, local, spec in gen:
        local = local.detach().contiguous().view(-1)
        snap = snaps.get(name)
        assert snap is not None and snap.numel() == local.numel(), (
            f"{name}: no seed snapshot for this shard; run the seed export first"
        )
        if spec.place is not None:
            # explicit exporter override: the backend declared the whole triple
            # (hybrid geometries are not derivable from DTensor facts alone).
            place, contributes, pg = spec.place, spec.contributes, spec.gather_group
        else:
            place, contributes, pg = derive_dtensor_placement(spec)
        if contributes:
            base = snap.to(local.device, non_blocking=True)
            lidx, lval = shard_delta_indices(local, base, 0)
        else:
            # replicated param owned by another rank; empty delta keeps lockstep.
            lidx = torch.empty(0, dtype=torch.int64, device=local.device)
            lval = torch.empty(0, dtype=local.dtype, device=local.device)
        snap.copy_(local, non_blocking=True)
        yield (*entry_fn(name, spec, place, lidx, lval), pg)


def prime_delta_snapshots(gen, snaps: dict, pin: bool) -> None:
    """Snapshot each rank's current shards to CPU as the steady diff base. Run
    right after the seed's full-weight sync: weights do not move during the
    sync, so the snapshots equal exactly what the rollout side received.

    ``pin`` selects pinned vs pageable host memory and is the ENGINE's call
    (``BaseEngine.delta_pin_snapshots``): pinning a whole shard set
    (cudaHostAlloc) competes with everything else that pins on the node and its
    failure surfaces as a CUDA out-of-memory; pageable costs a slower H2D on
    the diff read-back but cannot OOM the device."""
    for name, local, _spec in gen:
        local = local.detach().contiguous().view(-1)
        snap = snaps.get(name)
        if snap is None or snap.numel() != local.numel():
            snap = torch.empty_like(local, device="cpu", pin_memory=pin)
            snaps[name] = snap
        snap.copy_(local, non_blocking=True)
