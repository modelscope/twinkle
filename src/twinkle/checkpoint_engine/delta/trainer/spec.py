# Copyright (c) ModelScope Contributors. All rights reserved.
"""Shard placement metadata"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup
    from torch.distributed.device_mesh import DeviceMesh

__all__ = ["BlockPlacement", "ShardSpec", "derive_dtensor_placement", "translate_flat_indices"]


@dataclass
class ShardSpec:
    """Declarative placement descriptor for one exported local parameter shard."""

    # Logical (full) tensor shape; the distribution facts below refer to it.
    full_shape: tuple
    # Distribution: torch DeviceMesh + per-mesh-dim Placement. None = unsharded.
    mesh: Optional[DeviceMesh] = None
    placements: Optional[tuple] = None
    # Explicit placement override for trainers whose sharding is not fully captured
    # by DTensor placements (e.g. veomni's manual expert-dim split): ``place`` is an
    # int flat offset or a BlockPlacement in a *virtual* full tensor, and
    # ``gather_group`` is the ProcessGroup covering every rank that holds a block
    # (pass a real group object -- the engine treats ``None`` as "unsharded").
    # ``contributes=False`` marks a rank whose block is a replica owned by a
    # peer (e.g. HSDP replicate dims): it keeps lockstep with an empty delta.
    place: Optional[int | BlockPlacement] = None
    gather_group: Optional[ProcessGroup] = None
    contributes: bool = True
    @classmethod
    def from_param(cls, param: torch.Tensor) -> ShardSpec:
        if isinstance(param, DTensor):
            return cls(full_shape=tuple(param.shape), mesh=param.device_mesh, placements=tuple(param.placements))
        return cls(full_shape=tuple(param.shape))


def _prod(xs: tuple | list) -> int:
    n = 1
    for x in xs:
        n *= int(x)
    return n


def _row_major_strides(shape: tuple | list) -> tuple:
    strides, acc = [], 1
    for d in reversed([int(x) for x in shape]):
        strides.append(acc)
        acc *= d
    return tuple(reversed(strides))


@dataclass(frozen=True)
class BlockPlacement:
    """This rank's local shard is one hyper-rectangular block of the full tensor:
    ``full[o0:o0+l0, o1:o1+l1, ...]`` with ``local_shape=(l0, l1, ...)`` and
    ``global_offset=(o0, o1, ...)``. Produced by :func:`derive_dtensor_placement` for every
    sharded geometry, including the dim-0 cut (FSDP2 ``Shard(0)``): that block is
    flat-contiguous, and :func:`translate_flat_indices` detects it via
    ``is_flat_contiguous`` and keeps the single-add fast path."""

    local_shape: tuple
    global_offset: tuple
    full_shape: tuple

    @property
    def local_strides(self) -> tuple:
        return _row_major_strides(self.local_shape)

    @property
    def full_strides(self) -> tuple:
        return _row_major_strides(self.full_shape)

    @property
    def is_flat_contiguous(self) -> bool:
        """True when the block is one contiguous flat range (only dim 0 is cut).

        Then every trailing dim is whole, so the trailing offsets are all zero
        and the block occupies flat positions ``[flat_offset, flat_offset+numel)``.
        """
        return all(int(lo) == int(fu) for lo, fu in zip(self.local_shape[1:], self.full_shape[1:], strict=False))

    @property
    def flat_offset(self) -> int:
        """Flat start of the block; only meaningful when ``is_flat_contiguous``."""
        return int(self.global_offset[0]) * _prod(self.full_shape[1:]) if self.full_shape else 0


def _shard_dim(p) -> Optional[int]:
    """Which tensor dim this placement cuts, or None. ``is_shard()`` misses ``_StridedShard`` on torch 2.13."""
    if p.is_shard():
        return int(p.dim)
    return int(p.dim) if type(p).__name__ == "_StridedShard" else None


def _assert_even_strided(spec: ShardSpec, placements: tuple) -> None:
    """Strided offsets assume equal shards, and switch every tensor dim to that rule -- so check them all."""
    if not any(type(p).__name__ == "_StridedShard" for p in placements):
        return
    cuts: dict[int, int] = {}
    for mesh_dim, p in enumerate(placements):
        tdim = _shard_dim(p)
        if tdim is not None:
            cuts[tdim] = cuts.get(tdim, 1) * spec.mesh.size(mesh_dim)
    for tdim, total in cuts.items():
        if spec.full_shape[tdim] % total:
            raise NotImplementedError(
                f"sharded delta does not support uneven strided sharding: tensor dim {tdim} "
                f"has size {spec.full_shape[tdim]}, which {total} shards do not divide evenly "
                f"(placements={placements})"
            )


def translate_flat_indices(lidx: torch.Tensor, place: int | BlockPlacement) -> torch.Tensor:
    """Map shard-local flat positions to full-tensor flat positions.

    ``place`` is what the caller's dispatch produced (``spec.place`` or
    :func:`derive_dtensor_placement`): an ``int`` for the
    identity cases (unsharded / replicated / explicit exporter overrides, translate
    = add), or a :class:`BlockPlacement`. A flat-contiguous block (only dim 0 cut,
    e.g. FSDP2 ``Shard(0)``) keeps the single-add fast path; any other block does a
    mixed-radix decompose by the local shape, adds the per-dim offset, and recomposes
    with the full-tensor strides -- a few divmods on the nnz tensor, no collectives.
    """
    if isinstance(place, int):
        return lidx + place if place else lidx
    if place.is_flat_contiguous:
        off = place.flat_offset
        return lidx + off if off else lidx
    out = torch.zeros_like(lidx)
    rem = lidx
    for lstride, off, fstride in zip(place.local_strides, place.global_offset, place.full_strides, strict=False):
        coord = torch.div(rem, lstride, rounding_mode="floor")
        rem = rem - coord * lstride
        out = out + (coord + int(off)) * fstride
    return out


def derive_dtensor_placement(spec: ShardSpec) -> tuple[int | BlockPlacement, bool, Optional[ProcessGroup]]:
    """Derive ``(place, contributes, gather_group)`` for THIS rank from a spec
    whose distribution is fully declared by ``mesh`` + ``placements`` (or is
    unsharded). Specs carrying an explicit exporter override (``spec.place``)
    never reach this function -- the caller dispatches on ``spec.place`` and
    reads the exporter's own ``(place, contributes, gather_group)`` triple
    verbatim, because a hybrid geometry (e.g. veomni's manual ep split) is not
    derivable from the DTensor facts alone.

    ``place`` feeds :func:`translate_flat_indices`:

    * unsharded (``mesh is None``) or fully replicated: ``0``; no group (the local
      tensor is already the full parameter).
    * any sharded geometry: a :class:`BlockPlacement` computed from
      ``compute_local_shape_and_global_offset`` (pure math, no collective). For a
      single Shard dim, only ranks at coordinate 0 of every Replicate dim
      contribute and the gather group is the Shard dim's subgroup; the FSDP2
      default ``Shard(0)`` yields a flat-contiguous block, which keeps the add
      fast path in :func:`translate_flat_indices`. With several Shard dims (e.g.
      automodel's EP x FSDP ``(Shard(0), Shard(1))`` expert mesh) every rank holds
      a distinct block and the gather group spans every Shard dim at once; a
      Replicate dim beside them is held fixed rather than spanned.

    ``_StridedShard`` is a block too: it permutes which block a rank gets, and
    ``compute_local_shape_and_global_offset`` returns the permuted offset.
    """
    import torch.distributed as dist

    assert spec.place is None, "explicit-place specs are dispatched by the caller, not derived"

    if spec.mesh is None:
        return 0, (dist.get_rank() == 0 if dist.is_initialized() else True), None

    placements = spec.placements
    shard_dims = [d for d, p in enumerate(placements) if _shard_dim(p) is not None]
    _assert_even_strided(spec, placements)

    coord = spec.mesh.get_coordinate()
    contributes = True
    if coord is not None:
        for d, p in enumerate(placements):
            if p.is_replicate() and coord[d] != 0:
                contributes = False
                break

    if not shard_dims:
        # replicated across every mesh dim: full tensor on each rank, no gather
        return 0, contributes, None

    local_shape, global_offset = compute_local_shape_and_global_offset(spec.full_shape, spec.mesh, list(placements))
    place = BlockPlacement(tuple(local_shape), tuple(global_offset), tuple(spec.full_shape))

    if len(shard_dims) == 1:
        return place, contributes, spec.mesh.get_group(mesh_dim=shard_dims[0])
    return place, contributes, _shard_dims_group(spec.mesh, shard_dims)


_GATHER_GROUPS: dict[tuple, ProcessGroup] = {}


def _shard_dims_group(mesh: DeviceMesh, shard_dims: list[int]) -> ProcessGroup:
    """The ranks reached by varying the Shard mesh dims, with this rank's coordinate fixed on the others."""
    import itertools

    import torch.distributed as dist

    coord = mesh.get_coordinate()
    assert coord is not None, "cannot derive a gather group for a rank outside the mesh it exports from"
    other = [d for d in range(mesh.ndim) if d not in shard_dims]
    mine = None
    for combo in itertools.product(*[range(mesh.size(d)) for d in other]):
        index: list = [slice(None)] * mesh.ndim
        for d, c in zip(other, combo, strict=True):
            index[d] = c
        ranks = tuple(sorted(int(r) for r in mesh.mesh[tuple(index)].flatten().tolist()))
        got = _GATHER_GROUPS.get(ranks)
        if got is None:
            got = dist.new_group(list(ranks))
            _GATHER_GROUPS[ranks] = got
        if all(coord[d] == c for d, c in zip(other, combo, strict=True)):
            mine = got
    assert mine is not None, f"no gather group covers coordinate {coord} of mesh {tuple(mesh.mesh.shape)}"
    return mine
