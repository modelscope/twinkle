# Copyright (c) ModelScope Contributors. All rights reserved.
"""In-process sparse delta apply for SGLang, loaded via ``custom_weight_loader``.

SGLang's ``update_weights_from_tensor`` supports pluggable loaders: when the request's
``load_format`` names an import path registered in the server args' ``custom_weight_loader``
list, SGLang ``dynamic_import``s it inside every TP worker process and calls
``loader(model, named_tensors)``. That is exactly the hook a sparse delta needs: the delta
payload is decoded and applied in place onto SGLang's live weights (masked overwrite of only
the changed positions), so the receiver never stages a full-model mirror anywhere -- peak
memory is one decode chunk, independent of model size.

Wire contract (what the delta collect engine sends per flush, see
:mod:`..transport.collect_engine`):

* ``__delta_spec__``  -- uint8 tensor holding a JSON manifest
  ``{"encoding", "params": [DeltaParam-dict...], "checksum"}``.
* ``__positions__`` -- uint8 blob of packed positions; per-param slices are byte offsets
  ``pos_start:pos_end`` (``indices`` packs little-endian int32 absolute positions). Absent
  for values-only (dense) flushes.
* ``__values__``   -- the changed values in the flush's (uniform) dtype; per-param slices are
  element offsets ``val_start:val_end``.
"""
from __future__ import annotations
import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch

from ..protocol.decode import DeltaPatch, decode_delta_wire

logger = logging.getLogger(__name__)

# Cap on the densified tensors handed to one model.load_weights call, matching SGLang's own
# delta-apply chunking default.
CHUNK_BYTES = 512 << 20

# Import path callers pass both in the server args' custom_weight_loader list and, per request,
# as load_format.
LOADER_FQN = 'twinkle.checkpoint_engine.delta.rollout.sglang_loader.apply_delta'


def apply_delta(model: torch.nn.Module, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
    """Decode one sparse delta flush and masked-apply it onto ``model`` in place."""
    encoding, patches = decode_delta_wire(named_tensors)  # validates the checksum, slices params
    with _masked_copy():
        chunk: list[tuple[str, torch.Tensor]] = []
        chunk_bytes = 0
        for patch in patches:
            tensor = _densify(encoding, patch)
            nbytes = tensor.numel() * tensor.element_size()
            if chunk and chunk_bytes + nbytes > CHUNK_BYTES:
                model.load_weights(chunk)
                chunk, chunk_bytes = [], 0
            chunk.append((patch.name, tensor))
            chunk_bytes += nbytes
        if chunk:
            model.load_weights(chunk)


def _densify(encoding: str, patch: DeltaPatch) -> torch.Tensor:
    """One patch as a full-shape tensor: dense values are reshaped, sparse ones NaN-masked.

    The int32 position view is reinterpreted (8 B/element transient) rather than unpacked
    per byte, and the NaN sentinel keeps the masked ``copy_`` below on-stream -- no
    device-to-host sync per parameter.
    """
    if encoding == 'dense':
        return patch.values.to(patch.dtype).view(patch.shape)
    numel = 1
    for dim in patch.shape:
        numel *= dim
    flat = torch.full((numel,), float('nan'), dtype=patch.dtype, device=patch.values.device)
    if patch.values.numel():
        flat.index_copy_(0, patch.positions.view(torch.int32).to(torch.int64),
                         patch.values.to(patch.dtype))
    return flat.view(patch.shape)


@contextmanager
def _masked_copy() -> Iterator[None]:
    """Temporarily make ``Tensor.copy_`` skip NaN positions in the source.

    SGLang's per-model ``load_weights`` ultimately lands on ``param.copy_(loaded)`` (possibly
    on a narrowed TP slice). Under this context a NaN-masked source overwrites only the changed
    positions; fully dense sources (the first full-seed flush) degenerate to a plain copy.

    The NaN sentinel assumes the model's own weights never legitimately contain NaN -- the same
    assumption the bf16 bitwise diff on the trainer side makes (a NaN weight would not round-trip
    the shard snapshot comparison anyway).
    """
    orig_copy = torch.Tensor.copy_

    def masked_copy_(self: torch.Tensor, src: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if isinstance(src, torch.Tensor) and src.is_floating_point() and self.shape == src.shape:
            cast = src.to(self.dtype)
            return orig_copy(self, torch.where(torch.isnan(cast), self, cast))
        return orig_copy(self, src, *args, **kwargs)

    torch.Tensor.copy_ = masked_copy_
    try:
        yield
    finally:
        torch.Tensor.copy_ = orig_copy
