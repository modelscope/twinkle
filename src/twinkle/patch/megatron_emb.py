# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch a Megatron causal LM into a sentence-embedding model.

Two mutations applied to every pipeline-last-stage chunk (``post_process=True``):

1. ``output_layer.forward`` (a ``ColumnParallelLinear``) is replaced with an
   identity that returns ``(hidden_states, None)``. When ``sequence_parallel``
   is enabled, the gather across the TP group that ``ColumnParallelLinear``
   normally performs is mirrored, so the chunk's forward hook always sees a
   full-length ``[s, b, h]`` tensor.
2. A forward hook on the chunk gathers across CP (when ``cp_size > 1``),
   pools the last valid token (per-segment via ``packed_seq_params.cu_seqlens_q``
   for padding-free batches; per-row via ``position_ids`` for padded batches),
   L2-normalises and returns ``[n_seqs, hidden]`` embeddings.

Intermediate PP stages (``post_process=False``) are left untouched.

Both mutations are reverted by ``unpatch``.
"""
import torch
import torch.nn.functional as F
from types import MethodType
from typing import List, Optional

from twinkle.patch import Patch
from twinkle.utils.torch_utils import gather_cp_load_balanced


def _last_valid_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    if position_ids.dim() == 3:
        position_ids = position_ids[0]
    valid = (position_ids >= 0).int()
    seq_len = valid.shape[-1]
    return seq_len - 1 - torch.fliplr(valid).argmax(dim=-1)


def _last_valid_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    seq_len = attention_mask.shape[1]
    return seq_len - 1 - torch.fliplr(attention_mask).argmax(dim=1)


def _resolve_cp_group(module) -> Optional[object]:
    cp_group = getattr(module, 'cp_group', None)
    if cp_group is None:
        pg = getattr(module, 'pg_collection', None)
        cp_group = getattr(pg, 'cp', None) if pg is not None else None
    return cp_group


def _output_embedding_hook(module, args, kwargs, output):
    if not torch.is_tensor(output) or output.dim() != 3:
        return output

    cp_group = _resolve_cp_group(module)
    packed_seq_params = kwargs.get('packed_seq_params', None)
    cu_seqlens_q = getattr(packed_seq_params, 'cu_seqlens_q', None) if packed_seq_params is not None else None
    if cp_group is not None and cp_group.size() > 1:
        output = gather_cp_load_balanced(output, cp_group, seq_dim=1, cu_seqlens=cu_seqlens_q)

    if cu_seqlens_q is not None and cu_seqlens_q.numel() >= 2:
        # cu is full-seq based (built before CP split), so it indexes the gathered output directly.
        assert int(cu_seqlens_q[-1]) == output.shape[1], (
            f'Expected [B, T, H] with T == cu_seqlens_q[-1]; got output.shape={tuple(output.shape)}, '
            f'cu_seqlens_q[-1]={int(cu_seqlens_q[-1])}. GPTModel layout contract may have changed.')
        last_idx = (cu_seqlens_q[1:].long() - 1).to(output.device)
        embeddings = output[0, last_idx]
        return F.normalize(embeddings, p=2, dim=1).contiguous()

    position_ids = kwargs.get('position_ids', None)
    attention_mask = kwargs.get('attention_mask', None)
    if position_ids is not None and cp_group is not None and cp_group.size() > 1:
        # Non-packed branch only; reject packed-via-position_ids since CP gather without cu_seqlens scrambles segments.
        assert cu_seqlens_q is None or cu_seqlens_q.numel() < 2, (
            'Packed batch detected without usable cu_seqlens_q; CP gather of position_ids '
            'cannot reconstruct multi-segment ordering. Pass cu_seqlens_q via packed_seq_params.')
        position_ids = gather_cp_load_balanced(
            position_ids if position_ids.dim() >= 2 else position_ids.unsqueeze(0),
            cp_group,
            seq_dim=1,
        )

    if position_ids is not None:
        last_idx = _last_valid_from_position_ids(position_ids)
    elif attention_mask is not None and attention_mask.dim() == 2:
        last_idx = _last_valid_from_attention_mask(attention_mask)
    else:
        last_idx = torch.full((output.shape[0], ), output.shape[1] - 1, device=output.device, dtype=torch.long)

    last_idx = last_idx.to(device=output.device, dtype=torch.long)
    embeddings = output[torch.arange(output.shape[0], device=output.device), last_idx]
    return F.normalize(embeddings, p=2, dim=1).contiguous()


def _identity_output_layer(self, hidden_states, weight=None, runtime_gather_output=None, **kwargs):
    # Mirror ColumnParallelLinear's seq-parallel gather so the hook sees full [s, b, h].
    if getattr(self, 'sequence_parallel', False):
        from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
        hidden_states = gather_from_sequence_parallel_region(
            hidden_states, tensor_parallel_output_grad=True, group=self.tp_group)
    return hidden_states, None


def _iter_chunks(module) -> List[torch.nn.Module]:
    if isinstance(module, (list, tuple)):
        return [m for m in module if isinstance(m, torch.nn.Module)]
    return [module]


def _find_post_process_owner(chunk: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Locate the GPTModel-like owner of ``output_layer`` inside a chunk.

    Walks all submodules and matches only modules that *directly* register
    ``output_layer`` as a child (i.e., it appears in ``sub._modules``), so
    proxies via ``__getattr__`` on wrappers like ``Float16Module`` /
    ``DistributedDataParallel`` / ``PeftModel`` do not steal the match.
    """
    for sub in chunk.modules():
        if 'output_layer' not in sub._modules:
            continue
        post_process = getattr(sub, 'post_process', None)
        if post_process is None or post_process:
            return sub
    return None


class MegatronEmbeddingPatch(Patch):
    """Convert a Megatron causal LM into a sentence-embedding model. Reversible via ``unpatch``."""

    def __call__(self, module, *args, **kwargs):
        self._patched = []
        for chunk in _iter_chunks(module):
            owner = _find_post_process_owner(chunk)
            if owner is None:
                continue
            output_layer = owner.output_layer
            origin_forward = output_layer.forward
            output_layer.forward = MethodType(_identity_output_layer, output_layer)
            hook_handle = owner.register_forward_hook(_output_embedding_hook, with_kwargs=True)
            self._patched.append((output_layer, origin_forward, hook_handle))
        return module

    def unpatch(self, module, *args, **kwargs):
        for output_layer, origin_forward, hook_handle in self._patched:
            hook_handle.remove()
            output_layer.forward = origin_forward
        self._patched = []
        return module
