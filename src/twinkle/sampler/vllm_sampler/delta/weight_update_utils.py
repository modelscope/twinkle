# Copyright (c) ModelScope Contributors. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright contributors to the vLLM project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Checkpoint update filtering from VERL and sparse loader adaptations for vLLM."""
from __future__ import annotations

import inspect
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from vllm.model_executor.model_loader.checkpoint_weight_patch import CheckpointWeightPatch


def tied_embedding_aliases(model: torch.nn.Module) -> set[str]:
    """Match the canonical embedding traversal used by vLLM's AutoWeightsLoader."""
    from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

    canonical: dict[int, str] = {}
    aliases: set[str] = set()
    for prefix, submodule in model.named_modules(remove_duplicate=False):
        if not isinstance(submodule, VocabParallelEmbedding):
            continue
        for name, parameter in submodule.named_parameters(remove_duplicate=False):
            qualified_name = f'{prefix}.{name}' if prefix else name
            if canonical.setdefault(id(parameter), qualified_name) != qualified_name:
                aliases.add(qualified_name)
    return aliases


def drop_tied_alias_patches(
    model: torch.nn.Module, patches: list[CheckpointWeightPatch],
) -> list[CheckpointWeightPatch]:
    """Drop redundant tied embedding names before each checkpoint patch batch.

    Canonical patches update the shared storage even when the alias arrives in
    another flush. Map checkpoint names only for filtering; vLLM still receives
    the original names and owns checkpoint-to-runtime loading.
    """
    aliases = tied_embedding_aliases(model)
    if not aliases:
        return patches
    mapper = getattr(model, 'hf_to_vllm_mapper', None)
    retained = []
    for patch in patches:
        mapped_names = mapper.apply_list([patch.name]) if mapper is not None else [patch.name]
        if not mapped_names or any(name not in aliases for name in mapped_names):
            retained.append(patch)
    return retained


def prepare_sparse_weight_loaders(model: torch.nn.Module) -> None:
    """Adapt vLLM's Mamba slice writes to the sparse patch copy_ contract.

    The native loader assigns slices directly, which bypasses vLLM's NaN mask
    and overwrites unchanged positions. Keep its TP geometry, but write through
    copy_ so the checkpoint patch API can preserve those positions.
    """
    for _, parameter in model.named_parameters():
        loader = getattr(parameter, 'weight_loader', None)
        if getattr(loader, '_twinkle_sparse_copy', False):
            continue
        if (getattr(loader, '__module__', '') == 'vllm.model_executor.layers.mamba.mamba_mixer2'
                and getattr(loader, '__qualname__', '') == 'mamba_v2_sharded_weight_loader.<locals>.loader'):
            parameter.weight_loader = _mamba_copy_weight_loader(loader)


def _mamba_copy_weight_loader(original_loader):
    geometry = inspect.getclosurevars(original_loader).nonlocals
    if not {'shard_spec', 'tp_size', 'tp_rank'}.issubset(geometry):
        raise RuntimeError('Unsupported vLLM Mamba loader closure; cannot safely apply sparse checkpoint patches')
    shard_spec = geometry['shard_spec']
    tp_size = geometry['tp_size']
    tp_rank = geometry['tp_rank']

    @wraps(original_loader)
    def copy_weight_loader(param, loaded_weight):
        boundary, loaded_boundary = 0, 0
        for full_dim, extra, duplicate_groups in shard_spec:
            shard_size = full_dim // tp_size
            rank = 0 if duplicate_groups else tp_rank
            loaded_skip = rank * shard_size
            loaded_start = loaded_boundary + loaded_skip
            take = min(shard_size, full_dim - extra - loaded_skip)
            destination = param.data[boundary:boundary + take, ...]
            source = loaded_weight[loaded_start:loaded_start + take]
            destination.copy_(source)
            boundary += shard_size
            loaded_boundary += full_dim - extra

    copy_weight_loader._twinkle_sparse_copy = True
    return copy_weight_loader


# vllm-ascend (ops/gdn.py, ops/kimi_kda.py) keeps a kernel-layout copy of each linear-attention conv1d
# weight: ``[D, 1, W]`` source -> ``[W, D]`` contiguous parameter, repacked only by
# ``process_weights_after_loading``. Sparse patches never run that hook, so the copy must be refreshed here.
_ASCEND_PACKED_CONV_WEIGHT = 'ascend_conv1d_weight'


def refresh_derived_weights(model: torch.nn.Module) -> int:
    """Re-derive runtime copies of source weights after a sparse checkpoint patch.

    Dense updates rebuild these in ``finalize_layerwise_reload``; sparse updates only write the source
    parameters. Copies are refreshed in place so captured graphs keep their addresses. Returns the number
    of refreshed tensors.
    """
    refreshed = 0
    for name, module in model.named_modules():
        packed = (getattr(module, '_parameters', None) or {}).get(_ASCEND_PACKED_CONV_WEIGHT)
        if packed is None:
            continue
        source = module.weight
        if source.dim() != 3 or source.size(1) != 1 or tuple(packed.shape) != (source.size(2), source.size(0)):
            raise RuntimeError(
                f'{name}.{_ASCEND_PACKED_CONV_WEIGHT}: unsupported packed conv layout '
                f'{tuple(packed.shape)} for source {tuple(source.shape)}; expected [W, D] from [D, 1, W]')
        # ``.data`` keeps the copy out of autograd; copy_ also casts to the packed dtype like vllm-ascend.
        packed.data.copy_(source.data.view(source.size(0), source.size(2)).transpose(0, 1))
        refreshed += 1
    return refreshed
