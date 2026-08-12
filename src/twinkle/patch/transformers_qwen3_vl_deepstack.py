# Copyright (c) ModelScope Contributors. All rights reserved.
from types import MethodType
from typing import TYPE_CHECKING, Any

from twinkle.patch import Patch

if TYPE_CHECKING:
    import torch

_MARKER = '_twinkle_sp_deepstack_patched'


class Qwen3VLDeepstackSPPatch(Patch):
    """Make Qwen3-VL's ``_deepstack_process`` sequence-parallel aware.

    Qwen3-VL injects visual features back into the text hidden states inside
    ``_deepstack_process(hidden_states, visual_pos_masks, visual_embeds)``. Under sequence
    parallelism ``hidden_states`` is already sharded along the sequence dimension for the local
    rank, while ``visual_pos_masks`` / ``visual_embeds`` still describe the full sequence, so the
    mask indexing no longer lines up. Upstream transformers has no notion of SP, hence the patch.

    The replacement does three things beyond the original:

    1. Splits ``visual_pos_masks`` / ``visual_embeds`` to the local sequence shard via
       ``sequence_parallel.pad_and_split_mm_tokens`` when SP is active.
    2. On pure-text batches (``visual_pos_masks is None``) returns
       ``hidden_states + visual_embeds.mean() * 0`` so the vision tower still receives gradients
       and DDP/FSDP collective ops stay in lockstep across ranks.
    3. Squeezes a trailing mask dim for qwen3-omni on transformers < 5.0.

    It is applied per model instance (via ``apply_patch(model, ..., sequence_parallel=sp)``); the
    ``sequence_parallel`` context is captured in the closure. Idempotent through ``_MARKER``.
    """

    def __call__(self, module: 'torch.nn.Module', *args, **kwargs) -> Any:
        sequence_parallel = kwargs.get('sequence_parallel', None)
        if module is None or sequence_parallel is None:
            return module

        def _patch_one(submodule: 'torch.nn.Module') -> bool:
            origin = getattr(submodule, '_deepstack_process', None)
            if not callable(origin):
                return False
            if getattr(submodule, _MARKER, False):
                return False

            def _deepstack_process(_self, hidden_states, visual_pos_masks, visual_embeds):
                world_size = sequence_parallel.world_size
                if world_size and world_size > 1 and visual_pos_masks is not None:
                    visual_pos_masks, visual_embeds = sequence_parallel.pad_and_split_mm_tokens(
                        visual_pos_masks, visual_embeds)
                if visual_pos_masks is None:
                    # Pure-text shard: keep the vision path in the autograd graph without altering
                    # values, so cross-rank gradient sync does not hang.
                    return hidden_states + visual_embeds.mean() * 0
                visual_pos_masks = visual_pos_masks.to(hidden_states.device)
                visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
                if hidden_states.ndim == 3 and visual_pos_masks.ndim == 3:
                    # https://github.com/huggingface/transformers/pull/41741
                    # fix qwen3-omni transformers<5.0
                    visual_pos_masks = visual_pos_masks[..., 0]
                # Clone the whole tensor, not just the masked slice: the scatter assignment below
                # writes in place, and under gradient checkpointing / SP ``hidden_states`` may be a
                # view produced by a custom autograd Function. Mutating that view in place raises
                # or corrupts gradients. See huggingface/transformers#41535 -- cloning the slice
                # (the pre-fix style) does not help because the assignment target is still a view.
                hidden_states = hidden_states.clone()
                local_this = hidden_states[visual_pos_masks, :] + visual_embeds
                hidden_states[visual_pos_masks, :] = local_this
                return hidden_states

            submodule._deepstack_process = MethodType(_deepstack_process, submodule)
            setattr(submodule, _MARKER, True)
            return True

        # ``nn.Module.modules()`` yields ``module`` itself first, so iterating covers the whole
        # tree including the root; the marker keeps it idempotent.
        for submodule in module.modules():
            _patch_one(submodule)
        return module
