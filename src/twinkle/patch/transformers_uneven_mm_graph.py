# Copyright (c) ModelScope Contributors. All rights reserved.
"""Keep vision-tower parameters in the computation graph during pure-text steps.

In distributed training (DDP / FSDP) every parameter must receive a gradient on
every rank at every step, otherwise gradient synchronization hangs.  When a
multimodal model is trained with mixed text-only and text+image data, some ranks
may receive a pure-text batch while others process images.  The text-only ranks
never invoke the vision tower, so its parameters miss the graph and the
collective stalls.

This patch fixes it by detecting uneven media distribution **across ranks** and,
on ranks that lack images, running a tiny dummy image through the vision tower
and adding ``feats.mean() * 0`` to the embedding output.  The zero term changes
nothing numerically but wires the vision parameters into autograd so they receive
a zero gradient and participate in the collective.

Implementation
--------------
Two hooks are registered on the **unwrapped** model:

1. **Top-level forward_pre_hook** — runs once per forward, safe to do
   ``all_reduce``.  It inspects the kwargs for ``pixel_values`` (and
   ``pixel_values_videos``) and synchronizes a boolean flag across ranks via
   ``dist.all_reduce(..., MAX)``.  The flag records "this rank needs a dummy
   image injection".

2. **Embedding-layer forward_hook** (on ``get_input_embeddings()``) — reads
   the flag set by (1) and, if true, runs ``model.get_image_features`` with
   dummy inputs from ``template.dummy_mm_inputs`` and adds ``feats.mean()*0``
   to the embedding output.  This hook contains **no collective** and is safe
   under reentrant gradient checkpointing (which may re-execute embed but
   never re-executes the top hook).

Both hooks are no-ops when ``torch.is_grad_enabled()`` is False (inference /
generate), so there is zero cost outside training.

Usage:
    patch = MultimodalUnevenGraphPatch(template)
    with apply_context(model, patch):
        outputs = model(**inputs)
"""
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from twinkle.patch import Patch
from twinkle.utils.torch_utils import first_tensor

if TYPE_CHECKING:
    import torch

_MARKER = '_twinkle_uneven_mm_graph'


class MultimodalUnevenGraphPatch(Patch):
    """Inject dummy vision features on pure-text ranks so DDP/FSDP gradient sync doesn't hang."""

    def __init__(self, template):
        self.template = template
        self._pre_hook_handle = None
        self._embed_hook_handle = None
        self._need_dummy: bool = False

    def __call__(self, module, *args, **kwargs):
        if getattr(module, _MARKER, False):
            return module
        import torch
        # Require both: model is multimodal and has get_image_features
        if not hasattr(module, 'get_image_features'):
            from twinkle.utils import get_logger
            get_logger().warning_once('[MultimodalUnevenGraphPatch] Model has no get_image_features; '
                                      'uneven-graph protection is disabled for this model.')
            return module

        # 1. Top-level pre-hook: cross-rank sync (safe, runs once per forward, not re-run by GC)
        def _pre_hook(model, args_tuple, kwargs_dict):
            import torch
            import torch.distributed as dist
            if not torch.is_grad_enabled():
                self._need_dummy = False
                return
            local_has_image = (
                kwargs_dict.get('pixel_values') is not None or kwargs_dict.get('pixel_values_videos') is not None)
            if dist.is_initialized():
                flag = torch.tensor([local_has_image], dtype=torch.int32, device=next(model.parameters()).device)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                global_has_image = flag.item() > 0
            else:
                global_has_image = local_has_image
            self._need_dummy = global_has_image and not local_has_image

        # 2. Embedding hook: inject zero-valued vision features into graph
        def _embed_hook(embed_module, args_tuple, output):
            import torch
            if not torch.is_grad_enabled() or not self._need_dummy:
                return output
            # Get the model (embed_module's parent)
            model = self._patched_model
            device = output.device if isinstance(output, torch.Tensor) else next(model.parameters()).device
            dtype = output.dtype if isinstance(output, torch.Tensor) else next(model.parameters()).dtype
            dummy_kwargs = self.template.dummy_mm_inputs(device=device, dtype=dtype)
            feats_out = model.get_image_features(**dummy_kwargs)
            feats = first_tensor(feats_out)
            if feats is None:
                return output
            if isinstance(output, torch.Tensor):
                return output + feats.mean() * 0.
            if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
                return (output[0] + feats.mean() * 0., ) + output[1:]
            return output

        self._patched_model = module
        self._pre_hook_handle = module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
        embed_layer = module.get_input_embeddings()
        self._embed_hook_handle = embed_layer.register_forward_hook(_embed_hook)
        setattr(module, _MARKER, True)
        return module

    def unpatch(self, module, *args, **kwargs):
        if self._pre_hook_handle is not None:
            self._pre_hook_handle.remove()
            self._pre_hook_handle = None
        if self._embed_hook_handle is not None:
            self._embed_hook_handle.remove()
            self._embed_hook_handle = None
        if hasattr(module, _MARKER):
            delattr(module, _MARKER)
        self._patched_model = None
        return module
