# Copyright (c) ModelScope Contributors. All rights reserved.
"""Keep the gradient path into a frozen input-embedding layer usable.

Two independent forward-hook mutations on the module that produces the first
activation of the graph (``get_input_embeddings()`` unless ``module_key`` says
otherwise):

1. ``TransformersInputRequireGradsPatch`` -- ``output.requires_grad_(True)``.
   Required by the *reentrant* gradient-checkpointing variant
   (``use_reentrant=True``): its backward re-runs forward under ``no_grad`` and
   then calls the autograd engine w.r.t. the block inputs, so when no input
   requires grad the recomputed subgraph carries no ``grad_fn`` and backward dies
   with ``element 0 of tensors does not require grad and does not have a
   grad_fn``. LoRA freezes the embedding, so LoRA + reentrant GC is exactly this
   case.

   ``PreTrainedModel.gradient_checkpointing_enable`` already installs such a hook
   when ``main_input_name == 'input_ids'`` or a peft config is loaded, which
   covers most LLMs/VLMs. This patch is for the models it skips (``input_features``
   / ``input_values`` audio towers), or when the grad entry point is not the input
   embedding at all -- a conv stem takes ``module_key='model.encoder.conv1'``.

2. ``TransformersOutputClonePatch`` -- ``output.requires_grad_(True).clone()``.
   Repairs the fallout of (1): a tensor that had ``requires_grad_(True)`` called on
   it is a **leaf**, and multimodal forwards that splice encoder features in place
   (``inputs_embeds.masked_scatter_(mask, feats)`` -- ovis2, csm, qwen2_5_omni and
   plenty of remote-code models) then raise ``a leaf Variable that requires grad is
   being used in an in-place operation``. ``clone()`` re-parents the tensor onto
   ``CloneBackward`` so it is no longer a leaf and the in-place write becomes
   legal. The order is load-bearing: ``requires_grad_`` must run *before*
   ``clone``, otherwise the clone gets no ``grad_fn`` and stays a leaf. This also
   breaks tensor aliasing for identity-like modules -- ``nn.Dropout(p=0)`` returns
   its input object untouched and propagates leaf-ness downstream.

   Composes with the hook transformers registers: that one runs first (registered
   earlier) and only flips the flag, then this one clones, so the module's final
   output is a non-leaf either way.

The two are alternatives, not a pair: (2)'s hook already does (1)'s flag flip, so
applying both to the same module only buys a redundant no-op. Using them together
is meaningful only on *different* modules -- e.g. (1) on an audio conv stem that
is the real grad entry point, plus (2) on ``embed_tokens`` where features get
spliced in place.

Neither patch is needed under ``use_reentrant=False``: the non-reentrant variant
builds a real graph via saved-tensor hooks and does not care whether its inputs
require grad. That is the transformers>=5 default and therefore what
``TransformersModel``'s bare ``gradient_checkpointing_enable()`` selects, so
prefer fixing the checkpointing flavour over stacking these hooks -- they exist
for the cases where that is not an option. Same for (2): when the only reason the
activation is a leaf is the hook transformers installed, dropping that hook via
``model.disable_input_require_grads()`` is cheaper than cloning it away, since the
clone costs one ``[B, T, H]`` copy per step.

Both are reverted by ``unpatch``, and both no-op while grad mode is off so
inference/generate pays neither the hook nor the extra ``[B, T, H]`` copy.
"""
from typing import TYPE_CHECKING, Any, Callable, Optional

from twinkle.patch import Patch
from twinkle.utils import deep_getattr

if TYPE_CHECKING:
    import torch


def _map_first_tensor(output: Any, fn: Callable[['torch.Tensor'], 'torch.Tensor']) -> Any:
    """Apply ``fn`` to the activation tensor, leaving any co-returned values alone."""
    import torch
    if isinstance(output, torch.Tensor):
        return fn(output)
    # Some modules (and remote-code embeddings) return (hidden_states, *rest).
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (fn(output[0]), *output[1:])
    if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
        return [fn(output[0]), *output[1:]]
    return output


def _require_grads(tensor: 'torch.Tensor') -> 'torch.Tensor':
    # Integer outputs cannot carry grad; asking would raise instead of being a no-op.
    if tensor.is_floating_point():
        tensor.requires_grad_(True)
    return tensor


def _require_grads_hook(module, args, output):
    import torch
    if not torch.is_grad_enabled():
        return output
    return _map_first_tensor(output, _require_grads)


def _clone_hook(module, args, output):
    import torch
    if not torch.is_grad_enabled():
        return output
    return _map_first_tensor(output, lambda tensor: _require_grads(tensor).clone())


def _resolve_target(module, module_key: Optional[str]) -> 'torch.nn.Module':
    from torch.nn import Module
    if module_key:
        target = deep_getattr(module, module_key)
        assert isinstance(target, Module), f'module_key {module_key!r} does not resolve to a torch Module: {target}'
        return target
    get_input_embeddings = getattr(module, 'get_input_embeddings', None)
    assert callable(get_input_embeddings), (
        f'{type(module).__name__} has no get_input_embeddings(); pass module_key to name the grad entry point.')
    target = get_input_embeddings()
    assert isinstance(target, Module), f'get_input_embeddings() returned {target}'
    return target


class _InputGradsHookPatch(Patch):
    """Register one reversible forward hook on the graph's first activation producer."""

    def __init__(self, module_key: Optional[str] = None):
        self._module_key = module_key
        self._hook_handle = None

    def _make_hook(self) -> Callable:
        raise NotImplementedError()

    def __call__(self, module, *args, **kwargs):
        # Idempotent: re-applying the same instance must not stack a second hook.
        if self._hook_handle is not None:
            return module
        target = _resolve_target(module, self._module_key)
        self._hook_handle = target.register_forward_hook(self._make_hook())
        return module

    def unpatch(self, module, *args, **kwargs):
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None
        return module


class TransformersInputRequireGradsPatch(_InputGradsHookPatch):
    """Mark the first activation as requiring grad, so reentrant GC can reach frozen layers."""

    def _make_hook(self) -> Callable:
        return _require_grads_hook


class TransformersOutputClonePatch(_InputGradsHookPatch):
    """Clone the first activation so it is a non-leaf and in-place feature splicing is legal."""

    def _make_hook(self) -> Callable:
        return _clone_hook
