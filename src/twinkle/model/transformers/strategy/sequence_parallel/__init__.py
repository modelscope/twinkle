# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import threading
import torch
import torch.distributed as dist
from copy import copy
from dataclasses import asdict, dataclass, is_dataclass
from contextlib import contextmanager, nullcontext
from functools import wraps
from types import MethodType
from weakref import WeakKeyDictionary, WeakMethod, WeakSet
from transformers import PreTrainedTokenizer
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from twinkle.patch import apply_patch
from twinkle.utils import DeviceMesh, get_logger
from twinkle.utils.transformers_utils import get_llm_model, is_flash_attention_implementation
from twinkle.utils.utils import call_with_supported_kwargs, has_signature_parameter
from twinkle.patch.transformers_qwen3_vl_deepstack import Qwen3VLDeepstackSPPatch
from .linear_attention_sp import Qwen3_5GatedDeltaNetUlyssesPatch, _iter_qwen35_gated_delta_net_classes
from .utils import (DistributedAttention, GatherLoss, _derive_sequence_parallel_sizes, _get_seq_groups_from_device_mesh,
                    _get_ulysses_size, _SeqAllToAll, get_config_attr, get_cu_seqlens_from_position_ids, is_hccl_backend,
                    is_moe_config, post_all2all)

logger = get_logger()


_MODULE_TO_CTX = WeakKeyDictionary()
_CTX_LOCAL = threading.local()


def _current_ctx():
    stack = getattr(_CTX_LOCAL, 'stack', ())
    return stack[-1] if stack else None


@contextmanager
def _use_context(sp):
    stack = getattr(_CTX_LOCAL, 'stack', None)
    if stack is None:
        stack = _CTX_LOCAL.stack = []
    depth = len(stack)
    stack.append(sp)
    try:
        yield
    finally:
        del stack[depth:]


def _attention_context(module):
    sp = _MODULE_TO_CTX.get(module)
    active = _current_ctx()
    if sp is None:
        if active is not None:
            raise RuntimeError(f'SP model {active.model_name}: unregistered attention module '
                               f'{type(module).__module__}.{type(module).__name__}')
        return None
    if module not in sp._text_modules:
        return None  # Explicitly registered local (e.g. vision) computation.
    if active is not sp:
        raise RuntimeError(f'SP model {sp.model_name}: attention {sp._module_names[module]} '
                           'called outside its owning context')
    return sp


def _call_flash_attention(origin, *args, **kwargs):
    position_ids = kwargs.get('position_ids')
    if torch.is_tensor(position_ids):
        # These are boundary metadata, not the positions used for RoPE. HF may
        # infer sequence starts from position_ids.min(); SP's -1 padding would
        # then hide every real sequence. Continue the preceding positions over
        # padding while retaining all real positions and packed restarts.
        indices = torch.arange(position_ids.shape[-1], device=position_ids.device)
        last_valid = torch.where(position_ids != -1, indices, 0).cummax(dim=-1).values
        continued = position_ids.gather(-1, last_valid).clamp_min(0) + indices - last_valid
        kwargs['position_ids'] = torch.where(position_ids == -1, continued, position_ids)
    return origin(*args, **kwargs)


def is_qwen3_vl(model):
    mt = getattr(getattr(model, 'config', None), 'model_type', '')
    return 'qwen3_vl' in mt


def _call_create_causal_mask(fn, config, input_embeds, attention_mask, cache_position_or_past_key_values, *args,
                             **kwargs):
    # Newer transformers replaces cache_position with q_length/q_offset kwargs; on the no-cache prefill
    # path the positional cache_position we supply is not accepted and must be dropped.
    drop_positional = (not has_signature_parameter(fn, 'cache_position') and cache_position_or_past_key_values is None
                       and 'past_key_values' in kwargs)
    if drop_positional:
        return call_with_supported_kwargs(fn, config, input_embeds, attention_mask, *args, **kwargs)
    return call_with_supported_kwargs(
        fn,
        config,
        input_embeds,
        attention_mask,
        cache_position_or_past_key_values,
        *args,
        **kwargs,
    )


# `attn_implementation` values that follow the FlashAttention contract (transposed q/k/v, optional
# 2D padding mask, `cu_seq_lens_*` for packed batches). They share transformers' single
# `flash_attention_forward` entry point and differ only in the leaf kernel, so SP treats them alike.
FLASH_ATTENTION_IMPLS = ('flash_attention_2', 'flash_attention_3')


# main content copied from ms-swift
class SequenceParallel:

    _global_inited: bool = False

    def __init__(self):
        self.seq_world_size = None
        self.sp_world_size = None
        self.rp_world_size = None
        self.dp_world_size = None
        self.world_size = None
        self.attn_implementation = None
        self.model_dtype = None
        self.tokenizer = None
        self.device_mesh = None
        self._sp_group = None
        self._rp_group = None
        self._data_rank_group = None
        self._sp_rank = 0
        self._rp_rank = 0
        self.num_heads = None
        self.causal_mask_func = None
        self.extra_kwargs = {}
        self._dist_attns = {}
        self._text_modules = WeakSet()
        self._module_names = WeakKeyDictionary()
        self._defer_inputs = False
        self._packed_linear_supported = True

    @property
    def real_position_ids(self) -> torch.Tensor:
        """The real position ids, this is different from the position_ids in mrope"""
        return self.extra_kwargs.get('position_ids')

    @staticmethod
    def _extract_real_position_ids(position_ids: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if position_ids is None or not torch.is_tensor(position_ids):
            return position_ids
        if position_ids.dim() == 3:
            return position_ids[0]
        return position_ids

    @property
    def sp_rank(self) -> int:
        return self._sp_rank

    @property
    def rp_rank(self) -> int:
        return self._rp_rank

    def _get_dist_attn(self, key: str, local_attention=None) -> DistributedAttention:
        """Per-context DistributedAttention cache (one per attention entry point)."""
        dist_attn = self._dist_attns.get(key)
        if dist_attn is None:
            dist_attn = DistributedAttention(local_attention, self)
            self._dist_attns[key] = dist_attn
        return dist_attn

    @staticmethod
    def _sdpa_mask_impl(sp, origin_sdpa, origin_uses_cache_position, batch_size, q_length=None, kv_length=None, *args,
                        **kwargs):
        q_length = q_length if q_length is not None else kwargs.pop('cache_position', None)
        device = q_length.device if torch.is_tensor(q_length) else kwargs.pop('device', None)
        if device is None and sp is not None:
            device = sp.real_position_ids.device

        cache_position = None
        if sp is not None and sp.world_size > 1:
            padded_position_ids = sp.pad(
                sp.real_position_ids[0],
                padding_value=-1,
                position_ids=sp.real_position_ids,
                dim=0,
            )
            global_length = padded_position_ids.shape[0]
            if origin_uses_cache_position:
                cache_position = torch.arange(0, global_length, device=device)
                kv_length = global_length
            else:
                # Newer Transformers passes q_length/q_offset instead of cache_position. In SP training,
                # create_causal_mask may still see the local shard length, so restore the global query length
                # only for the no-cache prefill path; cache/sliding paths keep their upstream offsets.
                q_offset = kwargs.get('q_offset', 0)
                kv_offset = kwargs.get('kv_offset', 0)
                no_cache_offsets = ((not torch.is_tensor(q_offset) and q_offset == 0)
                                    and (not torch.is_tensor(kv_offset) and kv_offset == 0))
                if no_cache_offsets:
                    q_length = global_length
                    attention_mask = kwargs.get('attention_mask')
                    if attention_mask is not None and torch.is_tensor(attention_mask):
                        kv_length = attention_mask.shape[-1]
                    else:
                        kv_length = global_length

        # padding_mask may be an AttentionMask wrapper instead of Tensor
        _pm = kwargs.get('padding_mask')
        if _pm is not None and not isinstance(_pm, torch.Tensor):
            if hasattr(_pm, 'to_tensor'):
                kwargs['padding_mask'] = _pm.to_tensor()
            elif hasattr(_pm, 'mask'):
                kwargs['padding_mask'] = _pm.mask

        if origin_uses_cache_position:
            if cache_position is None:
                cache_position = q_length if torch.is_tensor(q_length) else torch.arange(q_length, device=device)
            return origin_sdpa(batch_size, cache_position, kv_length, *args, **kwargs)

        return origin_sdpa(batch_size, q_length, kv_length, *args, device=device, **kwargs)

    @staticmethod
    def _create_causal_mask_impl(sp,
                                 config,
                                 input_embeds,
                                 attention_mask,
                                 cache_position_or_past_key_values=None,
                                 *args,
                                 **kwargs):
        from transformers import masking_utils

        if sp is None or sp.world_size == 1:
            return _call_create_causal_mask(
                masking_utils.origin_create_causal_mask,
                config,
                input_embeds,
                attention_mask,
                cache_position_or_past_key_values,
                *args,
                **kwargs,
            )
        input_embeds = torch.ones(
            (input_embeds.shape[0], input_embeds.shape[1] * sp.sp_world_size, input_embeds.shape[2]),
            dtype=input_embeds.dtype,
            device=input_embeds.device)
        if has_signature_parameter(masking_utils.origin_create_causal_mask, 'cache_position'):
            cache_position_or_past_key_values = torch.arange(
                0,
                input_embeds.shape[1],
                device=input_embeds.device,
            )
        return _call_create_causal_mask(
            masking_utils.origin_create_causal_mask,
            config,
            input_embeds,
            attention_mask,
            cache_position_or_past_key_values,
            *args,
            **kwargs,
        )

    def _local_flash_attn_impl(self, module, query_states, key_states, value_states, attention_mask, *args, **kwargs):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        dist_attn = self._get_dist_attn('flash_attention_2')
        if dist_attn.local_attn is None:

            # Pass the current layer at call time. Capturing it in this cached
            # closure would keep a weak-registry key alive through its context.
            def _attention(query, key, value, *args, _sp_module, **kwargs):
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                if self.rp_world_size > 1:
                    from .zigzag_ring_attn import zigzag_ring_flash_attn_varlen_func

                    position_ids = kwargs.get('position_ids')
                    if position_ids is None:
                        position_ids = self.real_position_ids
                    position_ids = self._extract_real_position_ids(position_ids)
                    cu_seqlens = get_cu_seqlens_from_position_ids(position_ids).to(torch.int32)
                    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                    position_ids = self._split_packed(position_ids, cu_seqlens, dim=-1)
                    mask = position_ids != -1
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    query, key, value = self._mask_qkv(query, key, value, mask)
                    return zigzag_ring_flash_attn_varlen_func(
                        query,
                        key,
                        value,
                        cu_seqlens=cu_seqlens,
                        max_seqlen=max_seqlen,
                        dropout_p=kwargs.get('dropout', 0.0),
                        softmax_scale=kwargs.get('scaling'),
                        causal=_sp_module.is_causal,
                        window_size=kwargs.get('sliding_window') or (-1, -1),
                        group=self._rp_group,
                        backend=self.attn_implementation,
                    )
                elif self.extra_kwargs.get('padding_free', False) or 'cu_seq_lens_q' in kwargs:
                    position_ids = kwargs.get('position_ids')
                    if position_ids is None:
                        position_ids = self.real_position_ids
                    if position_ids is None:
                        raise ValueError('Packed/varlen flash_attention_2 requires position_ids to derive '
                                         'cu_seq_lens_q.')
                    position_ids = self._extract_real_position_ids(position_ids)
                    position_ids = self.pad(position_ids, padding_value=-1, position_ids=position_ids)
                    cu_seqlens = get_cu_seqlens_from_position_ids(position_ids).to(
                        dtype=torch.int32,
                        device=query.device,
                    )
                    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                    total_tokens = int(cu_seqlens[-1].item())
                    if query.shape[2] != total_tokens:
                        raise ValueError('Packed/varlen flash_attention_2 expects query sequence length to match '
                                         f'cu_seqlens total tokens, got query_seq_len={query.shape[2]} '
                                         f'and cu_seqlens_total={total_tokens}.')
                    kwargs['cu_seq_lens_q'] = cu_seqlens
                    kwargs['cu_seq_lens_k'] = cu_seqlens
                    kwargs['max_length_q'] = max_seqlen
                    kwargs['max_length_k'] = max_seqlen
                    if self.extra_kwargs.get('padding_free', False) and len(args) > 0:
                        args = (None, *args[1:])
                backend = getattr(getattr(_sp_module, 'config', None), '_attn_implementation', None)
                origin = ALL_ATTENTION_FUNCTIONS.get(f'{backend}_origin',
                                                     ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'])
                return _call_flash_attention(origin, _sp_module, query, key, value, *args, **kwargs)[0]

            dist_attn.local_attn = _attention

        return dist_attn(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
            *args, _sp_module=module, **kwargs), None

    def _local_sdpa_attn_impl(self, module, query_states, key_states, value_states, attention_mask, *args, **kwargs):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        # Policy: packed (PackingDataset/padding-free) batches require FlashAttention2 varlen/packed semantics.
        # SDPA does not have a native packed/varlen interface; supporting packed batches would require building a
        # large block-diagonal causal mask (slow / memory heavy).
        if self.extra_kwargs.get('padding_free', False):
            raise RuntimeError('SequenceParallel: detected padding_free/packed batch. '
                               'SDPA backend is not supported for padding_free/packed batches; please use '
                               'flash_attention_2.')
        dist_attn = self._get_dist_attn('sdpa')
        if dist_attn.local_attn is None:

            # The cache is per context, not per layer; do not bind the first
            # layer or retain it through the global weak-key registry.
            def _attention(query, key, value, *args, _sp_module, **kwargs):
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                if self.rp_world_size > 1:
                    raise NotImplementedError('SDPA does not support derived ring attention.')
                return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](_sp_module, query, key, value, *args, **kwargs)[0]

            dist_attn.local_attn = _attention
        return dist_attn(
            query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
            *args, _sp_module=module, **kwargs), None

    def _prepare_flash_attn(self):
        """Install the global SP-aware attention/mask wrappers (once per process).

        The installed wrappers carry no per-model state; they resolve the owning
        SequenceParallel context at call time via ``_MODULE_TO_CTX`` (attention
        functions) or the thread-local ctx stack (mask helpers), so multiple
        SP-enabled models (and non-SP models) can coexist in one process.
        """
        try:
            from transformers import masking_utils

            origin_sdpa = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa']
            origin_uses_cache_position = has_signature_parameter(origin_sdpa, 'cache_position')

            def sdpa_mask(batch_size, q_length=None, kv_length=None, *args, **kwargs):
                return SequenceParallel._sdpa_mask_impl(_current_ctx(), origin_sdpa, origin_uses_cache_position,
                                                        batch_size, q_length, kv_length, *args, **kwargs)

            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa_origin'] = origin_sdpa
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa'] = sdpa_mask

            def create_causal_mask(config,
                                   input_embeds,
                                   attention_mask,
                                   cache_position_or_past_key_values=None,
                                   *args,
                                   **kwargs):
                return SequenceParallel._create_causal_mask_impl(_current_ctx(), config, input_embeds, attention_mask,
                                                                 cache_position_or_past_key_values, *args, **kwargs)

            masking_utils.origin_create_causal_mask = masking_utils.create_causal_mask
            masking_utils.create_causal_mask = create_causal_mask
        except ImportError:
            pass

        from transformers.modeling_flash_attention_utils import is_flash_attn_available
        if is_flash_attn_available():
            # TODO this works for multi-modal models like qwen2.5-vl
            # SDPA is not supported here, because we need to copy the code to our project, which will bring
            # more work for maintaining.
            from transformers import modeling_flash_attention_utils
            from transformers.modeling_flash_attention_utils import _flash_attention_forward

            modeling_flash_attention_utils._flash_attention_forward_origin = _flash_attention_forward

            def local_flash_attention_forward(*args, **kwargs):
                return _call_flash_attention(_flash_attention_forward, *args, **kwargs)

            def flash_attention_forward(query_states: torch.Tensor, key_states: torch.Tensor,
                                        value_states: torch.Tensor, attention_mask: Optional[torch.Tensor], q_len,
                                        *args, **kwargs):
                sp = _current_ctx()
                if sp is None or sp.world_size == 1:
                    return _flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len,
                                                    *args, **kwargs)
                dist_attn = sp._get_dist_attn('flash_attention_utils', local_flash_attention_forward)
                return dist_attn(query_states, key_states, value_states, attention_mask, q_len * sp.sp_world_size,
                                 *args, **kwargs)

            modeling_flash_attention_utils._flash_attention_forward = flash_attention_forward

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def local_flash_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                             **kwargs):
            sp = _attention_context(module)
            if sp is None or sp.world_size == 1:
                backend = getattr(getattr(module, 'config', None), '_attn_implementation', None)
                origin = ALL_ATTENTION_FUNCTIONS.get(f'{backend}_origin',
                                                     ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'])
                return origin(module, query_states, key_states, value_states, attention_mask, *args, **kwargs)
            return sp._local_flash_attn_impl(module, query_states, key_states, value_states, attention_mask, *args,
                                             **kwargs)

        def local_sdpa_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                            **kwargs):
            sp = _attention_context(module)
            if sp is None or sp.world_size == 1:
                return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query_states, key_states, value_states,
                                                              attention_mask, *args, **kwargs)
            return sp._local_sdpa_attn_impl(module, query_states, key_states, value_states, attention_mask, *args,
                                            **kwargs)

        # `flash_attention_2/3` all hold the same `flash_attention_forward` object in transformers; the
        # concrete kernel is picked at call time from `module.config._attn_implementation`. So the SP
        # wrapper has to be registered under every FlashAttention name: `AttentionInterface.get_interface`
        # is an exact-key lookup, so an FA3 config left unwrapped would silently train *without* sequence
        # parallelism. One `_origin` alias is enough because `local_flash_attn` dispatches through the
        # module config, not through the key it was reached by.
        ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']
        for _impl in FLASH_ATTENTION_IMPLS:
            if _impl not in ALL_ATTENTION_FUNCTIONS:
                continue
            ALL_ATTENTION_FUNCTIONS[f'{_impl}_origin'] = ALL_ATTENTION_FUNCTIONS[_impl]
            ALL_ATTENTION_FUNCTIONS[_impl] = local_flash_attn
        ALL_ATTENTION_FUNCTIONS['sdpa'] = local_sdpa_attn
        logger.info_once(f'[SequenceParallel] attention wrappers registered: {list(FLASH_ATTENTION_IMPLS)} -> '
                         'local_flash_attn, sdpa -> local_sdpa_attn (SP-aware)')

    @property
    def causal_mask_func(self):
        func = self._causal_mask_func
        return func() if isinstance(func, WeakMethod) else func

    @causal_mask_func.setter
    def causal_mask_func(self, func):
        # The global weak-key registry owns this context. A bound method must
        # not keep the owning model (and therefore its weak key) alive.
        self._causal_mask_func = WeakMethod(func) if getattr(func, '__self__', None) is not None else func

    @contextmanager
    def context(self, state=None):
        previous = self.extra_kwargs
        if state is not None:
            self.extra_kwargs = state
        try:
            with _use_context(self):
                yield
        finally:
            self.extra_kwargs = previous

    def _prepare_forward_boundaries(self, model, text_model):
        self.model_name = type(model).__name__
        text_modules = tuple(text_model.modules())
        named_modules = tuple(model.named_modules())
        self._defer_inputs = any(getattr(getattr(module, 'config', None), 'vision_config', None) is not None
                                 for _, module in named_modules)
        if not any('Attention' in type(module).__name__ or 'GatedDeltaNet' in type(module).__name__
                   for module in text_modules):
            raise NotImplementedError(f'SP {self.model_name}: cannot identify the text attention modules')
        for module in text_modules:
            if 'GatedDeltaNet' in type(module).__name__ and any(
                    getattr(module, name) % self.sp_world_size for name in ('num_k_heads', 'num_v_heads')):
                raise NotImplementedError(f'SP {self.model_name}: incompatible GDN head counts')
        vision = [module for name, module in named_modules
                  if name.rsplit('.', 1)[-1] in ('visual', 'vision_tower', 'vision_model')]
        if self._defer_inputs and (not vision or any(module in text_modules for module in vision)):
            raise NotImplementedError(f'SP {self.model_name}: cannot isolate local vision from text')
        self._text_modules.update(text_modules)
        for name, module in named_modules:
            _MODULE_TO_CTX[module] = self
            self._module_names[module] = name or '<root>'

        def wrap_forward(module, local=False):
            origin = WeakMethod(module.forward)
            @wraps(module.forward.__func__)
            def forward(_module, *args, **kwargs):
                if local:
                    with _use_context(None):
                        return origin()(*args, **kwargs)
                if args:
                    raise NotImplementedError(f'{self.model_name}: SP text entry requires keyword inputs')
                kwargs = self._split_model_inputs(dict(kwargs))
                return origin()(**kwargs)
            module.forward = MethodType(forward, module)

        if self._defer_inputs:
            wrap_forward(text_model)
            for module in vision:
                wrap_forward(module, local=True)
        for module in text_model.modules():
            original = getattr(module, '_gradient_checkpointing_func', None)
            if original is None:
                continue
            def checkpoint(function, *args, _checkpoint=original, **kwargs):
                state = copy(self.extra_kwargs)
                def recompute(*args, **kwargs):
                    with self.context(state):
                        return function(*args, **kwargs)
                return _checkpoint(recompute, *args, **kwargs)
            module._gradient_checkpointing_func = checkpoint

    def _split_model_inputs(self, inputs):
        positions = self._extract_real_position_ids(inputs.get('position_ids'))
        model_positions = inputs.get('position_ids')
        if self._defer_inputs and torch.is_tensor(model_positions) and model_positions.dim() == 3:
            positions = self.real_position_ids  # mRoPE axes are not the packed token layout.
        if positions is not None:
            self.extra_kwargs['position_ids'] = positions.clone()
        if inputs.get('attention_mask') is None and not self.extra_kwargs.get('padding_free', False):
            inputs['attention_mask'] = (self.real_position_ids >= 0).long()
        mask = inputs.get('attention_mask')
        if torch.is_tensor(mask) and mask.dim() == 4:
            length = self.pad(self.real_position_ids, padding_value=-1).shape[-1]
            if self.attn_implementation != 'sdpa' or mask.shape[1] != 1 or mask.shape[-2:] != (length, length):
                raise NotImplementedError('SP needs a 2D padding mask or an aligned broadcast SDPA 4D mask')
        if torch.is_tensor(mask) and mask.dim() == 2:
            inputs['attention_mask'] = self.pad(mask, padding_value=0, position_ids=self.real_position_ids)
        ids, embeds, _, positions, mask, _, _ = self.pad_and_split_inputs(
            inputs.get('input_ids'), inputs.get('inputs_embeds'), None,
            inputs.get('position_ids'), inputs.get('attention_mask'), None,
            real_position_ids=self.real_position_ids, cache_position=inputs.get('cache_position'))
        self.extra_kwargs['attention_mask'] = inputs.get('attention_mask')
        inputs.update(input_ids=ids, inputs_embeds=embeds, position_ids=positions, attention_mask=mask)
        return inputs


    def _prepare_multimodal_deepstack(self, base_model: torch.nn.Module):
        if not is_qwen3_vl(base_model):
            return
        apply_patch(base_model, Qwen3VLDeepstackSPPatch, sequence_parallel=self)

    @staticmethod
    def _is_qwen35_model(model: torch.nn.Module) -> bool:
        config = getattr(model, 'config', None)
        model_type = str(getattr(config, 'model_type', '') or '')
        if model_type in {'qwen3_5', 'qwen3_5_moe'}:
            return True

        architectures = getattr(config, 'architectures', None) or []
        if any('Qwen3_5' in str(arch) for arch in architectures):
            return True

        model_module = getattr(model.__class__, '__module__', '') or ''
        return 'transformers.models.qwen3_5' in model_module or 'transformers.models.qwen3_5_moe' in model_module

    def _prepare_qwen35_linear_attention(self, model: torch.nn.Module):
        has_qwen35_linear_attention = self._is_qwen35_model(model)
        if not has_qwen35_linear_attention:
            gated_delta_net_classes = tuple(_iter_qwen35_gated_delta_net_classes())
            if not gated_delta_net_classes:
                return
            has_qwen35_linear_attention = any(isinstance(module, gated_delta_net_classes) for module in model.modules())
        if not has_qwen35_linear_attention:
            return
        if int(self.rp_world_size or 1) > 1:
            raise NotImplementedError(
                'SequenceParallel: Qwen3.5 linear attention sequence parallel does not support rp_world_size > 1 '
                '(derived ring attention).')
        from .linear_attention_sp import _FLA_CAUSAL_CONV1D_FN, _FLA_CHUNK_GATED_DELTA_RULE
        self._packed_linear_supported = bool(_FLA_CAUSAL_CONV1D_FN and _FLA_CHUNK_GATED_DELTA_RULE) or all(
            getattr(module, '_twinkle_npu_patched', False) for module in model.modules()
            if 'GatedDeltaNet' in type(module).__name__)
        apply_patch(None, Qwen3_5GatedDeltaNetUlyssesPatch, sequence_parallel=self)

    def _prepare_moe_aux_loss(self, base_model: torch.nn.Module):

        def moe_aux_loss_hook(module, args, kwargs, output):
            router_logits = getattr(output, 'router_logits', None)
            if router_logits is None:
                return output

            attention_mask = kwargs['attention_mask']
            if attention_mask is None:
                batch_size = 1
            else:
                batch_size = attention_mask.shape[0]

            assert router_logits[0].shape[0] % batch_size == 0
            seq_len = router_logits[0].shape[0] // batch_size

            _gathered_logits = []
            for i in range(batch_size):
                _slice = slice(i * seq_len, (i + 1) * seq_len)
                _bs_logits = [logit[_slice] for logit in router_logits]
                compute_device = _bs_logits[0].device
                _bs_logits = torch.stack([layer_gate.to(compute_device) for layer_gate in _bs_logits], dim=0)
                _bs_logits, _ = GatherLoss.apply(self, _bs_logits, None, 1, self.real_position_ids)
                _gathered_logits.append(_bs_logits)
            router_logits = torch.stack(_gathered_logits, dim=0)
            if self.real_position_ids is not None:
                router_logits = router_logits[:, :, :self.real_position_ids.shape[1], :]
            output['router_logits'] = tuple(
                [logit.reshape(-1, logit.shape[-1]) for logit in router_logits.split(1, dim=1)])
            return output

        base_model.register_forward_hook(moe_aux_loss_hook, with_kwargs=True)

    def prepare(
        self,
        sp_size: int,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        root_model = model
        while isinstance(getattr(root_model, 'module', None), torch.nn.Module):
            root_model = root_model.module
        llm_model = get_llm_model(root_model)
        if not (getattr(llm_model, '_supports_attention_backend', False) or self._is_qwen35_model(llm_model)):
            raise NotImplementedError(f'SP {type(llm_model).__name__}: unsupported text attention entry point')
        llm_model.config.use_cache = False
        config_candidates = [getattr(root_model, 'config', None)]
        llm_config = getattr(llm_model, 'config', None)
        if llm_config is not None and llm_config not in config_candidates:
            config_candidates.append(llm_config)
        text_config = getattr(getattr(root_model, 'config', None), 'text_config', None)
        if text_config is not None and text_config not in config_candidates:
            config_candidates.append(text_config)

        self.num_heads = None
        for config in config_candidates:
            if config is None:
                continue
            self.num_heads = get_config_attr(config, 'num_key_value_heads')
            if self.num_heads is None:
                self.num_heads = get_config_attr(config, 'num_attention_heads')
            if self.num_heads is not None:
                break
        if not isinstance(self.num_heads, int) or self.num_heads < 1:
            raise NotImplementedError(f'SP {type(model).__name__}: cannot determine positive attention head count')
        self.seq_world_size = sp_size
        self.sp_world_size, self.rp_world_size = _derive_sequence_parallel_sizes(self.num_heads, self.seq_world_size)
        self.world_size = self.seq_world_size

        self.attn_implementation = None
        for config in config_candidates:
            if config is None:
                continue
            self.attn_implementation = getattr(config, '_attn_implementation', None)
            if self.attn_implementation is not None:
                break

        if hasattr(llm_model, 'language_model'):
            if hasattr(llm_model.language_model, '_update_causal_mask'):
                self.causal_mask_func = llm_model.language_model._update_causal_mask
        else:
            if hasattr(llm_model, '_update_causal_mask'):
                self.causal_mask_func = llm_model._update_causal_mask

        if self.attn_implementation not in ('sdpa', *FLASH_ATTENTION_IMPLS):
            raise NotImplementedError(f'SP {type(model).__name__}: unsupported attention {self.attn_implementation!r}')
        if self.attn_implementation in FLASH_ATTENTION_IMPLS:
            from transformers.modeling_flash_attention_utils import is_flash_attn_available
            from transformers.utils import is_flash_attn_2_available, is_flash_attn_3_available
            device = next(model.parameters()).device.type
            available = device in ('cuda', 'npu') and (
                is_flash_attn_2_available() if device == 'cuda' else is_flash_attn_available())
            if self.attn_implementation == 'flash_attention_3':
                available = device == 'cuda' and is_flash_attn_3_available()
            if not available:
                raise NotImplementedError(f'SP {self.attn_implementation} kernel unavailable on {device}')
        if self.rp_world_size > 1 and self.attn_implementation == 'sdpa':
            raise NotImplementedError('SDPA does not support derived ring attention')
        if self.rp_world_size > 1 and next(model.parameters()).device.type != 'cuda':
            raise NotImplementedError('Derived ring attention requires CUDA FlashAttention kernels')
        self._init_device_mesh(device_mesh)
        if not SequenceParallel._global_inited:
            # these operations are global initializations and patches
            self._prepare_flash_attn()
            SequenceParallel._global_inited = True
        self._prepare_qwen35_linear_attention(llm_model)

        text_model = getattr(llm_model, 'language_model', llm_model)
        self._prepare_forward_boundaries(model, text_model)
        self._prepare_multimodal_deepstack(llm_model)

        if is_moe_config(getattr(llm_model, 'config', None)):
            self._prepare_moe_aux_loss(llm_model)

        self.model_dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer
        if self.rp_world_size > 1:
            attn_impl = self.attn_implementation
            if attn_impl not in FLASH_ATTENTION_IMPLS:
                raise NotImplementedError(
                    f'Derived ring attention only supports {"/".join(FLASH_ATTENTION_IMPLS)} backends, '
                    f'got {attn_impl!r}.')
        logger.info_once(f'[SequenceParallel] enabled: world_size={self.world_size} '
                         f'(ulysses={self.sp_world_size} x ring={self.rp_world_size}), '
                         f'attn_implementation={getattr(root_model.config, "_attn_implementation", None)!r}')

    def _mask_qkv(self, query, key, value, mask):
        mask = mask.unsqueeze(2).unsqueeze(3)
        query = query * mask
        value = value * mask
        key = key + ((~mask) * -1e5).to(key.dtype)
        return query, key, value

    def pad(self, tensor, padding_value, position_ids=None, dim=1):
        """Pad tensor for sequence parallel."""
        if tensor is None:
            return None
        if self.rp_world_size and self.rp_world_size > 1:
            world_size = self.world_size * 2
        else:
            world_size = self.world_size

        dim = dim if dim >= 0 else tensor.dim() + dim

        def _do_pad(tensor):
            length = tensor.shape[dim]
            pad_num = world_size - (length % world_size)
            if pad_num == 0 or pad_num == world_size:
                return tensor
            if not isinstance(padding_value, torch.Tensor):
                pad_shape = (*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1:])
                pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
                return torch.cat([tensor, pad], dim=dim)
            pad = padding_value.unsqueeze(0).repeat(tensor.shape[0], pad_num, 1)
            return torch.cat([tensor, pad], dim=dim)

        if position_ids is not None and self.rp_world_size > 1:
            cu_seqlens = get_cu_seqlens_from_position_ids(position_ids)
            padded = []
            for i in range(len(cu_seqlens) - 1):
                start, end = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
                slices = [slice(None)] * tensor.dim()
                slices[dim] = slice(start, end)
                padded.append(_do_pad(tensor[tuple(slices)]))
            return torch.cat(padded, dim=dim)
        return _do_pad(tensor)

    def pad_and_split_mm_tokens(self, visual_mask, mm_embeds):
        input_ids = self.extra_kwargs['input_ids']
        empty_embeds = torch.empty(
            (input_ids.shape[0], input_ids.shape[1], mm_embeds.shape[-1])).to(mm_embeds.device).to(mm_embeds.dtype)
        empty_embeds[visual_mask] = mm_embeds

        embeds = SimpleNamespace(weight=mm_embeds)

        _, split_input_embeds, _, _, _, _, extra_values = self.pad_and_split_inputs(
            None,
            empty_embeds,
            None,
            None,
            None,
            None,
            embeds,
            self.real_position_ids,
            extra_split_values=[(visual_mask, 0, -1)])
        visual_mask = extra_values[0]
        return visual_mask, split_input_embeds[visual_mask]

    def gather(self, local_output, dim: int, position_ids=None):
        """Gather tensor for sequence parallel - reverse of split."""
        if self.world_size == 1:
            return local_output

        dim = dim if dim >= 0 else local_output.dim() + dim

        def _slice(value, start, end):
            slices = [slice(None)] * value.dim()
            slices[dim] = slice(start, end)
            return value[tuple(slices)]

        def _assign(dst, start, end, src):
            slices = [slice(None)] * dst.dim()
            slices[dim] = slice(start, end)
            dst[tuple(slices)] = src

        if self.rp_world_size > 1:
            if position_ids is None:
                raise ValueError('position_ids are required to gather derived ring outputs.')
            position_ids = self.pad(position_ids, padding_value=-1, position_ids=position_ids)

            if self.sp_world_size > 1:
                gathered_sp = [torch.zeros_like(local_output) for _ in range(self.sp_world_size)]
                dist.all_gather(gathered_sp, local_output.contiguous(), group=self._sp_group)
                rp_chunk = torch.cat(gathered_sp, dim=dim)
            else:
                rp_chunk = local_output.contiguous()

            gathered_rp = [torch.zeros_like(rp_chunk) for _ in range(self.rp_world_size)]
            dist.all_gather(gathered_rp, rp_chunk, group=self._rp_group)

            cu_seqlens = get_cu_seqlens_from_position_ids(position_ids)
            padded_lengths = []
            for i in range(len(cu_seqlens) - 1):
                length = int((cu_seqlens[i + 1] - cu_seqlens[i]).item())
                padded_length = math.ceil(length / (self.world_size * 2)) * (self.world_size * 2)
                padded_lengths.append(padded_length)

            full_shape = list(rp_chunk.shape)
            full_shape[dim] = sum(padded_lengths)
            full_output = torch.zeros(full_shape, dtype=local_output.dtype, device=local_output.device)
            for idx_rp, rp_tensor in enumerate(gathered_rp):
                accumulated_local_length = 0
                for padded_length in padded_lengths:
                    local_length = padded_length // self.rp_world_size
                    local_tensor = _slice(rp_tensor, accumulated_local_length, accumulated_local_length + local_length)
                    chunk_size = local_length // 2
                    full_start = accumulated_local_length * self.rp_world_size + idx_rp * chunk_size
                    _assign(full_output, full_start, full_start + chunk_size, _slice(local_tensor, 0, chunk_size))
                    full_start = accumulated_local_length * self.rp_world_size + (2 * self.rp_world_size - idx_rp
                                                                                  - 1) * chunk_size
                    _assign(
                        full_output,
                        full_start,
                        full_start + chunk_size,
                        _slice(local_tensor, chunk_size, local_length),
                    )
                    accumulated_local_length += local_length
            return full_output.contiguous()

        if self.sp_world_size > 1:
            if is_hccl_backend(self._sp_group):
                gathered_sp_chunks = [torch.zeros_like(local_output) for _ in range(self.sp_world_size)]
                dist.all_gather(gathered_sp_chunks, local_output.contiguous(), group=self._sp_group)
                gathered_sp = torch.cat(gathered_sp_chunks, dim=dim)
            else:
                gathered_sp = torch.empty(
                    [local_output.shape[0] * self.sp_world_size] + list(local_output.shape[1:]),
                    dtype=local_output.dtype,
                    device=local_output.device)
                dist.all_gather_into_tensor(gathered_sp, local_output, group=self._sp_group)
                gathered_sp = torch.cat(gathered_sp.split(local_output.shape[0], dim=0), dim=dim)
            return gathered_sp.contiguous()
        return local_output

    def _split_packed(self, value, cu_seqlens, dim=1):
        dim = dim if dim >= 0 else value.dim() + dim
        local_values = []
        for i in range(len(cu_seqlens) - 1):
            start, end = int(cu_seqlens[i].item()), int(cu_seqlens[i + 1].item())
            slices = [slice(None)] * value.dim()
            slices[dim] = slice(start, end)
            sub_value = value[tuple(slices)]
            local_value = sub_value.chunk(2 * self.rp_world_size, dim=dim)
            local_values.extend([
                local_value[self.rp_rank],
                local_value[2 * self.rp_world_size - 1 - self.rp_rank],
            ])
        return torch.cat(local_values, dim=dim).contiguous()

    def split(self, input, dim: int, position_ids=None):
        """Split tensor for sequence parallel."""
        if self.world_size == 1:
            return input

        dim = dim if dim >= 0 else input.dim() + dim
        if self.rp_world_size > 1:
            if position_ids is None:
                raise ValueError('position_ids are required to split derived ring inputs.')
            cu_seqlens = get_cu_seqlens_from_position_ids(position_ids)
            if not torch.all(cu_seqlens % (2 * self.rp_world_size) == 0):
                raise ValueError(
                    f'Each packed sequence length must be divisible by {2 * self.rp_world_size} after padding.')
            value_chunks = self._split_packed(input, cu_seqlens, dim=dim)
            if self.sp_world_size > 1:
                return value_chunks.chunk(self.sp_world_size, dim=dim)[self.sp_rank].contiguous()
            return value_chunks.contiguous()

        dim_size = input.size(dim)
        assert dim_size % self.sp_world_size == 0, (f'The dimension to split ({dim_size}) is not a multiple of '
                                                    f'world size ({self.sp_world_size}), cannot split tensor evenly')
        tensor_list = torch.split(input, dim_size // self.sp_world_size, dim=dim)
        return tensor_list[self.sp_rank].contiguous()

    def pad_and_split_inputs(self,
                             input_ids,
                             input_embeds,
                             labels,
                             position_ids,
                             attention_mask,
                             loss_scale,
                             embed_tokens=None,
                             real_position_ids=None,
                             cache_position=None,
                             extra_split_values=None):
        """Common implementation for padding and splitting inputs

        Pad to a length divisible by the sequence-parallel size, then split across SP ranks.

        Args:
            input_ids: input_ids
            input_embeds: input_embeds
            labels: labels
            position_ids: position_ids or, position_ids for mrope
            attention_mask: attention_mask
            loss_scale: loss_scale
            embed_tokens: embed_tokens
            real_position_ids: the real position_ids to represent the seq length information
            extra_split_values: List of Tuples for extra split values, e.g.: (tensor, pad_value, split_dim)
        """
        tokenizer = self.tokenizer
        real_position_ids = real_position_ids if real_position_ids is not None else position_ids
        extra_values = []
        batch_size = input_ids.shape[
            0] if input_ids is not None else input_embeds.shape[0] if input_embeds is not None else None
        if real_position_ids is not None and batch_size is not None and real_position_ids.shape[0] == batch_size:
            # TODO clone everytime, but the position_ids is a small tensor
            self.extra_kwargs['position_ids'] = real_position_ids.clone()
        if input_ids is not None:
            input_ids = self.pad(input_ids, padding_value=tokenizer.pad_token_id, position_ids=real_position_ids)
            self.extra_kwargs['input_ids'] = input_ids.clone()
        if input_embeds is not None:
            pad_emb = input_embeds.new_zeros((1, input_embeds.shape[-1]))
            input_embeds = self.pad(input_embeds, padding_value=pad_emb, position_ids=real_position_ids)
        batch_size = input_ids.shape[
            0] if input_ids is not None else input_embeds.shape[0] if input_embeds is not None else 1
        if self.rp_world_size > 1 and batch_size > 1:
            raise NotImplementedError(
                'Derived ring attention only supports padding-free / packed batches with batch_size == 1.')
        if position_ids is not None:
            position_ids = self.pad(position_ids, padding_value=-1, position_ids=real_position_ids, dim=-1)
        if labels is not None:
            labels = self.pad(labels, padding_value=-100, position_ids=real_position_ids)
        if loss_scale is not None:
            loss_scale = self.pad(loss_scale, padding_value=0., position_ids=real_position_ids)
        if real_position_ids is not None:
            real_position_ids = self.pad(real_position_ids, padding_value=-1, position_ids=real_position_ids)
        # Build a 2D attention_mask whenever we padded for SP alignment so FlashAttention2 can unpad correctly.
        # For packed batches (batch_size==1 with multiple position_id resets), relying on position_ids alone is
        # unsafe if we also appended SP-alignment padding (position_ids=-1), because HF's FA2 varlen path will
        # include the padded tail in the last segment when attention_mask is None.
        if (input_ids is not None or input_embeds is not None) and batch_size > 1:
            # not padding_free, so not ring-attention
            inputs = input_ids if input_ids is not None else input_embeds
            attn_shape = inputs.shape[1]  # The sequence length
            if attention_mask is None:
                # Mask out padded positions introduced by sequence-parallel padding.
                # `real_position_ids` is padded with `-1` (see above), so use it to build a valid-token mask.
                attention_mask = (real_position_ids != -1).to(dtype=torch.int64)
            # no need position_ids here, because padding_free does not need attention_mask,
            # so this is not ring-attention
            attention_mask = self.pad(attention_mask, padding_value=0)
            local_cache_position = torch.arange(0, attn_shape, device=inputs.device)
            # FlashAttention expects a 2D padding mask (or None). Converting it to a 4D causal mask here breaks
            # the later per-rank sequence split and changes the attention contract relative to the baseline path.
            if (cache_position is None and hasattr(self, 'causal_mask_func') and self.causal_mask_func is not None
                    and not is_flash_attention_implementation(self.attn_implementation)):
                attention_mask = self.causal_mask_func(attention_mask, inputs.to(self.model_dtype),
                                                       local_cache_position, None, None)
        if extra_split_values is not None:
            for (tensor, pad_value, split_dim) in extra_split_values:
                extra_values.append(
                    self.pad(tensor, padding_value=pad_value, position_ids=real_position_ids, dim=split_dim))
        if input_ids is not None:
            input_ids = self.split(input_ids, dim=1, position_ids=real_position_ids)
        if input_embeds is not None:
            input_embeds = self.split(input_embeds, dim=1, position_ids=real_position_ids)
        if labels is not None:
            labels = self.split(labels, dim=-1, position_ids=real_position_ids)
        if loss_scale is not None:
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1)
            loss_scale = self.split(loss_scale, dim=-1, position_ids=real_position_ids)

        if position_ids is not None:
            position_ids = self.split(position_ids, dim=-1, position_ids=real_position_ids)
        # if attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
        #     attention_mask = self.split(attention_mask, dim=1, position_ids=real_position_ids)
        if extra_split_values is not None:
            for i in range(len(extra_values)):
                extra_values[i] = self.split(
                    extra_values[i], dim=extra_split_values[i][2], position_ids=real_position_ids)
        return input_ids, input_embeds, labels, position_ids, attention_mask, loss_scale, extra_values

    def _init_device_mesh(self, device_mesh: Optional[DeviceMesh] = None):
        """Initialize process groups for sequence parallel."""
        if not isinstance(device_mesh, DeviceMesh):
            raise RuntimeError('SequenceParallel requires a twinkle DeviceMesh for initialization.')

        self.device_mesh = device_mesh
        self.dp_world_size = device_mesh.data_world_size or 1
        (self._sp_group, self._rp_group, self._data_rank_group, self._sp_rank,
         self._rp_rank) = _get_seq_groups_from_device_mesh(device_mesh, self.seq_world_size, self.sp_world_size,
                                                           self.rp_world_size)

    @staticmethod
    def _is_packed_position_ids(position_ids: Optional[torch.Tensor]) -> bool:
        """Heuristic: detect packed samples by multiple (0,1,...) resets in position_ids.

        PackingDataset packs multiple sequences into one row by resetting position_ids to 0/1/... at each boundary.
        """
        if position_ids is None or not torch.is_tensor(position_ids):
            return False
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        if position_ids.dim() != 2:
            return False
        # A batch may contain multiple packed samples; consider it "packed" if any row is packed.
        for i in range(position_ids.size(0)):
            row = position_ids[i]
            zero_count = int((row == 0).sum().item())
            one_count = int((row == 1).sum().item())
            if zero_count > 1 and one_count > 1:
                return True
        return False

    def prepare_inputs(self, inputs, token_labels=True):
        """Cache the global layout and shard the fixed token-level loss inputs once."""
        positions = self._extract_real_position_ids(inputs.get('position_ids'))
        if self.rp_world_size > 1 and (positions[:, 0] != 0).any():
            raise NotImplementedError(
                'Ring SP requires sequences beginning at position zero; left padding is unsupported')
        padding_free = bool(inputs.pop('padding_free', False)) or self._is_packed_position_ids(positions)
        if padding_free and not self._packed_linear_supported:
            raise NotImplementedError('Packed Qwen3.5 SP requires FLA or a supported native GDN kernel')
        if padding_free and not is_flash_attention_implementation(self.attn_implementation):
            raise NotImplementedError('Packed SP inputs require FlashAttention 2 or 3')
        self.extra_kwargs = {'position_ids': positions.clone(), 'padding_free': padding_free}
        if inputs.get('input_ids') is not None:
            self.extra_kwargs['input_ids'] = inputs['input_ids'].clone()
        padded_positions = self.pad(positions, padding_value=-1, position_ids=positions)
        sharded = []
        for name, pad_value in (('labels', -100), ('completion_mask', 0), ('loss_scale', 0.0)):
            value = inputs.get(name)
            if value is None or not token_labels:
                continue
            if value.dim() != 2 or value.shape[-1] != positions.shape[-1]:
                raise ValueError(f'SP {name} must match the token layout, got {tuple(value.shape)}')
            value = self.pad(value, padding_value=pad_value, position_ids=positions)
            inputs[name] = self.split(value, dim=-1, position_ids=padded_positions)
            sharded.append(name)
        indices = torch.arange(positions.shape[-1], device=positions.device).expand_as(positions)
        indices = self.pad(indices, padding_value=-1, position_ids=positions)
        indices = self.split(indices, dim=-1, position_ids=padded_positions)
        valid_positions = positions
        mask = inputs.get('attention_mask')
        if torch.is_tensor(mask) and mask.dim() == 2:
            valid_positions = positions.masked_fill(~mask.bool(), -1)
        inputs['_sequence_layout'] = {'position_ids': valid_positions, 'indices': indices, 'sharded': sharded}
        return inputs


@dataclass(frozen=True)
class SequenceParallelConfig:
    enabled: bool = True
    ulysses_size: Optional[int] = None


class SequenceParallelStrategy:
    """Ulysses sequence-parallel strategy implementation."""

    def __init__(
        self,
        device_mesh=None,
        sp_config: Optional[Union[Dict[str, Any], SequenceParallelConfig]] = None,
        model: Optional[torch.nn.Module] = None,
        tokenizer_id: Optional[str] = None,
    ):
        self.device_mesh = device_mesh
        if isinstance(sp_config, SequenceParallelConfig):
            self.sp_config = asdict(sp_config)
        elif sp_config is not None and is_dataclass(sp_config):
            self.sp_config = asdict(sp_config)
        else:
            self.sp_config = sp_config or {}
        self.enabled = bool(self.sp_config.get('enabled', True))
        self.ulysses_size = _get_ulysses_size(device_mesh, self.sp_config)
        self._model_ref = model
        self._tokenizer_id = tokenizer_id
        self._tokenizer = None
        self._initialized = False
        self._ctx = SequenceParallel()

    def _get_tokenizer(self) -> Optional[PreTrainedTokenizer]:
        if self._tokenizer is not None:
            return self._tokenizer
        if not self._tokenizer_id:
            return None
        try:
            from twinkle.template import Template

            self._tokenizer = Template(self._tokenizer_id).tokenizer
            return self._tokenizer
        except Exception:
            return None

    def initialize(self) -> bool:
        if not self.enabled or self.ulysses_size <= 1:
            return False
        if self._initialized:
            return True
        if not dist.is_initialized():
            raise RuntimeError('torch.distributed must be initialized before enabling sequence parallel.')
        if not isinstance(self.device_mesh, DeviceMesh):
            raise RuntimeError('SequenceParallelStrategy requires a twinkle DeviceMesh when ulysses_size > 1.')
        if self._model_ref is None:
            raise RuntimeError('SequenceParallelStrategy requires a model reference to initialize.')
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            raise RuntimeError('SequenceParallelStrategy requires a tokenizer to initialize.')
        self._ctx.prepare(
            self.ulysses_size,
            self._model_ref,
            tokenizer,
            device_mesh=self.device_mesh,
        )
        self._initialized = True
        return True

    def prepare_model_inputs(self, inputs, token_labels=True, padding_free=False):
        if not self.enabled or self.ulysses_size <= 1:
            return inputs
        inputs = dict(inputs)
        if inputs.get('past_key_values') is not None or inputs.get('use_cache') is True:
            raise NotImplementedError('SP supports training/prefill without KV cache only')
        ids = inputs.get('input_ids')
        embeds = inputs.get('inputs_embeds')
        tensor = ids if ids is not None else embeds
        infer_model_positions = inputs.get('position_ids') is None and self._ctx._defer_inputs
        if inputs.get('position_ids') is None:
            if tensor is None:
                raise ValueError('SP requires input_ids, inputs_embeds or position_ids')
            positions = torch.arange(tensor.shape[1], device=tensor.device).expand(tensor.shape[:2])
            mask = inputs.get('attention_mask')
            inputs['position_ids'] = (positions.masked_fill(~mask.bool(), -1)
                                      if torch.is_tensor(mask) and mask.dim() == 2 else positions)
        inputs['padding_free'] = bool(inputs.get('padding_free', False) or padding_free)
        self._ctx.prepare_inputs(inputs, token_labels=token_labels)
        if not self._ctx._defer_inputs:
            inputs = self._ctx._split_model_inputs(inputs)
        elif infer_model_positions:
            inputs.pop('position_ids')  # Let HF compute multimodal RoPE before the text boundary.
        return inputs

    def context(self):
        if not self.enabled or self.ulysses_size <= 1:
            return nullcontext()
        if not self._initialized:
            raise RuntimeError('Initialize sequence parallel before entering the model context')
        return self._ctx.context()

    def _trim_gathered_sequence_padding(self, tensor: torch.Tensor, real_position_ids: torch.Tensor) -> torch.Tensor:
        if real_position_ids is None or not torch.is_tensor(real_position_ids) or real_position_ids.dim() < 2:
            return tensor
        if self._ctx.rp_world_size > 1:
            cu_seqlens = get_cu_seqlens_from_position_ids(real_position_ids)
            pieces = []
            padded_offset = 0
            divisor = self._ctx.world_size * 2
            for i in range(len(cu_seqlens) - 1):
                real_len = int((cu_seqlens[i + 1] - cu_seqlens[i]).item())
                padded_len = math.ceil(real_len / divisor) * divisor
                pieces.append(tensor[:, padded_offset:padded_offset + real_len])
                padded_offset += padded_len
            return torch.cat(pieces, dim=1).contiguous() if pieces else tensor[:, :0].contiguous()
        return tensor[:, :real_position_ids.shape[-1]].contiguous()

    def gather_loss_tensors(self, inputs, outputs):
        """Restore complete sequence tensors; values do not require labels or logps."""
        if inputs is None or outputs is None or not self.enabled or self.ulysses_size <= 1:
            return inputs, outputs
        inputs, outputs = copy(inputs), copy(outputs)
        positions = self._ctx.real_position_ids
        for name in ('logps', 'entropies', 'values', 'logits'):
            value = outputs.get(name)
            if value is None:
                continue
            if not torch.is_tensor(value) or value.dim() < 2:
                raise TypeError(f'SP output {name} must be a batched sequence tensor')
            value, _ = GatherLoss.apply(self._ctx, value, None, 1, positions)
            outputs[name] = self._trim_gathered_sequence_padding(value, positions)
        layout = inputs.pop('_sequence_layout', None)
        if layout is None and any(inputs.get(name) is not None for name in ('labels', 'completion_mask', 'loss_scale')):
            raise ValueError('SP loss inputs must come from prepare_model_inputs')
        for name in (layout or {}).get('sharded', ()):
            value = self._ctx.gather(inputs[name], dim=1, position_ids=positions)
            inputs[name] = self._trim_gathered_sequence_padding(value, positions)
        inputs['position_ids'] = positions
        return inputs, outputs

    def all_reduce(self, tensor, op=dist.ReduceOp.SUM):
        from torch.distributed.nn.functional import all_reduce
        for group, size in ((self._ctx._sp_group, self._ctx.sp_world_size),
                            (self._ctx._rp_group, self._ctx.rp_world_size)):
            if size and size > 1:
                tensor = all_reduce(tensor, op=op, group=group)
        return tensor

    def wrap_model(self, model, optimizer=None):
        self.initialize()
        return model, optimizer

    def unwrap_model(self, model):
        return model

    def needs_wrapped_optimizer_state(self) -> bool:
        return False

    def save_optimizer_checkpoint(self, model, optimizer, output_path: str, *, param_name_mapping=None):
        from twinkle.utils.platforms import Platform
        from twinkle.model.state_dict_utils import remap_optimizer_state_names
        if Platform.is_master():
            optim_state = optimizer.state_dict()
            remap_optimizer_state_names(optim_state, param_name_mapping or {})
            torch.save(optim_state, output_path)

    def load_optimizer_checkpoint(self, model, optimizer, input_path: str, *, param_name_mapping=None):
        from twinkle.model.state_dict_utils import remap_optimizer_state_names
        # weights_only=True: optimizer state is tensors + plain containers (verified), so a
        # hostile checkpoint's pickle payload never reaches an unrestricted unpickler.
        optim_state = torch.load(input_path, map_location='cpu', weights_only=True)
        remap_optimizer_state_names(optim_state, param_name_mapping or {})
        optimizer.load_state_dict(optim_state)
