# Copyright (c) ModelScope Contributors. All rights reserved.
import math
import torch
import torch.distributed as dist
from copy import copy
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
from transformers import PreTrainedTokenizer
from types import MethodType, SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

from twinkle.patch import apply_patch
from twinkle.utils import DeviceMesh
from twinkle.utils.transformers_utils import get_llm_model
from .linear_attention_sp import Qwen3_5GatedDeltaNetUlyssesPatch
from .utils import (DistributedAttention, GatherLoss, _derive_sequence_parallel_sizes, _get_seq_groups_from_device_mesh,
                    _get_ulysses_size, _SeqAllToAll, get_config_attr, get_cu_seqlens_from_position_ids, is_hccl_backend,
                    is_moe_config, post_all2all)


def is_qwen3_vl(model):
    mt = getattr(getattr(model, 'config', None), 'model_type', '')
    return 'qwen3_vl' in mt


def is_qwen3_omni(model):
    mt = getattr(getattr(model, 'config', None), 'model_type', '')
    return 'qwen3_omni' in mt


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

    def _prepare_flash_attn(self, base_model: torch.nn.Module):
        try:
            from transformers import masking_utils

            _origin_flash_attention_mask = masking_utils.flash_attention_mask

            # Patch attention masks for SP: avoid masking when full sequence is reconstructed.
            def flash_attention_mask(batch_size,
                                     cache_position,
                                     kv_length,
                                     kv_offset=0,
                                     mask_function=masking_utils.causal_mask_function,
                                     attention_mask=None,
                                     **kwargs):
                if self.world_size == 1:
                    return _origin_flash_attention_mask(batch_size, cache_position, kv_length, kv_offset, mask_function,
                                                        attention_mask, **kwargs)
                if attention_mask is not None:
                    if attention_mask.all():
                        attention_mask = None

                return attention_mask

            masking_utils.flash_attention_mask = flash_attention_mask
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['flash_attention_2'] = flash_attention_mask

            def sdpa_mask(batch_size, cache_position, kv_length, *args, **kwargs):
                if self.world_size == 1:
                    return masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa_origin'](batch_size,
                                                                                                     cache_position,
                                                                                                     kv_length, *args,
                                                                                                     **kwargs)
                device = cache_position.device
                cache_position = self.real_position_ids[0]
                cache_position = self.pad(cache_position, padding_value=-1, position_ids=self.real_position_ids, dim=0)
                cache_position = torch.arange(0, cache_position.shape[0], device=device)
                kv_length = cache_position.shape[0]
                return masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa_origin'](batch_size,
                                                                                                 cache_position,
                                                                                                 kv_length, *args,
                                                                                                 **kwargs)

            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping[
                'sdpa_origin'] = masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa']
            masking_utils.ALL_MASK_ATTENTION_FUNCTIONS._global_mapping['sdpa'] = sdpa_mask

            def create_causal_mask(config, input_embeds, attention_mask, cache_position, *args, **kwargs):
                if self.world_size == 1:
                    return masking_utils.origin_create_causal_mask(config, input_embeds, attention_mask, cache_position,
                                                                   *args, **kwargs)
                input_embeds = torch.ones(
                    (input_embeds.shape[0], input_embeds.shape[1] * self.sp_world_size, input_embeds.shape[2]),
                    dtype=input_embeds.dtype,
                    device=input_embeds.device)
                cache_position = torch.arange(0, input_embeds.shape[1], device=input_embeds.device)
                return masking_utils.origin_create_causal_mask(config, input_embeds, attention_mask, cache_position,
                                                               *args, **kwargs)

            masking_utils.origin_create_causal_mask = masking_utils.create_causal_mask
            masking_utils.create_causal_mask = create_causal_mask
        except ImportError:
            pass

        if hasattr(base_model, 'language_model'):
            text_model = base_model.language_model
        else:
            text_model = base_model

        from transformers.modeling_flash_attention_utils import is_flash_attn_available
        if is_flash_attn_available():
            # TODO this works for multi-modal models like qwen2.5-vl
            # SDPA is not supported here, because we need to copy the code to our project, which will bring
            # more work for maintaining.
            from transformers import modeling_flash_attention_utils
            from transformers.modeling_flash_attention_utils import _flash_attention_forward
            _distributed_flash_attention = DistributedAttention(_flash_attention_forward, self)

            modeling_flash_attention_utils._flash_attention_forward_origin = _flash_attention_forward

            def flash_attention_forward(query_states: torch.Tensor, key_states: torch.Tensor,
                                        value_states: torch.Tensor, attention_mask: Optional[torch.Tensor], q_len,
                                        *args, **kwargs):
                if self.world_size == 1:
                    return _flash_attention_forward(query_states, key_states, value_states, attention_mask, q_len,
                                                    *args, **kwargs)
                return _distributed_flash_attention(query_states, key_states, value_states, attention_mask,
                                                    q_len * self.sp_world_size, *args, **kwargs)

            modeling_flash_attention_utils._flash_attention_forward = flash_attention_forward

        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        def local_flash_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                             dist_attn, **kwargs):
            if self.world_size == 1 or module.__class__ not in [m.__class__ for m in text_model.modules()]:
                return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query_states, key_states,
                                                                           value_states, attention_mask, *args,
                                                                           **kwargs)
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
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
                            causal=module.is_causal,
                            window_size=kwargs.get('sliding_window') or (-1, -1),
                            group=self._rp_group,
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
                    return ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'](module, query, key, value, *args,
                                                                               **kwargs)[0]

                dist_attn.local_attn = _attention

            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        def local_sdpa_attn(module: torch.nn.Module, query_states, key_states, value_states, attention_mask, *args,
                            dist_attn, **kwargs):
            # Bypass SP logic when world_size == 1 (SP disabled) or module not in text_model
            if self.world_size == 1 or module.__class__ not in [m.__class__ for m in text_model.modules()]:
                return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query_states, key_states, value_states,
                                                              attention_mask, *args, **kwargs)
            # Policy: packed (PackingDataset/padding-free) batches require FlashAttention2 varlen/packed semantics.
            # SDPA does not have a native packed/varlen interface; supporting packed batches would require building a
            # large block-diagonal causal mask (slow / memory heavy).
            if self.extra_kwargs.get('padding_free', False):
                raise RuntimeError(
                    'SequenceParallel: detected padding_free/packed batch. '
                    'SDPA backend is not supported for padding_free/packed batches; please use flash_attention_2.')
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
                    if self.rp_world_size > 1:
                        raise NotImplementedError('SDPA does not support derived ring attention.')
                    return ALL_ATTENTION_FUNCTIONS['sdpa_origin'](module, query, key, value, *args, **kwargs)[0]

                dist_attn.local_attn = _attention
            return dist_attn(
                query_states.transpose(1, 2), key_states.transpose(1, 2), value_states.transpose(1, 2), attention_mask,
                *args, **kwargs), None

        ALL_ATTENTION_FUNCTIONS['flash_attention_2_origin'] = ALL_ATTENTION_FUNCTIONS['flash_attention_2']
        ALL_ATTENTION_FUNCTIONS['sdpa_origin'] = ALL_ATTENTION_FUNCTIONS['sdpa']
        ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = partial(
            local_flash_attn, dist_attn=DistributedAttention(None, self))
        ALL_ATTENTION_FUNCTIONS['sdpa'] = partial(local_sdpa_attn, dist_attn=DistributedAttention(None, self))

    def _prepare_forward_hook(self, base_model: torch.nn.Module):

        def pre_forward_split_hook(_self, args, kwargs):
            if self.world_size == 1:
                return args, kwargs
            # Pad to multiple of SP size and split inputs per SP rank before forward.
            input_ids = kwargs.get('input_ids', None)
            inputs_embeds = kwargs.get('inputs_embeds', None)
            position_ids = kwargs['position_ids']
            real_position_ids = self._extract_real_position_ids(position_ids)
            attention_mask = kwargs.get('attention_mask', None)
            cache_position = kwargs.get('cache_position', None)
            if hasattr(_self, 'language_model'):
                embed_tokens = getattr(_self.language_model, 'embed_tokens', None)
            else:
                embed_tokens = getattr(_self, 'embed_tokens', None)
            input_ids, inputs_embeds, _, position_ids, attention_mask, _, _ = self.pad_and_split_inputs(
                input_ids,
                inputs_embeds,
                None,
                position_ids,
                attention_mask,
                None,
                embed_tokens=embed_tokens,
                real_position_ids=real_position_ids,
                cache_position=cache_position)
            kwargs['input_ids'] = input_ids
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask
            return args, kwargs

        base_model.register_forward_pre_hook(pre_forward_split_hook, with_kwargs=True)

    def _prepare_multimodal_deepstack(self, base_model: torch.nn.Module):
        if not is_qwen3_vl(base_model):
            return

        def _patch_deepstack_process(module: torch.nn.Module) -> bool:
            origin = getattr(module, '_deepstack_process', None)
            if not callable(origin):
                return False
            if getattr(module, '_twinkle_sp_mm_patched', False):
                return False

            def _deepstack_process(_self, hidden_states: torch.Tensor, visual_pos_masks: torch.Tensor,
                                   visual_embeds: torch.Tensor):
                world_size = sequence_parallel.world_size
                if world_size and world_size > 1 and visual_pos_masks is not None:
                    visual_pos_masks, visual_embeds = sequence_parallel.pad_and_split_mm_tokens(
                        visual_pos_masks, visual_embeds)
                if visual_pos_masks is None:
                    return hidden_states + visual_embeds.mean() * 0
                visual_pos_masks = visual_pos_masks.to(hidden_states.device)
                visual_embeds = visual_embeds.to(hidden_states.device, hidden_states.dtype)
                if hidden_states.ndim == 3 and visual_pos_masks.ndim == 3:
                    visual_pos_masks = visual_pos_masks[..., 0]
                local_this = hidden_states[visual_pos_masks, :].clone() + visual_embeds
                hidden_states[visual_pos_masks, :] = local_this
                return hidden_states

            module._deepstack_process = MethodType(_deepstack_process, module)
            module._twinkle_sp_mm_patched = True
            return True

        for submodule in base_model.modules():
            _patch_deepstack_process(submodule)
        _patch_deepstack_process(base_model)

    @staticmethod
    def _is_qwen35_model(model: torch.nn.Module) -> bool:
        config = getattr(model, 'config', None)
        model_type = str(getattr(config, 'model_type', '') or '')
        if model_type == 'qwen3_5':
            return True

        architectures = getattr(config, 'architectures', None) or []
        if any('Qwen3_5' in str(arch) for arch in architectures):
            return True

        model_module = getattr(model.__class__, '__module__', '') or ''
        return 'transformers.models.qwen3_5' in model_module

    def _prepare_qwen35_linear_attention(self, model: torch.nn.Module):
        has_qwen35_linear_attention = self._is_qwen35_model(model)
        if not has_qwen35_linear_attention:
            try:
                from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
            except Exception:
                return
            has_qwen35_linear_attention = any(isinstance(module, Qwen3_5GatedDeltaNet) for module in model.modules())
        if not has_qwen35_linear_attention:
            return
        if int(self.rp_world_size or 1) > 1:
            raise NotImplementedError(
                'SequenceParallel: Qwen3.5 linear attention sequence parallel does not support rp_world_size > 1 '
                '(derived ring attention).')
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
                _bs_logits, _ = GatherLoss.apply(_bs_logits, None, 1, self.real_position_ids)
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
        llm_model = get_llm_model(model)
        config_candidates = [getattr(model, 'config', None)]
        llm_config = getattr(llm_model, 'config', None)
        if llm_config is not None and llm_config not in config_candidates:
            config_candidates.append(llm_config)
        text_config = getattr(getattr(model, 'config', None), 'text_config', None)
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
        assert self.num_heads is not None, 'Cannot find num_attention_heads/num_key_value_heads in model config'
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

        self._init_device_mesh(device_mesh)
        if not SequenceParallel._global_inited:
            # these operations are global initializations and patches
            self._prepare_flash_attn(llm_model)
            SequenceParallel._global_inited = True
        self._prepare_qwen35_linear_attention(llm_model)

        self._prepare_forward_hook(llm_model)
        self._prepare_multimodal_deepstack(llm_model)

        if is_moe_config(getattr(model, 'config', None)):
            self._prepare_moe_aux_loss(llm_model)

        self.model_dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer
        if self.rp_world_size > 1:
            attn_impl = getattr(model.config, '_attn_implementation', None)
            if attn_impl != 'flash_attention_2':
                raise NotImplementedError('Derived ring attention only supports flash_attention_2 backend.')

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
            pad_emb = torch.zeros(
                (1, embed_tokens.weight.shape[-1])).to(embed_tokens.weight.device).to(embed_tokens.weight.dtype)
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
            # FlashAttention2 expects a 2D padding mask (or None). Converting it to a 4D causal mask here breaks
            # the later per-rank sequence split and changes the attention contract relative to the baseline path.
            if (cache_position is None and hasattr(self, 'causal_mask_func') and self.causal_mask_func is not None
                    and self.attn_implementation != 'flash_attention_2'):
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

    def prepare_inputs(self, inputs):
        """Prepare inputs

        1. set extra_kwargs['position_ids']
        2. split labels
        """
        input_ids = inputs.get('input_ids')
        position_ids = inputs.get('position_ids')
        padding_free = bool(inputs.pop('padding_free', False))
        if padding_free and self.attn_implementation not in ('flash_attention_2', 'flash_attention_3'):
            raise RuntimeError('Transformers SequenceParallel does not support padding_free/packed inputs with '
                               f'attn_implementation={self.attn_implementation!r}. '
                               'Use flash_attention_2 or flash_attention_3, or disable padding_free/packing. '
                               'SDPA/eager attention cannot safely preserve packed sequence boundaries in this path.')
        real_position_ids = self._extract_real_position_ids(position_ids)
        if real_position_ids is not None and input_ids is not None and real_position_ids.shape[0] == input_ids.shape[0]:
            self.extra_kwargs['position_ids'] = real_position_ids.clone()
        self.extra_kwargs['padding_free'] = padding_free
        if input_ids is not None:
            self.extra_kwargs['input_ids'] = input_ids.clone()
        if 'labels' in inputs:
            labels = inputs.get('labels')
            _, _, labels, _, _, _, _ = self.pad_and_split_inputs(
                None,
                None,
                labels,
                None,
                None,
                None,
                real_position_ids=real_position_ids,
            )
            inputs['labels'] = labels
        return inputs


sequence_parallel = SequenceParallel()


@dataclass(frozen=True)
class SequenceParallelConfig:
    enabled: bool = True
    ulysses_size: Optional[int] = None
    gather_logits: bool = True


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
        if not dist.is_initialized():
            raise RuntimeError('torch.distributed must be initialized before enabling sequence parallel.')
        if not isinstance(self.device_mesh, DeviceMesh):
            raise RuntimeError('SequenceParallelStrategy requires a twinkle DeviceMesh when ulysses_size > 1.')
        if self._model_ref is None:
            raise RuntimeError('SequenceParallelStrategy requires a model reference to initialize.')
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            raise RuntimeError('SequenceParallelStrategy requires a tokenizer to initialize.')
        sequence_parallel.prepare(
            self.ulysses_size,
            self._model_ref,
            tokenizer,
            device_mesh=self.device_mesh,
        )
        self._initialized = True
        return True

    def preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.ulysses_size <= 1:
            return inputs
        return sequence_parallel.prepare_inputs(inputs)

    def postprocess_outputs(self, outputs: Any) -> Any:
        if (not self.enabled or self.ulysses_size <= 1 or not self.sp_config.get('gather_logits', True)):
            return outputs
        # Twinkle expects dict-like ModelOutput containers in the main training path
        # (uses `.get(...)` and `outputs[...] = ...`). Keep SP postprocess consistent.
        if outputs is None or not hasattr(outputs, 'get') or not hasattr(outputs, '__setitem__'):
            raise TypeError('SequenceParallelStrategy.postprocess_outputs expects a dict-like ModelOutput. '
                            f'Got type={type(outputs)}')
        logits = outputs.get('logits', None)
        if logits is None or not torch.is_tensor(logits) or logits.dim() < 2:
            return outputs
        gathered = sequence_parallel.gather(logits, dim=1, position_ids=sequence_parallel.real_position_ids)
        # Scheme A: SP pads to make seq_len divisible by sp_size. Trim back to the original
        # (unpadded) length using the cached real_position_ids.
        real_pos = sequence_parallel.real_position_ids
        if real_pos is not None and torch.is_tensor(real_pos) and real_pos.dim() >= 2:
            gathered = gathered[:, :real_pos.shape[1]].contiguous()
        outputs['logits'] = gathered
        return outputs

    @staticmethod
    def _trim_gathered_sequence_padding(tensor: torch.Tensor, real_position_ids: torch.Tensor) -> torch.Tensor:
        if real_position_ids is None or not torch.is_tensor(real_position_ids) or real_position_ids.dim() < 2:
            return tensor
        if sequence_parallel.rp_world_size > 1:
            cu_seqlens = get_cu_seqlens_from_position_ids(real_position_ids)
            pieces = []
            padded_offset = 0
            divisor = sequence_parallel.world_size * 2
            for i in range(len(cu_seqlens) - 1):
                real_len = int((cu_seqlens[i + 1] - cu_seqlens[i]).item())
                padded_len = math.ceil(real_len / divisor) * divisor
                pieces.append(tensor[:, padded_offset:padded_offset + real_len])
                padded_offset += padded_len
            return torch.cat(pieces, dim=1).contiguous() if pieces else tensor[:, :0].contiguous()
        return tensor[:, :real_position_ids.shape[-1]].contiguous()

    def gather_loss_tensors(
        self,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if inputs is None or outputs is None:
            return inputs, outputs
        if not self.enabled or self.ulysses_size <= 1:
            return inputs, outputs
        labels = inputs.get('labels')
        logps = outputs.get('logps')
        if labels is None or logps is None:
            return inputs, outputs
        if not torch.is_tensor(logps) or logps.dim() < 2:
            raise TypeError('SequenceParallelStrategy.gather_loss_inputs expects outputs[\"logps\"] to be a '
                            f'sequence tensor, got type={type(logps)} shape={getattr(logps, "shape", None)}')
        inputs = copy(inputs)
        outputs = copy(outputs)
        real_position_ids = sequence_parallel.real_position_ids
        gathered_logps, gathered_labels = GatherLoss.apply(logps, labels, 1, real_position_ids)
        gathered_logps = self._trim_gathered_sequence_padding(gathered_logps, real_position_ids)
        gathered_labels = self._trim_gathered_sequence_padding(gathered_labels, real_position_ids)
        outputs['logps'] = gathered_logps
        inputs['labels'] = gathered_labels
        return inputs, outputs

    def wrap_model(self, model, optimizer=None):
        self.initialize()
        return model, optimizer

    def unwrap_model(self, model):
        return model

    def needs_wrapped_optimizer_state(self) -> bool:
        return False

    def save_optimizer_checkpoint(self, model, optimizer, output_path: str):
        from twinkle.utils.platforms import Platform
        if Platform.is_master():
            torch.save(optimizer.state_dict(), output_path)

    def load_optimizer_checkpoint(self, model, optimizer, input_path: str):
        optimizer.load_state_dict(torch.load(input_path, map_location='cpu', weights_only=False))
