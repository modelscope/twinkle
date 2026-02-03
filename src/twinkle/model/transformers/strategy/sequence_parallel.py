# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from functools import partial
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributed as dist
from transformers import PreTrainedTokenizer
from dataclasses import asdict, dataclass, is_dataclass

from twinkle.utils import DeviceMesh


def get_llm_model(model):  # type: ignore
    return getattr(model, "language_model", model)


class HfConfigFactory:  # type: ignore
    @staticmethod
    def get_config_attr(config, attr_name: str, include_vit: bool = False):
        return getattr(config, attr_name, None)


def get_cu_seqlens_from_position_ids(position_ids: torch.LongTensor):
    position_ids = position_ids[0]
    seq_start_indices = torch.where(position_ids == 0)[0]
    seq_end_indices = torch.cat(
        [seq_start_indices[1:], torch.tensor([len(position_ids)], device=position_ids.device)]
    )
    seq_lengths = seq_end_indices - seq_start_indices
    cu_seqlens = torch.cumsum(
        torch.cat([torch.tensor([0], device=position_ids.device), seq_lengths]), dim=0
    )
    return cu_seqlens


def _get_raw_data_world_size(device_mesh: DeviceMesh) -> int:
    dp_world_size = device_mesh.dp_world_size or 1
    fsdp_world_size = device_mesh.fsdp_world_size or 1
    if dp_world_size <= 0:
        dp_world_size = 1
    if fsdp_world_size <= 0:
        fsdp_world_size = 1
    return dp_world_size * fsdp_world_size


def _get_raw_data_rank(device_mesh: DeviceMesh, rank: int) -> Optional[int]:
    coord = device_mesh._get_coord_for_rank(rank)
    if coord is None:
        return None

    dp_rank = None
    fsdp_rank = None
    if device_mesh.has_dim("dp"):
        dp_rank = coord[device_mesh._get_dim_index("dp")]
    if device_mesh.has_dim("fsdp"):
        fsdp_rank = coord[device_mesh._get_dim_index("fsdp")]

    fsdp_world_size = device_mesh.fsdp_world_size
    data_rank = dp_rank if dp_rank is not None else None
    if fsdp_world_size is not None and fsdp_world_size > 1:
        if dp_rank is not None and fsdp_rank is not None:
            data_rank = dp_rank * fsdp_world_size + fsdp_rank
        elif fsdp_rank is not None:
            data_rank = fsdp_rank

    if data_rank is None:
        data_rank = 0
    return int(data_rank)


def _get_sp_group_from_device_mesh(
    device_mesh: Optional[DeviceMesh],
    sp_size: int,
) -> Optional[dist.ProcessGroup]:
    """Return the SP (sequence-parallel) process group for the current rank.

    If the mesh defines an explicit "sp" dimension, use it directly. Otherwise,
    derive SP groups by chunking data-parallel ranks (dp/fsdp) while keeping
    all other mesh dimensions (tp/pp/ep/etc.) fixed.

    Example (no explicit "sp" dim, sp_size=2):
        mesh_dim_names = ("dp", "fsdp", "tp")
        mesh = np.arange(8).reshape(2, 2, 2)
        # coords are (dp, fsdp, tp). dp/fsdp are "data" dims; tp is "non-data".
        # raw_data_rank = dp * fsdp_world_size + fsdp, so ranges [0..3].
        # group_id = raw_data_rank // sp_size partitions data ranks into 2 groups.
        #
        # For tp=0:
        #   data ranks 0,1 -> group_id=0  => ranks at coords:
        #     (dp=0,fsdp=0,tp=0) -> rank 0
        #     (dp=0,fsdp=1,tp=0) -> rank 2
        #   data ranks 2,3 -> group_id=1  => ranks at coords:
        #     (dp=1,fsdp=0,tp=0) -> rank 4
        #     (dp=1,fsdp=1,tp=0) -> rank 6
        #
        # For tp=1:
        #   data ranks 0,1 -> group_id=0  => ranks at coords:
        #     (dp=0,fsdp=0,tp=1) -> rank 1
        #     (dp=0,fsdp=1,tp=1) -> rank 3
        #   data ranks 2,3 -> group_id=1  => ranks at coords:
        #     (dp=1,fsdp=0,tp=1) -> rank 5
        #     (dp=1,fsdp=1,tp=1) -> rank 7
        #
        # Final SP groups (keyed by (group_id, non_data_key)):
        #   (0, (tp=0)) -> [0, 2]
        #   (1, (tp=0)) -> [4, 6]
        #   (0, (tp=1)) -> [1, 3]
        #   (1, (tp=1)) -> [5, 7]
        #
        # Each SP group has size=2 and never crosses tp.
    """
    if device_mesh is None or sp_size <= 1:
        return None
    if device_mesh.has_dim("sp"):
        return device_mesh.create_process_group(["sp"])
    if not dist.is_available() or not dist.is_initialized():
        return None

    raw_data_world_size = _get_raw_data_world_size(device_mesh)
    if raw_data_world_size % sp_size != 0:
        raise ValueError(
            f"data_world_size ({raw_data_world_size}) must be divisible by sp_size ({sp_size})."
        )

    rank = dist.get_rank()
    ref_coord = device_mesh._get_coord_for_rank(rank)
    if ref_coord is None:
        return None

    non_data_indices = []
    if device_mesh.mesh_dim_names is not None:
        for i, name in enumerate(device_mesh.mesh_dim_names):
            if name in ("dp", "fsdp"):
                continue
            non_data_indices.append(i)

    # Group ranks by (data-parallel chunk, non-data mesh coordinates).
    groups: Dict[Tuple[int, Tuple[int, ...]], list[int]] = {}
    for r in device_mesh.mesh.flatten().tolist():
        r = int(r)
        coord = device_mesh._get_coord_for_rank(r)
        if coord is None:
            continue
        raw_rank = _get_raw_data_rank(device_mesh, r)
        if raw_rank is None:
            continue
        group_id = raw_rank // sp_size
        non_data_key = tuple(coord[i] for i in non_data_indices)
        key = (group_id, non_data_key)
        groups.setdefault(key, []).append(r)

    group_list = []
    for key, ranks in groups.items():
        ranks = sorted(ranks)
        if len(ranks) != sp_size:
            raise ValueError(
                f"SP group size mismatch for key={key}: expected {sp_size}, got {len(ranks)}"
            )
        group_list.append((key, ranks))

    group_list.sort(key=lambda item: item[0])

    sp_group = None
    for _, ranks in group_list:
        pg = dist.new_group(ranks=ranks)
        if rank in ranks:
            sp_group = pg
    return sp_group


class GatherLoss(torch.autograd.Function):
    """Gather loss from sequence group."""

    @staticmethod
    def forward(ctx, loss, labels, gather_idx=None, position_ids=None):
        """
        Args:
            loss: loss tensor after splitting
            labels: labels tensor after splitting
            gather_idx: gather the tensors on this dim
        """
        ctx.scatter_shape = loss.shape[gather_idx or 0]
        ctx.gather_idx = gather_idx or 0
        if position_ids is not None:
            position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        ctx.position_ids = position_ids
        # Gather split losses/labels to compute aux losses on full sequence length.
        output = sequence_parallel.gather(loss, dim=ctx.gather_idx, position_ids=position_ids)
        if labels is not None:
            labels_output = sequence_parallel.gather(labels, dim=ctx.gather_idx, position_ids=position_ids)
        else:
            labels_output = None
        return output, labels_output

    @staticmethod
    def backward(ctx, *grad_output):
        # Split grads back to local sequence chunk; scale for all-gather semantics.
        _grad = grad_output[0] * sequence_parallel.world_size
        if sequence_parallel.world_size > 1 and sequence_parallel._sp_group is not None:
            _grad = sequence_parallel.split(_grad, dim=ctx.gather_idx, position_ids=ctx.position_ids).contiguous()
        return _grad, None, None, None


# Code borrowed from deepspeed, here is why:
# 1. Reduce the dependency
# 2. The original code is complex
def _generate_layout_params(scatter_idx, seq_world_size, input):
    if scatter_idx < 2:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        pre_all2all_inp_shape = [bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]
        pre_all2all_permute_idx = (1, 0, 2, 3, 4)

        post_all2all_permute_idx = (1, 2, 0, 3, 4)
        post_all2all_res_shape = [bs, global_seq_len // seq_world_size, seq_world_size * num_local_head, head_dim]
    else:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert num_total_head % seq_world_size == 0, (f'Number of heads ({num_total_head}) must be divisible '
                                                      f'by the sequence parallel size ({seq_world_size})!')
        pre_all2all_inp_shape = [bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]
        pre_all2all_permute_idx = (2, 0, 1, 3, 4)

        post_all2all_permute_idx = (1, 0, 2, 3, 4)
        post_all2all_res_shape = [bs, seq_world_size * local_seq_len, num_total_head // seq_world_size, head_dim]

    return pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape


def post_all2all(permute_idx, res_shape):
    """
    Post-processing function for `all2all` communication.
    """

    def post_func(input):
        if permute_idx is not None:
            input = input.permute(permute_idx).contiguous()
        output = input.reshape(res_shape).contiguous()

        return output

    return post_func


def pre_all2all_fun(permute_idx, inp_shape, input):
    """
    Pre-processing function for `all2all` communication.
    """
    input_t = input.reshape(inp_shape).contiguous()
    if permute_idx is not None:
        input_t = input_t.permute(permute_idx).contiguous()
    return input_t


def single_all_to_all(input, scatter_idx, gather_idx, group, **kwargs):
    seq_world_size = dist.get_world_size(group)
    num_heads = input.shape[2]
    if num_heads % seq_world_size != 0 and not scatter_idx < 2:
        raise NotImplementedError(f'num_heads {num_heads} cannot be split by sp world size {seq_world_size}')
    pre_all2all_permute_idx, pre_all2all_inp_shape, post_all2all_permute_idx, post_all2all_res_shape = (
        _generate_layout_params(scatter_idx, seq_world_size, input))

    input_t = pre_all2all_fun(pre_all2all_permute_idx, pre_all2all_inp_shape, input)

    post_all2all_fun = post_all2all(post_all2all_permute_idx, post_all2all_res_shape)
    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    res = post_all2all_fun(output)
    return res


class _SeqAllToAll(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        res = single_all_to_all(input, scatter_idx, gather_idx, group)
        return res

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[None, torch.Tensor, None, None]:
        # Reverse scatter/gather in backward to match forward layout transform.
        return None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None


class DistributedAttention(torch.nn.Module):

    def __init__(
        self,
        local_attention,
        sequence_parallel,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention
        self.sequence_parallel = sequence_parallel
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor,
                *args: Any, **kwargs) -> torch.Tensor:
        if self.sequence_parallel.world_size == 1:
            return self.local_attn(query, key, value, attention_mask, *args, **kwargs)

        # All-to-all to assemble full sequence for attention, then split back after.
        if self.sequence_parallel.sp_world_size > 1:
            query_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, query, self.scatter_idx, self.gather_idx)
            key_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, key, self.scatter_idx, self.gather_idx)
            value_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, value, self.scatter_idx, self.gather_idx)
        else:
            query_layer, key_layer, value_layer = query, key, value

        position_ids = kwargs.pop('position_ids')
        if position_ids is not None:
            shape0 = position_ids.shape[0]
            position_ids_output = torch.empty(
                (shape0 * self.sequence_parallel.sp_world_size, position_ids.shape[1]),
                dtype=position_ids.dtype,
                device=position_ids.device)
            dist.all_gather_into_tensor(position_ids_output, position_ids, group=self.sequence_parallel._sp_group)
            position_ids = torch.cat(position_ids_output.split(shape0, dim=0), dim=1)

        context_layer = self.local_attn(
            query_layer, key_layer, value_layer, attention_mask, *args, position_ids=position_ids, **kwargs)

        if self.sequence_parallel.sp_world_size > 1:
            output = _SeqAllToAll.apply(self.sequence_parallel._sp_group, context_layer, self.gather_idx,
                                        self.scatter_idx)
        else:
            output = context_layer

        return output

# main content copied from ms-swift
class SequenceParallel:

    _global_inited: bool = False

    def __init__(self):
        self.sp_world_size = None
        self.dp_world_size = None
        self.world_size = None
        self.model_dtype = None
        self.tokenizer = None
        self.device_mesh = None
        self._sp_group = None
        self.num_heads = None
        self.causal_mask_func = None
        self.extra_kwargs = {}

    @property
    def real_position_ids(self) -> torch.Tensor:
        """The real position ids, this is different from the position_ids in mrope"""
        return self.extra_kwargs.get('position_ids')

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
                # Rebuild cache positions from real (full) position ids.
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
                    if 'cu_seq_lens_q' in kwargs:
                        position_ids = kwargs.get('position_ids')
                        if position_ids is None:
                            position_ids = self.real_position_ids
                        position_ids = self.pad(position_ids, padding_value=-1, position_ids=position_ids)
                        cu_seqlens = get_cu_seqlens_from_position_ids(position_ids).to(torch.int32)
                        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
                        assert query.shape[2] == cu_seqlens[-1]
                        kwargs['cu_seq_lens_q'] = cu_seqlens
                        kwargs['cu_seq_lens_k'] = cu_seqlens
                        kwargs['max_length_q'] = max_seqlen
                        kwargs['max_length_k'] = max_seqlen
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
            if dist_attn.local_attn is None:

                def _attention(query, key, value, *args, **kwargs):
                    query = query.transpose(1, 2)
                    key = key.transpose(1, 2)
                    value = value.transpose(1, 2)
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
            attention_mask = kwargs.get('attention_mask', None)
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
                real_position_ids=self.real_position_ids)
            kwargs['input_ids'] = input_ids
            kwargs['inputs_embeds'] = inputs_embeds
            kwargs['position_ids'] = position_ids
            kwargs['attention_mask'] = attention_mask
            return args, kwargs

        base_model.register_forward_pre_hook(pre_forward_split_hook, with_kwargs=True)

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

    @staticmethod
    def _is_moe_model(config) -> bool:
        if 'Moe' in config.__class__.__name__:
            return True
        for key in ['num_experts', 'num_experts_per_tok', 'moe_intermediate_size']:
            if HfConfigFactory.get_config_attr(config, key):
                return True
        return False

    def prepare(
        self,
        sp_size: int,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        padding_free: bool,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        self.num_heads = HfConfigFactory.get_config_attr(model.config, 'num_key_value_heads')
        if self.num_heads is None:
            self.num_heads = HfConfigFactory.get_config_attr(model.config, 'num_attention_heads')
        assert self.num_heads is not None, 'Cannot find num_heads config in config.json'
        if sp_size > 1 and self.num_heads % sp_size != 0:
            raise ValueError(
                f'sp_size ({sp_size}) must divide num_heads ({self.num_heads}) for ulysses sequence parallel.'
            )
        self.padding_free = padding_free
        self.world_size = sp_size

        llm_model = get_llm_model(model)

        if hasattr(llm_model, 'language_model'):
            if hasattr(llm_model.language_model, '_update_causal_mask'):
                self.causal_mask_func = llm_model.language_model._update_causal_mask
        else:
            if hasattr(llm_model, '_update_causal_mask'):
                self.causal_mask_func = llm_model._update_causal_mask

        if not SequenceParallel._global_inited:
            # these operations are global initializations and patches
            self._init_device_mesh(device_mesh)
            self._prepare_flash_attn(llm_model)
            SequenceParallel._global_inited = True

        self._prepare_forward_hook(llm_model)

        if SequenceParallel._is_moe_model(getattr(model, "config", None)):
            self._prepare_moe_aux_loss(llm_model)

        self.model_dtype = next(model.parameters()).dtype
        self.tokenizer = tokenizer
        if not self.padding_free:
            pass

    def pad(self, tensor, padding_value, position_ids=None, dim=1):
        """Pad tensor for sequence parallel"""
        world_size = self.world_size

        def _do_pad(tensor):
            # Ensure seq length is divisible by SP size to allow even split.
            length = tensor.shape[dim]
            pad_num = world_size - (length % world_size)
            if pad_num == 0 or pad_num == world_size:
                return tensor
            if not isinstance(padding_value, torch.Tensor):
                # ids
                pad_shape = ((*tensor.shape[:dim], pad_num, *tensor.shape[dim + 1:]) if dim != -1 else
                             (*tensor.shape[:dim], pad_num))
                pad = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=dim)
            else:
                # For embeddings
                tensor = torch.cat([tensor, padding_value.unsqueeze(0).repeat(tensor.shape[0], pad_num, 1)], dim=dim)
            return tensor

        return _do_pad(tensor)

    def gather(self, local_output, dim: int, position_ids=None):
        """Gather tensor for sequence parallel - reverse of split"""
        if self.world_size == 1:
            return local_output

        # Gather local chunks from each SP rank and concatenate along sequence dim.
        gathered_sp = torch.empty(
            [local_output.shape[0] * self.sp_world_size] + list(local_output.shape[1:]),
            dtype=local_output.dtype,
            device=local_output.device)
        dist.all_gather_into_tensor(gathered_sp, local_output, group=self._sp_group)
        gathered_sp = torch.cat(gathered_sp.split(local_output.shape[0], dim=0), dim=dim)
        return gathered_sp.contiguous()

    def split(self, input, dim: int, position_ids=None):
        """Split tensor for sequence parallel"""
        if self.world_size == 1:
            return input

        # Split along sequence dimension; each rank keeps its local slice.
        rank = dist.get_rank(self._sp_group) if self._sp_group is not None else 0
        dim_size = input.size(dim)
        assert dim_size % self.sp_world_size == 0, (
            f'The dimension to split ({dim_size}) is not a multiple of '
            f'world size ({self.sp_world_size}), cannot split tensor evenly')

        tensor_list = torch.split(input, dim_size // self.sp_world_size, dim=dim)
        output = tensor_list[rank].contiguous()
        return output

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

    def pad_and_split_inputs(self,
                             input_ids,
                             input_embeds,
                             labels,
                             position_ids,
                             attention_mask,
                             loss_scale,
                             embed_tokens=None,
                             real_position_ids=None,
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
        if position_ids is not None:
            position_ids = self.pad(position_ids, padding_value=-1, position_ids=real_position_ids, dim=-1)
        if labels is not None:
            labels = self.pad(labels, padding_value=-100, position_ids=real_position_ids)
        if loss_scale is not None:
            loss_scale = self.pad(loss_scale, padding_value=0., position_ids=real_position_ids)
        if real_position_ids is not None:
            real_position_ids = self.pad(real_position_ids, padding_value=-1, position_ids=real_position_ids)
        if (input_ids is not None or input_embeds is not None) and batch_size > 1:
            inputs = input_ids if input_ids is not None else input_embeds
            attn_shape = inputs.shape[1]  # The sequence length
            if attention_mask is None:
                attention_mask = torch.ones_like(real_position_ids)
            # no need position_ids here, because padding_free does not need attention_mask,
            attention_mask = self.pad(attention_mask, padding_value=0)
            cache_position = torch.arange(0, attn_shape, device=inputs.device)
            # pad attention mask to 4d to avoid calculation errors
            if hasattr(self, 'causal_mask_func') and self.causal_mask_func is not None:
                attention_mask = self.causal_mask_func(attention_mask, inputs.to(self.model_dtype), cache_position,
                                                       None, None)
        if extra_split_values is not None:
            for (tensor, pad_value, split_dim) in extra_split_values:
                extra_values.append(
                    self.pad(tensor, padding_value=pad_value, position_ids=real_position_ids, dim=split_dim))
        if input_ids is not None:
            input_ids = self.split(input_ids, dim=1, position_ids=real_position_ids)
        if input_embeds is not None:
            input_embeds = self.split(input_embeds, dim=1, position_ids=real_position_ids)
        if labels is not None:
            # Align next-token labels before splitting so each rank sees local targets.
            labels = torch.roll(labels, shifts=-1, dims=-1)
            labels = self.split(labels, dim=-1, position_ids=real_position_ids)
        if loss_scale is not None:
            loss_scale = torch.roll(loss_scale, shifts=-1, dims=-1)
            loss_scale = self.split(loss_scale, dim=-1, position_ids=real_position_ids)

        if position_ids is not None:
            position_ids = self.split(position_ids, dim=-1, position_ids=real_position_ids)
        if extra_split_values is not None:
            for i in range(len(extra_values)):
                extra_values[i] = self.split(
                    extra_values[i], dim=extra_split_values[i][2], position_ids=real_position_ids)
        return input_ids, input_embeds, labels, position_ids, attention_mask, loss_scale, extra_values

    def _init_device_mesh(self, device_mesh: Optional[DeviceMesh] = None):
        """Initialize process groups for sequence parallel."""
        if not isinstance(device_mesh, DeviceMesh):
            raise RuntimeError("SequenceParallel requires a twinkle DeviceMesh for initialization.")

        self.device_mesh = device_mesh
        self.sp_world_size = self.world_size
        self.dp_world_size = device_mesh.data_world_size or 1
        self._sp_group = _get_sp_group_from_device_mesh(device_mesh, self.sp_world_size)
        if self._sp_group is None and self.sp_world_size > 1:
            raise RuntimeError("Failed to create sequence-parallel group from DeviceMesh.")


    def prepare_inputs(self, inputs):
        """Prepare inputs

        1. set extra_kwargs['position_ids']
        2. split labels
        """
        position_ids = None
        input_ids = inputs.get('input_ids')
        position_ids = inputs.get('position_ids')
        if position_ids is not None and input_ids is not None and position_ids.shape[0] == input_ids.shape[0]:
            self.extra_kwargs['position_ids'] = position_ids.clone()
        if input_ids is not None:
            self.extra_kwargs['input_ids'] = input_ids.clone()
        if 'labels' in inputs:
            labels = inputs['labels']
            _, _, labels, _, _, _, _ = self.pad_and_split_inputs(
                None, None, labels, None, None, None, real_position_ids=position_ids)
            inputs['labels'] = labels
        return inputs


sequence_parallel = SequenceParallel()


@dataclass(frozen=True)
class SequenceParallelConfig:
    enabled: bool = True
    ulysses_size: Optional[int] = None
    padding_free: bool = False
    gather_logits: bool = True


def _get_ulysses_size(device_mesh, sp_config: Optional[Dict[str, Any]] = None) -> int:
    if sp_config:
        cfg_size = sp_config.get("ulysses_size")
        if cfg_size is not None:
            return int(cfg_size)
    if device_mesh is None:
        return 1
    if getattr(device_mesh, "ulysses_size", None) is not None:
        return int(device_mesh.ulysses_size)
    if getattr(device_mesh, "has_dim", None) and device_mesh.has_dim("sp"):
        return device_mesh.get_dim_size("sp")
    return 1


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
        self.enabled = bool(self.sp_config.get("enabled", True))
        self.ulysses_size = _get_ulysses_size(device_mesh, self.sp_config)
        self.padding_free = bool(self.sp_config.get("padding_free", False))
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
            raise RuntimeError("torch.distributed must be initialized before enabling sequence parallel.")
        if not isinstance(self.device_mesh, DeviceMesh):
            raise RuntimeError("SequenceParallelStrategy requires a twinkle DeviceMesh when ulysses_size > 1.")
        if self._model_ref is None:
            raise RuntimeError("SequenceParallelStrategy requires a model reference to initialize.")
        tokenizer = self._get_tokenizer()
        if tokenizer is None:
            raise RuntimeError("SequenceParallelStrategy requires a tokenizer to initialize.")
        sequence_parallel.prepare(
            self.ulysses_size,
            self._model_ref,
            tokenizer,
            self.padding_free,
            device_mesh=self.device_mesh,
        )
        self._initialized = True
        return True

    def preprocess_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or self.ulysses_size <= 1:
            return inputs
        return sequence_parallel.prepare_inputs(inputs)

    def postprocess_outputs(self, outputs: Any) -> Any:
        if (
            not self.enabled
            or self.ulysses_size <= 1
            or not self.sp_config.get("gather_logits", True)
        ):
            return outputs
        # Optionally reassemble full-seq logits for downstream consumers.
        logits = None
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, dict):
            logits = outputs.get("logits")
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            logits = outputs[0]
        if logits is None or not torch.is_tensor(logits) or logits.dim() < 2:
            return outputs
        gathered = sequence_parallel.gather(
            logits, dim=1, position_ids=sequence_parallel.real_position_ids
        )
        if hasattr(outputs, "logits"):
            outputs.logits = gathered
            return outputs
        if isinstance(outputs, dict):
            outputs["logits"] = gathered
            return outputs
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            new = list(outputs)
            new[0] = gathered
            return type(outputs)(new)
        return outputs

    def reduce_loss(self, loss: torch.Tensor, labels: Optional[torch.Tensor], ignore_index: int = -100) -> torch.Tensor:
        if not self.enabled or self.ulysses_size <= 1:
            return loss
        if labels is None or sequence_parallel._sp_group is None:
            return loss
        # Reduce loss with token-count normalization across SP ranks.
        num_valid_tokens = (labels != ignore_index).sum().to(loss.device)
        reduced_loss = loss * num_valid_tokens
        dist.all_reduce(reduced_loss, group=sequence_parallel._sp_group)
        dist.all_reduce(num_valid_tokens, group=sequence_parallel._sp_group)
        return reduced_loss / num_valid_tokens

    def wrap_model(self, model, optimizer=None):
        self.initialize()
        return model, optimizer

    def unwrap_model(self, model):
        return model
