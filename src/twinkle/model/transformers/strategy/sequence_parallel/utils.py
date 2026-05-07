import math
import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional, Tuple

from twinkle.utils import DeviceMesh


def get_config_attr(config, key, default=None):
    return getattr(config, key, default)


def is_hccl_backend(group=None) -> bool:
    return dist.get_backend(group) == 'hccl'


def is_moe_config(config) -> bool:
    if config is None:
        return False
    if 'Moe' in config.__class__.__name__:
        return True
    for key in ['num_experts', 'num_experts_per_tok', 'moe_intermediate_size']:
        if get_config_attr(config, key):
            return True
    return False


def get_cu_seqlens_from_position_ids(position_ids: torch.LongTensor):
    position_ids = position_ids[0]
    seq_start_indices = torch.where(position_ids == 0)[0]
    seq_end_indices = torch.cat([seq_start_indices[1:], torch.tensor([len(position_ids)], device=position_ids.device)])
    seq_lengths = seq_end_indices - seq_start_indices
    cu_seqlens = torch.cumsum(torch.cat([torch.tensor([0], device=position_ids.device), seq_lengths]), dim=0)
    return cu_seqlens


def get_packed_cu_seqlens_from_sequence_parallel_context(
    sequence_parallel_context,
    *,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if sequence_parallel_context is None:
        return None

    extra_kwargs = getattr(sequence_parallel_context, 'extra_kwargs', {})
    if extra_kwargs.get('padding_free', False):
        position_ids = getattr(sequence_parallel_context, 'real_position_ids', None)
        if position_ids is not None:
            position_ids = sequence_parallel_context._extract_real_position_ids(position_ids)
            position_ids = sequence_parallel_context.pad(position_ids, padding_value=-1, position_ids=position_ids)
            return get_cu_seqlens_from_position_ids(position_ids).to(dtype=torch.int32, device=device)
    return None


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
    if device_mesh.has_dim('dp'):
        dp_rank = coord[device_mesh._get_dim_index('dp')]
    if device_mesh.has_dim('fsdp'):
        fsdp_rank = coord[device_mesh._get_dim_index('fsdp')]

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


def _derive_sequence_parallel_sizes(num_heads: int, seq_world_size: int) -> Tuple[int, int]:
    if seq_world_size <= 1:
        return 1, 1
    sp_world_size = math.gcd(int(num_heads), int(seq_world_size))
    sp_world_size = max(1, sp_world_size)
    if seq_world_size % sp_world_size != 0:
        raise ValueError(
            f'seq_world_size ({seq_world_size}) must be divisible by derived sp_world_size ({sp_world_size}).')
    rp_world_size = seq_world_size // sp_world_size
    return sp_world_size, rp_world_size


def _get_sequence_group_specs(
    device_mesh: Optional[DeviceMesh],
    seq_world_size: int,
    sp_world_size: int,
    rp_world_size: int,
) -> List[Dict[str, Any]]:
    if device_mesh is None or seq_world_size <= 1:
        return []

    if seq_world_size != sp_world_size * rp_world_size:
        raise ValueError(f'seq_world_size ({seq_world_size}) must equal sp_world_size ({sp_world_size}) * '
                         f'rp_world_size ({rp_world_size}).')

    raw_data_world_size = _get_raw_data_world_size(device_mesh)
    if raw_data_world_size % seq_world_size != 0:
        raise ValueError(
            f'data_world_size ({raw_data_world_size}) must be divisible by seq_world_size ({seq_world_size}).')

    non_data_indices = []
    if device_mesh.mesh_dim_names is not None:
        for i, name in enumerate(device_mesh.mesh_dim_names):
            if name in ('dp', 'fsdp'):
                continue
            non_data_indices.append(i)

    groups: Dict[Tuple[int, Tuple[int, ...]], List[Tuple[int, int]]] = {}
    for r in device_mesh.mesh.flatten().tolist():
        rank = int(r)
        coord = device_mesh._get_coord_for_rank(rank)
        if coord is None:
            continue
        raw_rank = _get_raw_data_rank(device_mesh, rank)
        if raw_rank is None:
            continue
        group_id = raw_rank // seq_world_size
        seq_local_rank = raw_rank % seq_world_size
        non_data_key = tuple(coord[i] for i in non_data_indices)
        key = (group_id, non_data_key)
        groups.setdefault(key, []).append((seq_local_rank, rank))

    group_specs = []
    for key, items in groups.items():
        items = sorted(items, key=lambda item: item[0])
        local_ranks = [local_rank for local_rank, _ in items]
        if local_ranks != list(range(seq_world_size)):
            raise ValueError(f'Invalid sequence-parallel rank layout for key={key}: {local_ranks}')
        seq_ranks = [rank for _, rank in items]
        if len(seq_ranks) != seq_world_size:
            raise ValueError(
                f'Sequence-parallel group size mismatch for key={key}: expected {seq_world_size}, got {len(seq_ranks)}')
        sp_groups = [seq_ranks[i * sp_world_size:(i + 1) * sp_world_size] for i in range(rp_world_size)]
        rp_groups = [[sp_groups[rp_idx][sp_idx] for rp_idx in range(rp_world_size)] for sp_idx in range(sp_world_size)]
        group_specs.append({
            'key': key,
            'seq_ranks': seq_ranks,
            'sp_groups': sp_groups,
            'rp_groups': rp_groups,
        })

    group_specs.sort(key=lambda item: item['key'])
    return group_specs


def _get_seq_groups_from_device_mesh(
    device_mesh: Optional[DeviceMesh],
    seq_world_size: int,
    sp_world_size: int,
    rp_world_size: int,
) -> Tuple[Optional[dist.ProcessGroup], Optional[dist.ProcessGroup], Optional[dist.ProcessGroup], int, int]:
    if device_mesh is None or seq_world_size <= 1:
        return None, None, None, 0, 0
    if not dist.is_available() or not dist.is_initialized():
        return None, None, None, 0, 0

    rank = dist.get_rank()
    sp_group = None
    rp_group = None
    data_rank_group = None
    sp_rank = 0
    rp_rank = 0
    group_specs = _get_sequence_group_specs(device_mesh, seq_world_size, sp_world_size, rp_world_size)

    for spec in group_specs:
        seq_pg = dist.new_group(ranks=spec['seq_ranks'])
        if rank in spec['seq_ranks']:
            data_rank_group = seq_pg

        if sp_world_size > 1:
            for ranks in spec['sp_groups']:
                pg = dist.new_group(ranks=ranks)
                if rank in ranks:
                    sp_group = pg
                    sp_rank = ranks.index(rank)

        if rp_world_size > 1:
            for ranks in spec['rp_groups']:
                pg = dist.new_group(ranks=ranks)
                if rank in ranks:
                    rp_group = pg
                    rp_rank = ranks.index(rank)

    if data_rank_group is None:
        raise RuntimeError('Failed to create sequence-parallel data group from DeviceMesh.')
    if sp_world_size > 1 and sp_group is None:
        raise RuntimeError('Failed to create sequence-parallel SP group from DeviceMesh.')
    if rp_world_size > 1 and rp_group is None:
        raise RuntimeError('Failed to create sequence-parallel ring group from DeviceMesh.')

    return sp_group, rp_group, data_rank_group, sp_rank, rp_rank


def _get_ulysses_size(device_mesh, sp_config: Optional[Dict[str, Any]] = None) -> int:
    if sp_config:
        cfg_size = sp_config.get('ulysses_size')
        if cfg_size is not None:
            return int(cfg_size)
    if device_mesh is None:
        return 1
    if getattr(device_mesh, 'ulysses_size', None) is not None:
        return int(device_mesh.ulysses_size)
    return 1


def seq_to_head_shard(tensor: torch.Tensor, sequence_parallel) -> torch.Tensor:
    if getattr(sequence_parallel, 'sp_world_size', 1) <= 1:
        return tensor
    # [B, local_S, H, D] -> [B, global_S, local_H, D]
    return _SeqAllToAll.apply(sequence_parallel._sp_group, tensor, 2, 1)


def head_to_seq_shard(tensor: torch.Tensor, sequence_parallel) -> torch.Tensor:
    if getattr(sequence_parallel, 'sp_world_size', 1) <= 1:
        return tensor
    # [B, global_S, local_H, D] -> [B, local_S, H, D]
    return _SeqAllToAll.apply(sequence_parallel._sp_group, tensor, 1, 2)


class GatherLoss(torch.autograd.Function):
    """Gather loss from sequence group."""

    @staticmethod
    def forward(ctx, loss, labels, gather_idx=None, position_ids=None):
        from . import sequence_parallel
        ctx.scatter_shape = loss.shape[gather_idx or 0]
        ctx.gather_idx = gather_idx or 0
        if position_ids is not None:
            position_ids = sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        ctx.position_ids = position_ids
        output = sequence_parallel.gather(loss, dim=ctx.gather_idx, position_ids=position_ids)
        if labels is not None:
            labels_output = sequence_parallel.gather(labels, dim=ctx.gather_idx, position_ids=position_ids)
        else:
            labels_output = None
        return output, labels_output

    @staticmethod
    def backward(ctx, *grad_output):
        from . import sequence_parallel
        _grad = grad_output[0] * sequence_parallel.world_size
        if sequence_parallel.rp_world_size > 1:
            _grad = sequence_parallel.split(_grad, dim=ctx.gather_idx, position_ids=ctx.position_ids).contiguous()
        else:
            _grad = _grad.split(
                ctx.scatter_shape, dim=ctx.gather_idx)[dist.get_rank(group=sequence_parallel._sp_group)].contiguous()
        return _grad, None, None, None


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

    def post_func(input):
        if permute_idx is not None:
            input = input.permute(permute_idx).contiguous()
        output = input.reshape(res_shape).contiguous()
        return output

    return post_func


def pre_all2all_fun(permute_idx, inp_shape, input):
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
        return None, _SeqAllToAll.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx), None, None


class DistributedAttention(torch.nn.Module):

    def __init__(
        self,
        local_attention,
        sequence_parallel,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:
        super().__init__()
        self.local_attn = local_attention
        self.sequence_parallel = sequence_parallel
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: torch.Tensor, *args:
                Any, **kwargs) -> torch.Tensor:
        if self.sequence_parallel.world_size == 1:
            return self.local_attn(query, key, value, attention_mask, *args, **kwargs)
        if self.sequence_parallel.rp_world_size > 1 and attention_mask is not None:
            if torch.is_tensor(attention_mask) and not attention_mask.all():
                raise NotImplementedError(
                    'Derived ring attention only supports padding-free / packed inputs without masked padding.')

        if self.sequence_parallel.sp_world_size > 1:
            query_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, query, self.scatter_idx, self.gather_idx)
            key_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, key, self.scatter_idx, self.gather_idx)
            value_layer = _SeqAllToAll.apply(self.sequence_parallel._sp_group, value, self.scatter_idx, self.gather_idx)
        else:
            query_layer, key_layer, value_layer = query, key, value

        if (self.sequence_parallel.sp_world_size > 1 and torch.is_tensor(attention_mask) and attention_mask.dim() == 4):
            if attention_mask.shape[-1] != key_layer.shape[1]:
                attention_mask = self.sequence_parallel.gather(attention_mask, dim=-1, position_ids=None)
            if attention_mask.shape[-2] != query_layer.shape[1]:
                attention_mask = self.sequence_parallel.gather(attention_mask, dim=-2, position_ids=None)

        if self.sequence_parallel.rp_world_size > 1:
            kwargs.pop('position_ids', None)
            position_ids = self.sequence_parallel.real_position_ids
            position_ids = self.sequence_parallel.pad(position_ids, padding_value=-1, position_ids=position_ids)
        else:
            position_ids = kwargs.pop('position_ids')
            if position_ids is not None and self.sequence_parallel.sp_world_size > 1:
                # Reuse the generic gather path to support both 2D and 3D position_ids (e.g. mrope).
                position_ids = self.sequence_parallel.gather(position_ids.contiguous(), dim=-1, position_ids=None)

        context_layer = self.local_attn(
            query_layer, key_layer, value_layer, attention_mask, *args, position_ids=position_ids, **kwargs)

        if self.sequence_parallel.sp_world_size > 1:
            output = _SeqAllToAll.apply(self.sequence_parallel._sp_group, context_layer, self.gather_idx,
                                        self.scatter_idx)
        else:
            output = context_layer

        return output
