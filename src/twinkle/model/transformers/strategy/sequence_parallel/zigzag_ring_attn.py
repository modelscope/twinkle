# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import torch
import torch.distributed as dist
import torch.nn.functional as F
from functools import cache
from typing import Optional, Tuple


class RingComm:

    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError('commit called twice')
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError('wait called before commit')
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


def get_half_index(cu_seqlens, *, front: bool):
    if len(cu_seqlens) == 2:
        if front:
            return slice(None, cu_seqlens[-1] // 2)
        return slice(cu_seqlens[-1] // 2, None)

    index = torch.zeros((cu_seqlens[-1].item(), ), dtype=torch.bool)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        if front:
            end = (start + end) // 2
        else:
            start = (start + end) // 2
        index[start:end] = True
    return index


@torch.jit.script
def get_half_lse(lse, cu_seqlens, *, front: bool):
    new_lse = torch.empty(
        (lse.shape[0], lse.shape[1] // 2),
        dtype=lse.dtype,
        device=lse.device,
    )
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        new_start, new_end = start // 2, end // 2
        if front:
            end -= (end - start) // 2
        else:
            start += (end - start) // 2
        new_lse[:, new_start:new_end] = lse[:, start:end]
    return new_lse


def update_out_and_lse(out, lse, block_out, block_lse):
    if out is None:
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
        sig_diff = None
    else:
        block_out = block_out.to(torch.float32)
        block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

        diff = block_lse - lse
        sig_diff = torch.sigmoid(diff)

        out = out - sig_diff * (out - block_out)
        lse = lse - F.logsigmoid(lse - block_lse)
    return out, lse, sig_diff


@cache
def _get_default_args(func):
    spec = inspect.getfullargspec(func)
    defaults = spec.defaults if spec.defaults is not None else ()
    padded_defaults = (None, ) * (len(spec.args) - len(defaults)) + defaults
    args = dict(zip(spec.args, padded_defaults))
    if 'softcap' in args:
        args['softcap'] = 0.0
    return args


def get_default_args(func):
    if inspect.isfunction(func):
        return _get_default_args(func)
    return _get_default_args(func._init_fn)


def squeeze_batch(*tensors):
    squeezed = []
    for sub in tensors:
        if sub.shape[0] == 1:
            squeezed.append(sub.squeeze(0))
        else:
            squeezed.append(sub)
    return tuple(squeezed)


def padding(tensor, cu_seqlens, padding_value, front):
    if len(cu_seqlens) == 2:
        if front:
            return torch.cat((tensor, torch.full_like(tensor, padding_value).to(tensor.dtype).to(tensor.device)), dim=0)
        return torch.cat((torch.full_like(tensor, padding_value).to(tensor.dtype).to(tensor.device), tensor), dim=0)

    output = []
    acc = 0
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        half_len = (end - start) // 2
        acc += half_len
        half_start = start // 2
        local_tensor = tensor[half_start:half_start + half_len]
        if front:
            output.append(local_tensor)
            output.append(torch.full_like(local_tensor, padding_value).to(local_tensor.dtype).to(local_tensor.device))
        else:
            output.append(torch.full_like(local_tensor, padding_value).to(local_tensor.dtype).to(local_tensor.device))
            output.append(local_tensor)
    assert acc == tensor.shape[0]
    return torch.cat(output)


def forward(
    q,
    k,
    v,
    causal,
    cu_seqlens,
    max_seqlen,
    block_seq_len,
    dropout_p,
    softmax_scale,
    alibi_slopes,
    window_size,
):
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]
    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2
    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward

    params = get_default_args(_flash_attn_varlen_forward).copy()
    params.update({
        'q': q,
        'k': k,
        'v': v,
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_kv,
        'max_seqlen_q': max_seqlen_q,
        'max_seqlen_k': max_seqlen_kv,
        'dropout_p': dropout_p,
        'softmax_scale': softmax_scale,
        'causal': causal,
        'alibi_slopes': alibi_slopes,
        'return_softmax': True and dropout_p > 0,
    })
    if 'window_size' in params:
        params.update({'window_size': window_size})
    else:
        params.update({
            'window_size_left': window_size[0],
            'window_size_right': window_size[1],
        })
    assert k.shape[-0] == cu_seqlens_kv[-1]
    assert q.shape[-0] == cu_seqlens_q[-1]
    assert max_seqlen_q == (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    assert max_seqlen_kv == (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()
    outputs = _flash_attn_varlen_forward(**params)
    if len(outputs) == 8:
        block_out, _, _, _, _, block_lse, _, _ = outputs
    else:
        assert len(outputs) == 4
        block_out, block_lse, _, _ = outputs
    return block_out, block_lse


def backward(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    causal,
    cu_seqlens,
    max_seqlen,
    block_seq_len,
    dq_buffer,
    dk_buffer,
    dv_buffer,
    dropout_p,
    softmax_scale,
    alibi_slopes,
    deterministic,
    window_size,
):
    seqlen_q = q.shape[0]
    seqlen_kv = k.shape[0]

    half_cu_seqlens = cu_seqlens // 2
    half_max_seqlen = max_seqlen // 2
    cu_seqlens_q = half_cu_seqlens if seqlen_q == block_seq_len else cu_seqlens
    max_seqlen_q = half_max_seqlen if seqlen_q == block_seq_len else max_seqlen
    cu_seqlens_kv = half_cu_seqlens if seqlen_kv == block_seq_len else cu_seqlens
    max_seqlen_kv = half_max_seqlen if seqlen_kv == block_seq_len else max_seqlen
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

    params = get_default_args(_flash_attn_varlen_backward).copy()
    params.update({
        'dout': dout,
        'q': q,
        'k': k,
        'v': v,
        'out': out,
        'softmax_lse': softmax_lse,
        'dq': dq_buffer[:seqlen_q],
        'dk': dk_buffer[:seqlen_kv],
        'dv': dv_buffer[:seqlen_kv],
        'cu_seqlens_q': cu_seqlens_q,
        'cu_seqlens_k': cu_seqlens_kv,
        'max_seqlen_q': max_seqlen_q,
        'max_seqlen_k': max_seqlen_kv,
        'dropout_p': dropout_p,
        'softmax_scale': softmax_scale,
        'causal': causal,
        'alibi_slopes': alibi_slopes,
        'deterministic': deterministic,
    })
    assert dout.shape[0] == q.shape[0]
    assert dout.shape[0] == out.shape[0]
    assert softmax_lse.shape[1] == q.shape[0]
    assert k.shape[0] == cu_seqlens_kv[-1]
    assert q.shape[0] == cu_seqlens_q[-1]
    assert max_seqlen_q == (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).max().item()
    assert max_seqlen_kv == (cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]).max().item()
    if 'window_size' in params:
        params.update({'window_size': window_size})
    else:
        params.update({
            'window_size_left': window_size[0],
            'window_size_right': window_size[1],
        })
    _flash_attn_varlen_backward(**params)


def lse_grad(out, lse, block_out, block_lse, sig, grad_out, grad_lse):
    grad_out_input = grad_out * (1 - sig)
    grad_block_out = grad_out * sig
    d_new_out_d_lse = (out - block_out) * (sig * (1 - sig))
    grad_lse_input = (grad_out * d_new_out_d_lse).sum(dim=-1, keepdim=True)
    grad_lse_input_final = grad_lse_input + grad_lse * torch.sigmoid(lse - block_lse)
    grad_block_lse = -grad_lse_input_final + grad_lse
    return grad_out_input, grad_lse_input_final, grad_block_out, grad_block_lse


def zigzag_ring_flash_attn_varlen_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens,
        max_seqlen,
        half_index0,
        half_index1,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    assert causal, 'zigzag ring is meaningless for causal=False'
    comm = RingComm(process_group)
    q, k, v = squeeze_batch(q, k, v)
    q1 = q[half_index1]
    cu_seqlens = cu_seqlens // comm.world_size
    max_seqlen = max_seqlen // comm.world_size
    block_seq_len = q.shape[0] // 2
    out = None
    lse = None
    next_k, next_v = None, None

    for step in range(comm.world_size):
        if step + 1 != comm.world_size:
            next_k, next_v = comm.send_recv_kv(k, v)
        if step == 0:
            block_out, block_lse = forward(q, k, v, True, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            out, lse, _ = update_out_and_lse(out, lse, block_out, block_lse)
        elif step <= comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            block_out, block_lse = forward(q, k0, v0, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            out, lse, _ = update_out_and_lse(out, lse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            out[half_index1], lse[half_index1], _ = update_out_and_lse(out[half_index1], lse[half_index1], block_out,
                                                                       block_lse)

        if step + 1 != comm.world_size:
            comm.wait()
            k, v = next_k, next_v

    out = out.to(q.dtype)
    lse = lse.squeeze(dim=-1).transpose(0, 1)
    return out.unsqueeze(0), lse.unsqueeze(0)


def zigzag_ring_flash_attn_varlen_backward(
        process_group,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        cu_seqlens,
        max_seqlen,
        half_index0,
        half_index1,
        softmax_scale,
        dropout_p=0,
        causal=True,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
):
    assert causal, 'zigzag ring is meaningless for causal=False'
    kv_comm = RingComm(process_group)
    d_kv_comm = RingComm(process_group)
    dk_comm_buffer = dv_comm_buffer = None
    dq = dk = dv = None
    next_dk = next_dv = None
    next_k = next_v = None

    dout, q, k, v, out, softmax_lse = squeeze_batch(dout, q, k, v, out, softmax_lse)
    q1 = q[half_index1]
    cu_seqlens = cu_seqlens // kv_comm.world_size
    max_seqlen = max_seqlen // kv_comm.world_size
    block_seq_len = q.shape[0] // 2

    dq_buffer = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    dk_buffer = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    dv_buffer = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    origin_q, origin_k, origin_v = q, k, v

    out_lse = []
    fout = None
    flse = None
    for step in range(kv_comm.world_size):
        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step == 0:
            block_out, block_lse = forward(q, k, v, True, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            fout, flse, sig_diff = update_out_and_lse(fout, flse, block_out, block_lse)
        elif step <= kv_comm.rank:
            k0 = k[half_index0]
            v0 = v[half_index0]
            block_out, block_lse = forward(q, k0, v0, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            fout, flse, sig_diff = update_out_and_lse(fout, flse, block_out, block_lse)
        else:
            block_out, block_lse = forward(q1, k, v, False, cu_seqlens, max_seqlen, block_seq_len, dropout_p,
                                           softmax_scale, alibi_slopes, window_size)
            fout[half_index1], flse[half_index1], sig_diff = update_out_and_lse(fout[half_index1], flse[half_index1],
                                                                                block_out, block_lse)

        block_lse = block_lse.transpose(0, 1).unsqueeze(-1)
        if step > kv_comm.rank:
            block_out = padding(block_out, cu_seqlens, 0, front=False)
            block_lse = padding(block_lse, cu_seqlens, -1e5, front=False)
            sig_diff = padding(sig_diff, cu_seqlens, 0, front=False)

        out_lse.append((fout, flse, block_out, block_lse, sig_diff))

        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

    current_dout = dout
    current_dlse = torch.zeros_like(softmax_lse.transpose(0, 1).unsqueeze(-1))
    block_gradients = {}

    for i in reversed(range(len(out_lse))):
        if i == 0:
            continue
        stored_out, stored_lse, stored_block_out, stored_block_lse, stored_sig = out_lse[i]
        grad_out_input, grad_lse_input, grad_block_out, grad_block_lse = lse_grad(
            stored_out,
            stored_lse,
            stored_block_out,
            stored_block_lse,
            stored_sig,
            current_dout,
            current_dlse,
        )
        current_dout = grad_out_input
        current_dlse = grad_lse_input
        block_gradients[i] = {'grad_block_out': grad_block_out, 'grad_block_lse': grad_block_lse}

    q, k, v = origin_q, origin_k, origin_v

    for step in range(kv_comm.world_size):
        _, _, block_out, block_lse, _ = out_lse[step]
        if block_out.isnan().any() or block_lse.isnan().any():
            raise RuntimeError('NaN detected in ring attention backward recompute.')
        block_lse = block_lse.transpose(0, 1).squeeze(2)

        if step + 1 != kv_comm.world_size:
            next_k, next_v = kv_comm.send_recv_kv(k, v)

        if step == 0:
            block_dout = current_dout
        else:
            block_dout = block_gradients[step]['grad_block_out']

        if block_dout.isnan().any():
            raise RuntimeError('NaN detected in ring attention dout.')

        if step == 0:
            backward(
                block_dout.to(dout.dtype), q, k, v, block_out, block_lse, True, cu_seqlens, max_seqlen, block_seq_len,
                dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes, deterministic, window_size)
            dq = dq_buffer.to(torch.float32)
            dk = dk_buffer.to(torch.float32)
            dv = dv_buffer.to(torch.float32)
            if dq.isnan().any() or dk.isnan().any() or dv.isnan().any():
                raise RuntimeError('NaN detected in ring attention gradients.')
        else:
            if step <= kv_comm.rank:
                k0 = k[half_index0]
                v0 = v[half_index0]
                backward(
                    block_dout.to(dout.dtype), q, k0, v0, block_out, block_lse, False, cu_seqlens, max_seqlen,
                    block_seq_len, dq_buffer, dk_buffer, dv_buffer, dropout_p, softmax_scale, alibi_slopes,
                    deterministic, window_size)
                dq += dq_buffer
            else:
                backward(block_dout[half_index1].to(dout.dtype), q1, k, v, block_out[half_index1],
                         get_half_lse(block_lse, cu_seqlens,
                                      front=False), False, cu_seqlens, max_seqlen, block_seq_len, dq_buffer, dk_buffer,
                         dv_buffer, dropout_p, softmax_scale, alibi_slopes, deterministic, window_size)
                dq[half_index1] += dq_buffer[:block_seq_len]

            d_kv_comm.wait()
            dk_comm_buffer = torch.empty_like(dk)
            dv_comm_buffer = torch.empty_like(dv)
            dk_comm_buffer.copy_(dk)
            dv_comm_buffer.copy_(dv)
            dk, dv = next_dk, next_dv

            if step <= kv_comm.rank:
                dk[half_index0] += dk_buffer[:block_seq_len]
                dv[half_index0] += dv_buffer[:block_seq_len]
            else:
                dk += dk_buffer
                dv += dv_buffer
            if dq.isnan().any() or dk.isnan().any() or dv.isnan().any():
                raise RuntimeError('NaN detected in accumulated ring attention gradients.')
        if step + 1 != kv_comm.world_size:
            kv_comm.wait()
            k, v = next_k, next_v

        next_dk, next_dv = d_kv_comm.send_recv_kv(dk, dv, dk_comm_buffer, dv_comm_buffer)

    d_kv_comm.wait()
    return dq.to(q.dtype).unsqueeze(0), next_dk.to(q.dtype).unsqueeze(0), next_dv.to(q.dtype).unsqueeze(0)


class ZigZagRingFlashAttnVarlenFunc(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1]**(-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        rp_world_size = dist.get_world_size(group)
        half_index0 = get_half_index(cu_seqlens // rp_world_size, front=True)
        half_index1 = get_half_index(cu_seqlens // rp_world_size, front=False)
        out, softmax_lse = zigzag_ring_flash_attn_varlen_forward(
            group,
            q,
            k,
            v,
            cu_seqlens,
            max_seqlen,
            half_index0,
            half_index1,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        is_half_index_tensor = isinstance(half_index0, torch.Tensor)
        ctx.is_half_index_tensor = is_half_index_tensor
        if is_half_index_tensor:
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens, half_index0, half_index1)
        else:
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens)
            ctx.half_index0 = half_index0
            ctx.half_index1 = half_index1
        ctx.max_seqlen = max_seqlen
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        if ctx.is_half_index_tensor:
            q, k, v, out, softmax_lse, cu_seqlens, half_index0, half_index1 = ctx.saved_tensors
        else:
            q, k, v, out, softmax_lse, cu_seqlens = ctx.saved_tensors
            half_index0 = ctx.half_index0
            half_index1 = ctx.half_index1
        dq, dk, dv = zigzag_ring_flash_attn_varlen_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens,
            ctx.max_seqlen,
            half_index0,
            half_index1,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def zigzag_ring_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        group=None,
):
    return ZigZagRingFlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens,
        max_seqlen,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
