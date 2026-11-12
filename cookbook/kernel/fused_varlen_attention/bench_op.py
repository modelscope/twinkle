"""单算子基准：本算子与昇腾上现有可行写法的对照。

第 ① 节（最严格）：**同一个拼接输入、同样词数，只换内核**。现状只能用 twinkle 已支持的
``sdpa_attention`` 配一张 ``T×T`` 块对角掩码，本节把它和 ``fused_varlen_attention`` 放在
同一份输入上比。

第 ② 节：训练里的真实前后——一边把文档补齐成矩阵调 ``sdpa_attention``（配 4D「因果 + 补齐」
掩码），一边是同样的文档拼成一条序列调本算子，处理的有效词数相同。

第 ③ 节（附录）：固定形状下与「逐文档掩码实现」「逐文档因果实现」「整段因果实现」对照。
最后一种在训练里不成立（文档互相可见），只用来标出内核自身的天花板。

用法：python bench_op.py [--out result.json] [--skip-appendix]
"""
from __future__ import annotations

import argparse
import json
import statistics
import torch


def timed(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.npu.synchronize()
    return start.elapsed_time(end) / iters


def bench(fn, repeats=5, iters=10):
    return statistics.median(timed(fn, iters) for _ in range(repeats))


def peak_memory(fn):
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    fn()
    torch.npu.synchronize()
    return torch.npu.max_memory_allocated() / 2**20


class _Module:
    """``sdpa_attention`` 只用到 ``num_key_value_groups`` 这一个属性。"""

    def __init__(self, groups):
        self.num_key_value_groups = groups


def hf_style_mask(lengths, dtype, device='npu:0'):
    """transformers 在补齐布局下传给 sdpa 的 4D 掩码：未来位置与前缀之外都置最小浮点。"""
    batch, widest = len(lengths), max(lengths)
    blocked = torch.finfo(dtype).min
    mask = torch.zeros((batch, 1, widest, widest), dtype=dtype, device=device)
    mask = mask.masked_fill(torch.triu(torch.ones(widest, widest, dtype=torch.bool, device=device), 1), blocked)
    for index, length in enumerate(lengths):
        mask[index, :, :, length:] = blocked
    return mask


def make_padded(lengths, heads, kv_heads, dim, dtype, mask):
    batch, widest = len(lengths), max(lengths)
    query = torch.randn(batch, heads, widest, dim, device='npu:0', dtype=dtype).requires_grad_()
    key = torch.randn(batch, kv_heads, widest, dim, device='npu:0', dtype=dtype).requires_grad_()
    value = torch.randn(batch, kv_heads, widest, dim, device='npu:0', dtype=dtype).requires_grad_()
    grad = torch.randn(batch, widest, heads, dim, device='npu:0', dtype=dtype)  # 接口返回 [B, S, H, D]
    return query, key, value, grad, mask


def make_packed(lengths, heads, kv_heads, dim, dtype):
    total = sum(lengths)
    query = torch.randn(total, heads, dim, device='npu:0', dtype=dtype).requires_grad_()
    key = torch.randn(total, kv_heads, dim, device='npu:0', dtype=dtype).requires_grad_()
    value = torch.randn(total, kv_heads, dim, device='npu:0', dtype=dtype).requires_grad_()
    grad = torch.randn(total, heads, dim, device='npu:0', dtype=dtype)
    cu = [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist())
    return query, key, value, grad, cu


def compare_with_supported_op(cases, dtypes):
    """主对照：twinkle 已支持的 sdpa_attention（补齐布局） vs 本算子（拼接布局）。"""
    from twinkle.kernel.ops.fused_varlen_attention import fused_varlen_attention
    from twinkle.kernel.ops.sdpa_attention.npu import npu_sdpa_attention_forward

    rows = []
    for dtype in dtypes:
        for name, lengths, heads, kv_heads, dim in cases:
            mask = hf_style_mask(lengths, dtype)
            query, key, value, grad, mask = make_padded(lengths, heads, kv_heads, dim, dtype, mask)
            module = _Module(heads // kv_heads)

            def native_fwd():
                return npu_sdpa_attention_forward(module, query, key, value, mask, is_causal=False)

            def native_fwd_bwd():
                out = npu_sdpa_attention_forward(module, query, key, value, mask, is_causal=False)[0]
                return torch.autograd.grad(out, (query, key, value), grad)

            p_query, p_key, p_value, p_grad, cu = make_packed(lengths, heads, kv_heads, dim, dtype)

            def fused_fwd():
                return fused_varlen_attention(p_query, p_key, p_value, cu)

            def fused_fwd_bwd():
                out = fused_varlen_attention(p_query, p_key, p_value, cu)
                return torch.autograd.grad(out, (p_query, p_key, p_value), p_grad)

            row = {
                'case': name,
                'dtype': str(dtype).replace('torch.', ''),
                'lengths': list(lengths),
                'heads': heads,
                'kv_heads': kv_heads,
                'dim': dim,
                'useful_tokens': sum(lengths),
                'padded_tokens': len(lengths) * max(lengths),
                'mask_build_ms': bench(lambda: hf_style_mask(lengths, dtype), repeats=3, iters=10),
                'native_fwd_ms': bench(native_fwd),
                'native_fwd_bwd_ms': bench(native_fwd_bwd),
                'native_peak_mb': peak_memory(native_fwd_bwd),
                'fused_fwd_ms': bench(fused_fwd),
                'fused_fwd_bwd_ms': bench(fused_fwd_bwd),
                'fused_peak_mb': peak_memory(fused_fwd_bwd),
            }
            row['speedup_fwd'] = row['native_fwd_ms'] / row['fused_fwd_ms']
            row['speedup_fwd_bwd'] = row['native_fwd_bwd_ms'] / row['fused_fwd_bwd_ms']
            rows.append(row)
            print(
                f"[sdpa_attention] {row['dtype']:<9} {name:<22} "
                f"词数 {row['useful_tokens']:>5}（补齐后 {row['padded_tokens']:>5}） "
                f"前向 {row['native_fwd_ms']:7.3f} -> {row['fused_fwd_ms']:7.3f} ms "
                f"({row['speedup_fwd']:5.2f}x)  前反向 {row['native_fwd_bwd_ms']:7.3f} -> "
                f"{row['fused_fwd_bwd_ms']:7.3f} ms ({row['speedup_fwd_bwd']:5.2f}x)  "
                f"显存 {row['native_peak_mb']:6.0f} -> {row['fused_peak_mb']:6.0f} MB",
                flush=True)
            del query, key, value, grad, mask, p_query, p_key, p_value, p_grad
            torch.npu.empty_cache()
    return rows


def expand_kv(tensor, groups):
    """按 HF 的做法把 KV 头复制到和 Q 头数一致（附录基线保持与 HF 同一写法）。"""
    if groups == 1:
        return tensor
    heads, length, dim = tensor.shape
    return tensor[:, None].expand(heads, groups, length, dim).reshape(heads * groups, length, dim)


def per_doc_sdpa(query, key, value, cu, groups, causal_flag):
    import torch.nn.functional as F

    outputs = []
    for start, end in zip(cu[:-1], cu[1:]):
        q = query[start:end].transpose(0, 1).unsqueeze(0)
        k = expand_kv(key[start:end].transpose(0, 1), groups).unsqueeze(0)
        v = expand_kv(value[start:end].transpose(0, 1), groups).unsqueeze(0)
        outputs.append(
            F.scaled_dot_product_attention(q, k, v, is_causal=True if causal_flag else False)[0].transpose(0, 1))
    return torch.cat(outputs, 0)


def packed_sdpa(query, key, value, groups):
    import torch.nn.functional as F

    q = query.transpose(0, 1).unsqueeze(0)
    k = expand_kv(key.transpose(0, 1), groups).unsqueeze(0)
    v = expand_kv(value.transpose(0, 1), groups).unsqueeze(0)
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)[0].transpose(0, 1)


def block_diagonal_float_mask(lengths, dtype, device='npu:0'):
    """拼接输入下给现有算子用的掩码：块对角因果，文档之间互相不可见。

    形状是 ``[1, 1, T, T]``，按 transformers 的约定用 0（可见）/ 最小浮点（屏蔽）表示。
    它本身就是要多付的显存与构造开销：``T`` 是拼接后的总长，所以这张表是 ``T×T``。
    """
    total = sum(lengths)
    blocked = torch.finfo(dtype).min
    mask = torch.full((1, 1, total, total), blocked, dtype=dtype, device=device)
    offset = 0
    for length in lengths:
        block = torch.zeros((length, length), dtype=dtype, device=device)
        block.masked_fill_(torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), 1), blocked)
        mask[0, 0, offset:offset + length, offset:offset + length] = block
        offset += length
    return mask


def compare_same_layout(cases, dtypes):
    """最严格的一组：同样的拼接输入、同样的词数，只换内核。

    - 现状：现有 ``sdpa_attention`` + 块对角掩码（这是昇腾上今天唯一能算对拼接输入的写法）；
    - 新增：``fused_varlen_attention`` + 累积长度。
    """
    from twinkle.kernel.ops.fused_varlen_attention import fused_varlen_attention
    from twinkle.kernel.ops.sdpa_attention.npu import npu_sdpa_attention_forward

    rows = []
    for dtype in dtypes:
        for name, lengths, heads, kv_heads, dim in cases:
            query, key, value, grad, cu = make_packed(lengths, heads, kv_heads, dim, dtype)
            module = _Module(heads // kv_heads)
            mask = block_diagonal_float_mask(lengths, dtype)
            total = sum(lengths)
            q4 = query.transpose(0, 1).unsqueeze(0)
            k4 = key.transpose(0, 1).unsqueeze(0)
            v4 = value.transpose(0, 1).unsqueeze(0)
            g4 = grad.unsqueeze(0)

            def native_fwd():
                return npu_sdpa_attention_forward(module, q4, k4, v4, mask, is_causal=False)

            def native_fwd_bwd():
                out = npu_sdpa_attention_forward(module, q4, k4, v4, mask, is_causal=False)[0]
                return torch.autograd.grad(out, (query, key, value), g4)

            def fused_fwd():
                return fused_varlen_attention(query, key, value, cu)

            def fused_fwd_bwd():
                out = fused_varlen_attention(query, key, value, cu)
                return torch.autograd.grad(out, (query, key, value), grad)

            row = {
                'case': name,
                'dtype': str(dtype).replace('torch.', ''),
                'lengths': list(lengths),
                'tokens': total,
                'mask_mb': mask.numel() * mask.element_size() / 2**20,
                'native_fwd_ms': bench(native_fwd),
                'native_fwd_bwd_ms': bench(native_fwd_bwd),
                'native_peak_mb': peak_memory(native_fwd_bwd),
                'fused_fwd_ms': bench(fused_fwd),
                'fused_fwd_bwd_ms': bench(fused_fwd_bwd),
                'fused_peak_mb': peak_memory(fused_fwd_bwd),
            }
            row['speedup_fwd'] = row['native_fwd_ms'] / row['fused_fwd_ms']
            row['speedup_fwd_bwd'] = row['native_fwd_bwd_ms'] / row['fused_fwd_bwd_ms']
            rows.append(row)
            print(
                f"[同布局同词数] {row['dtype']:<9} {name:<22} 词数 {total:>5} "
                f"掩码 {row['mask_mb']:6.1f} MB 前向 {row['native_fwd_ms']:7.3f} -> {row['fused_fwd_ms']:7.3f} ms "
                f"({row['speedup_fwd']:5.2f}x)  前反向 {row['native_fwd_bwd_ms']:7.3f} -> "
                f"{row['fused_fwd_bwd_ms']:7.3f} ms ({row['speedup_fwd_bwd']:5.2f}x)  "
                f"显存 {row['native_peak_mb']:6.0f} -> {row['fused_peak_mb']:6.0f} MB",
                flush=True)
            del query, key, value, grad, mask, q4, k4, v4, g4
            torch.npu.empty_cache()
    return rows


def compare_fixed_shapes(cases, dtypes):
    """附录：固定形状下与逐文档参考实现、整段因果实现的对照。"""
    from twinkle.kernel.ops.fused_varlen_attention import fused_varlen_attention
    from twinkle.kernel.ops.fused_varlen_attention.reference import fused_varlen_attention_reference

    baselines = (
        ('逐文档掩码', lambda q, k, v, cu, groups: fused_varlen_attention_reference(q, k, v, cu)),
        ('逐文档因果', lambda q, k, v, cu, groups: per_doc_sdpa(q, k, v, cu, groups, True)),
        ('整段因果', lambda q, k, v, cu, groups: packed_sdpa(q, k, v, groups)),
    )
    rows = []
    for dtype in dtypes:
        for name, lengths, heads, kv_heads, dim in cases:
            query, key, value, grad, cu = make_packed(lengths, heads, kv_heads, dim, dtype)
            groups = heads // kv_heads

            def fused_fwd():
                return fused_varlen_attention(query, key, value, cu)

            def fused_fwd_bwd():
                out = fused_varlen_attention(query, key, value, cu)
                return torch.autograd.grad(out, (query, key, value), grad)

            row = {
                'case': name,
                'dtype': str(dtype).replace('torch.', ''),
                'lengths': list(lengths),
                'useful_tokens': sum(lengths),
                'fused_fwd_ms': bench(fused_fwd),
                'fused_fwd_bwd_ms': bench(fused_fwd_bwd),
                'fused_peak_mb': peak_memory(fused_fwd_bwd),
            }
            for label, native in baselines:
                row[f'{label}_fwd_ms'] = bench(lambda native=native: native(query, key, value, cu, groups))
                row[f'{label}_fwd_bwd_ms'] = bench(lambda native=native: torch.autograd.grad(
                    native(query, key, value, cu, groups), (query, key, value), grad))
                row[f'speedup_{label}_fwd_bwd'] = row[f'{label}_fwd_bwd_ms'] / row['fused_fwd_bwd_ms']
            rows.append(row)
            print(
                f"[附录]         {row['dtype']:<9} {name:<22} "
                f"前反向融合 {row['fused_fwd_bwd_ms']:7.3f} ms  "
                + '  '.join(f"{label} {row[f'speedup_{label}_fwd_bwd']:5.2f}x" for label, _ in baselines),
                flush=True)
            del query, key, value, grad
            torch.npu.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='bench_fused_varlen_attention.json')
    parser.add_argument('--skip-appendix', action='store_true', help='只跑与 sdpa_attention 的主对照')
    args = parser.parse_args()

    # 主对照用的批次：前 6 个是固定长度（含单段），最后两个是真实训练里的长短混合批次
    main_cases = [
        ('512/300/1024/77', (512, 300, 1024, 77), 8, 8, 128),
        ('2x1024', (1024, 1024), 8, 8, 128),
        ('2x2048', (2048, 2048), 8, 8, 128),
        ('16x64-短', tuple([64] * 16), 8, 8, 128),
        ('1x8192-单段', (8192, ), 8, 8, 128),
        ('gqa-16/4-2x1024', (1024, 1024), 16, 4, 128),
        ('短长混批-8篇', (128, 96, 512, 224, 160, 320, 96, 1024), 8, 8, 128),
        ('长批-4篇', (2048, 1024, 3072, 512), 8, 8, 128),
    ]
    same_layout_cases = [
        ('512/300/1024/77', (512, 300, 1024, 77), 8, 8, 128),
        ('2x1024', (1024, 1024), 8, 8, 128),
        ('2x2048', (2048, 2048), 8, 8, 128),
        ('gqa-16/4-2x1024', (1024, 1024), 16, 4, 128),
        ('短长混批-8篇', (128, 96, 512, 224, 160, 320, 96, 1024), 8, 8, 128),
        ('长批-4篇', (2048, 1024, 3072, 512), 8, 8, 128),
    ]
    appendix_cases = [
        ('2x1024', (1024, 1024), 8, 8, 128),
        ('2x2048', (2048, 2048), 8, 8, 128),
        ('2x4096', (4096, 4096), 8, 8, 128),
        ('1x8192', (8192, ), 8, 8, 128),
        ('gqa-16/4-2x1024', (1024, 1024), 16, 4, 128),
    ]
    dtypes = (torch.bfloat16, torch.float32)
    payload = {'vs_supported_op': compare_with_supported_op(main_cases, dtypes)}
    payload['same_layout'] = compare_same_layout(same_layout_cases, dtypes)
    if not args.skip_appendix:
        payload['fixed_shapes'] = compare_fixed_shapes(appendix_cases, dtypes)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f'\n写入 {args.out}')


if __name__ == '__main__':
    main()
