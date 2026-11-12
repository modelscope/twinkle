#!/usr/bin/env python3
"""端到端对照：padded 原生 SDPA vs packed 融合变长注意力（自包含，不依赖任何实验框架）。

同一批变长文档、同一初始化权重、同一优化器与步数，只改变「布局 + 注意力实现」：

- ``padded``：每步把 4 篇文档 padding 成矩阵，走 twinkle 已支持的 ``sdpa_attention``；
- ``packed``：把同样 4 篇文档拼成一条序列（position_ids 在文档边界归零、
  ``cu_seq_lens_q`` 显式传入），走 ``fused_varlen_attention``。

输出：逐点 loss 曲线、有效 token 吞吐、每次更新耗时、峰值显存、物理 token 数。

用法：
    python bench_e2e.py                      # 随机小模型，几秒出结果
    python bench_e2e.py --model /models/Qwen3-0.6B --steps 15
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
import torch
import torch.nn.functional as F
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, help='本地 checkpoint；省略则用随机小模型')
    parser.add_argument('--device', default='npu:0')
    parser.add_argument('--dtype', default='bfloat16', choices=('float32', 'bfloat16'))
    parser.add_argument('--docs-per-step', type=int, default=4)
    parser.add_argument('--min-len', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument(
        '--length-dist',
        default='uniform',
        choices=('uniform', 'lognormal'),
        help='uniform：长度在 [min,max] 均匀；lognormal：以 min 为中位数的长尾，贴近真实 SFT 语料')
    parser.add_argument('--length-sigma', type=float, default=0.9, help='lognormal 的 sigma（越大尾巴越长）')
    parser.add_argument(
        '--layouts', default='padded,packed', help='要跑的布局，逗号分隔；padded 单独跑可用于确认拼接布局能放下而 padding 布局放不下的批量')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--ce-chunk', type=int, default=512, help='按 token 分块算交叉熵，避免全词表 logits 撑爆显存')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--out', type=Path, default=Path('bench_e2e.json'))
    parser.add_argument('--profile', action='store_true', help='用设备事件统计注意力内核占单步的比例')
    return parser.parse_args()


def build_model(args, dtype, vocab_size, seed):
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(seed)
    if args.model is None:
        config = AutoConfig.for_model(
            'qwen3',
            vocab_size=vocab_size,
            hidden_size=256,
            intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_hidden_layers=4,
            max_position_embeddings=8192,
            attention_dropout=0.0,
            use_cache=False)
        model = AutoModelForCausalLM.from_config(config)
    else:
        config = AutoConfig.from_pretrained(args.model, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, config=config, torch_dtype=dtype, local_files_only=True)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model.to(dtype=dtype).to(args.device).train()


def sample_docs(rng, vocab_size, count, min_len, max_len, dist='uniform', sigma=0.9):
    """按长度分布抽一批文档；lognormal 用 Box-Muller 从同一个 generator 取随机数，保证可复现。"""
    docs = []
    for _ in range(count):
        if dist == 'uniform':
            length = int(torch.randint(min_len, max_len + 1, (1, ), generator=rng))
        else:
            u1 = float(torch.rand((), generator=rng).clamp_min(1e-9))
            u2 = float(torch.rand((), generator=rng))
            gauss = (-2.0 * math.log(u1))**0.5 * math.cos(2.0 * math.pi * u2)
            length = int(round(min_len * math.exp(sigma * gauss)))
            length = max(min_len, min(max_len, length))
        docs.append(torch.randint(0, vocab_size, (length, ), generator=rng))
    return docs


def make_batch(docs, layout, device, pad_token=0):
    """两种布局的输入与监督标签；标签按文档边界右移一位，边界与 padding 记为 -100。"""
    targets = []
    for doc in docs:
        shifted = torch.full_like(doc, -100)
        shifted[:-1] = doc[1:]
        targets.append(shifted)
    lengths = [len(doc) for doc in docs]
    supervised = int(sum((t != -100).sum() for t in targets))

    if layout == 'packed':
        input_ids = torch.cat(docs).unsqueeze(0)
        position_ids = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
        cu = [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist())
        boundaries = torch.tensor(cu, dtype=torch.int32, device=device)
        inputs = dict(
            input_ids=input_ids.to(device),
            position_ids=position_ids.to(device),
            cu_seq_lens_q=boundaries,
            cu_seq_lens_k=boundaries,
            max_length_q=max(lengths),
            max_length_k=max(lengths))
    else:
        widest = max(lengths)
        input_ids = torch.full((len(docs), widest), pad_token, dtype=docs[0].dtype)
        attention_mask = torch.zeros((len(docs), widest), dtype=docs[0].dtype)
        position_ids = torch.zeros((len(docs), widest), dtype=torch.long)
        padded_targets = torch.full((len(docs), widest), -100, dtype=torch.long)
        for index, (doc, target) in enumerate(zip(docs, targets)):
            input_ids[index, :len(doc)] = doc
            attention_mask[index, :len(doc)] = 1
            position_ids[index, :len(doc)] = torch.arange(len(doc))
            padded_targets[index, :len(doc)] = target
        targets = list(padded_targets)
        inputs = dict(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            position_ids=position_ids.to(device))
    return inputs, torch.cat(targets).to(device), supervised


def install_supported_sdpa(model):
    """补齐侧用 twinkle 已经支持的 ``sdpa_attention``：昇腾上 kernelize 装的就是这个算子。"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    from twinkle.kernel.ops.sdpa_attention.npu import npu_sdpa_attention_forward
    ALL_ATTENTION_FUNCTIONS['sdpa'] = npu_sdpa_attention_forward
    model.config._attn_implementation = 'sdpa'


def install_packed_attention(model):
    """把融合变长注意力挂到 flash_attention_2 分发名上，并让模型按该实现名分派。

    不伪造 transformers 的 FA2 可用性：模型先按 sdpa 构造，再切换实现名。
    """
    from twinkle.kernel.ops.fused_varlen_attention import ATTENTION_NAME, install_varlen_attention
    from twinkle.kernel.ops.fused_varlen_attention.npu import npu_varlen_attention_forward
    install_varlen_attention(model, ATTENTION_NAME, npu_varlen_attention_forward)
    model.config._attn_implementation = ATTENTION_NAME


def instrument_attention(model):
    """用设备事件把每层注意力内核框起来，用来回答「单步里内核到底占多少」。"""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    impl = model.config._attn_implementation
    original = ALL_ATTENTION_FUNCTIONS[impl]
    events = []

    def timed(module, query, key, value, *args, **kwargs):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        output = original(module, query, key, value, *args, **kwargs)
        end.record()
        events.append((start, end))
        return output

    ALL_ATTENTION_FUNCTIONS[impl] = timed
    return events


def run(args, layout, init_state, batches, dtype, vocab_size):
    model = build_model(args, dtype, vocab_size, args.seed)
    if init_state is not None:
        model.load_state_dict(init_state)
    if layout == 'packed':
        install_packed_attention(model)
    else:
        install_supported_sdpa(model)
    collected = instrument_attention(model) if args.profile else None
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    def step(batch):
        inputs, targets, supervised = batch
        logits = model(**inputs, use_cache=False, return_dict=True).logits
        flat = logits.reshape(-1, logits.shape[-1])
        loss = None
        for start in range(0, flat.shape[0], args.ce_chunk):
            piece = F.cross_entropy(
                flat[start:start + args.ce_chunk].float(),
                targets[start:start + args.ce_chunk],
                ignore_index=-100,
                reduction='sum')
            loss = piece if loss is None else loss + piece
        loss = loss / supervised
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return loss.detach()

    for batch in batches[:args.warmup]:
        step(batch)
    torch.npu.synchronize()
    torch.npu.reset_peak_memory_stats()
    if collected is not None:
        collected.clear()
    started = time.perf_counter()
    losses = [float(step(batch)) for batch in batches[args.warmup:]]
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    supervised_tokens = sum(batch[2] for batch in batches[args.warmup:])
    kernel_ms = None
    if collected is not None and collected:
        kernel_ms = sum(start.elapsed_time(end) for start, end in collected) / len(losses)
    return dict(
        layout=layout,
        loss_curve=losses,
        steps=len(losses),
        window_seconds=elapsed,
        mean_update_ms=elapsed * 1000 / len(losses),
        supervised_tokens_per_second=supervised_tokens / elapsed,
        peak_memory_allocated_bytes=torch.npu.max_memory_allocated(),
        attention_kernel_ms_per_step=kernel_ms,
        attention_kernel_share=(kernel_ms / (elapsed * 1000 / len(losses)) if kernel_ms else None),
        initial_state_sha256=state_digest(init_state) if init_state is not None else None)


def state_digest(state):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in state.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    args = parse_args()
    if not args.device.startswith('npu:'):
        raise SystemExit('本脚本测量融合变长注意力，必须在 NPU 上运行')
    import torch_npu  # noqa: F401
    torch.npu.set_device(args.device)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed_all(args.seed)
    dtype = getattr(torch, args.dtype)

    probe = build_model(args, dtype, 1024 if args.model is None else 0, args.seed)
    vocab_size = probe.config.vocab_size
    del probe
    generator = torch.Generator().manual_seed(args.seed)
    batches = []
    for _ in range(args.warmup + args.steps):
        docs = sample_docs(generator, vocab_size, args.docs_per_step, args.min_len, args.max_len, args.length_dist,
                           args.length_sigma)
        batches.append((docs, [len(d) for d in docs]))
    layouts = [item.strip() for item in args.layouts.split(',') if item.strip()]
    prepared = {}
    for layout in layouts:
        prepared[layout] = [make_batch(docs, layout, args.device) for docs, _ in batches]

    reference = build_model(args, dtype, vocab_size, args.seed)
    init_state = copy.deepcopy({k: v.detach().clone() for k, v in reference.state_dict().items()})
    del reference
    results = {}
    for layout in layouts:
        results[layout] = run(args, layout, init_state, prepared[layout], dtype, vocab_size)
        print(
            f"[{args.dtype}] {layout:<6} loss[-1]={results[layout]['loss_curve'][-1]:.6f} "
            f"supervised_tok/s={results[layout]['supervised_tokens_per_second']:.1f} "
            f"step={results[layout]['mean_update_ms']:.1f}ms "
            f"peak={results[layout]['peak_memory_allocated_bytes'] / 2**20:.0f}MB" +
            (f"  attn_kernel={results[layout]['attention_kernel_share']:.1%}"
             if results[layout]['attention_kernel_share'] else ''),
            flush=True)

    padded, packed = results.get('padded'), results.get('packed')
    diffs = [abs(a - b) for a, b in zip(padded['loss_curve'], packed['loss_curve'])] if padded and packed else []
    summary = dict(
        dtype=args.dtype,
        model=str(args.model) if args.model else 'random-tiny-qwen3',
        docs_per_step=args.docs_per_step,
        length_dist=dict(kind=args.length_dist, sigma=args.length_sigma),
        lengths=[lengths for _, lengths in batches],
        physical_tokens=dict(
            padded=sum(max(lengths) * args.docs_per_step for _, lengths in batches),
            packed=sum(sum(lengths) for _, lengths in batches)),
        loss_curve_max_abs_difference=max(diffs) if diffs else None,
        supervised_tokens_per_second={
            layout: run_result['supervised_tokens_per_second']
            for layout, run_result in results.items()
        },
        speedup=(packed['supervised_tokens_per_second']
                 / padded['supervised_tokens_per_second'] if padded and packed else None),
        peak_memory_mb={
            layout: run_result['peak_memory_allocated_bytes'] / 2**20
            for layout, run_result in results.items()
        },
        attention_kernel_share={
            layout: run_result['attention_kernel_share']
            for layout, run_result in results.items()
        },
        runs=results)
    print(json.dumps({k: v for k, v in summary.items() if k != 'runs'}, ensure_ascii=False, indent=2))
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'写入 {args.out}')


if __name__ == '__main__':
    main()
