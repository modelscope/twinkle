#!/usr/bin/env python3
"""端到端训练对照：同一个脚本，分别放在 main 工作区与本分支工作区里跑。

它只依赖 ``src/``，不 import 本目录的任何东西，所以可以直接拷到 main 的工作区里跑基线：

    # 基线：main 的 src（补齐布局 + twinkle 已支持的 sdpa_attention）
    SRC=/path/to/main/src
    PYTHONPATH=$SRC python e2e_two_checkout.py --layout padded         --lens 128,96,512,224,160,320,96,384 --steps 10

    # 本分支：拼接布局 + 本算子（main 上这一路会 ModuleNotFoundError）
    SRC=/path/to/branch/src
    PYTHONPATH=$SRC python e2e_two_checkout.py --layout packed         --lens 128,96,512,224,160,320,96,384 --steps 10

`--lens` 写死每步的文档长度（推荐，可复核）；不写它就按 `--dist` 随机抽样——注意随机抽样的
倍数会被批次里最长那篇的平方放大，单次抽样不能当结论。

输出一行 JSON：单步耗时、有效 token/s、峰值显存、逐点 loss 曲线。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import torch
import torch.nn.functional as F


def state_digest(model):
    """初始化权重指纹：两边必须一致，否则后面的对比没有意义。"""
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def sample_lengths(args, generator):
    lengths = []
    for _ in range(args.docs):
        if args.dist == 'uniform':
            value = int(torch.randint(args.min_len, args.max_len + 1, (1, ), generator=generator))
        else:
            u1 = float(torch.rand((), generator=generator).clamp_min(1e-9))
            u2 = float(torch.rand((), generator=generator))
            gauss = (-2.0 * math.log(u1))**0.5 * math.cos(2.0 * math.pi * u2)
            value = int(round(args.min_len * math.exp(args.sigma * gauss)))
            value = max(args.min_len, min(args.max_len, value))
        lengths.append(value)
    return lengths


def build_batch(args, lengths, vocab, generator, device):
    docs = [torch.randint(0, vocab, (length, ), generator=generator) for length in lengths]
    targets = []
    for doc in docs:
        shifted = torch.full_like(doc, -100)
        shifted[:-1] = doc[1:]
        targets.append(shifted)
    supervised = int(sum((target != -100).sum() for target in targets))
    if args.layout == 'packed':
        input_ids = torch.cat(docs).unsqueeze(0).to(device)
        position_ids = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0).to(device)
        cumulative = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(lengths), 0).tolist()), dtype=torch.int32, device=device)
        inputs = dict(
            input_ids=input_ids,
            position_ids=position_ids,
            cu_seq_lens_q=cumulative,
            cu_seq_lens_k=cumulative,
            max_length_q=max(lengths),
            max_length_k=max(lengths))
        physical = int(input_ids.numel())
    else:
        widest = max(lengths)
        input_ids = torch.zeros((len(docs), widest), dtype=torch.long)
        attention_mask = torch.zeros((len(docs), widest), dtype=torch.long)
        position_ids = torch.zeros((len(docs), widest), dtype=torch.long)
        padded_targets = torch.full((len(docs), widest), -100, dtype=torch.long)
        for index, doc in enumerate(docs):
            length = len(doc)
            input_ids[index, :length] = doc
            attention_mask[index, :length] = 1
            position_ids[index, :length] = torch.arange(length)
            padded_targets[index, :length] = targets[index]
        targets = padded_targets
        inputs = dict(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            position_ids=position_ids.to(device))
        physical = int(input_ids.numel())
    flat_targets = torch.cat(targets) if isinstance(targets, list) else targets.reshape(-1)
    return inputs, flat_targets.to(device), supervised, physical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='/workspace/shared_assets/models/Qwen/Qwen3-0.6B')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--layout', default='padded', choices=('padded', 'packed'))
    parser.add_argument('--docs', type=int, default=6)
    parser.add_argument('--min-len', type=int, default=96)
    parser.add_argument('--max-len', type=int, default=1024)
    parser.add_argument('--dist', default='lognormal', choices=('uniform', 'lognormal'))
    parser.add_argument('--sigma', type=float, default=0.9)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--lens', default='', help='写死每步的文档长度，逗号分隔；定了它就不随机抽样')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--ce-chunk', type=int, default=512)
    args = parser.parse_args()

    import torch_npu  # noqa: F401
    torch.npu.set_device('npu:0')
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    summary = dict(src=os.environ.get('TWINKLE_SRC', '?'), layout=args.layout, docs=args.docs, steps=args.steps)
    if args.layout == 'packed':
        from twinkle.kernel.ops.fused_varlen_attention import ATTENTION_NAME, install_varlen_attention
        from twinkle.kernel.ops.fused_varlen_attention.npu import npu_varlen_attention_forward
    from twinkle.kernel.ops.sdpa_attention.npu import npu_sdpa_attention_forward

    dtype = getattr(torch, args.dtype)
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, torch_dtype=dtype, local_files_only=True)
    model.config.use_cache = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    model = model.to(dtype=dtype).to('npu:0').train()
    if args.layout == 'packed':
        install_varlen_attention(model, ATTENTION_NAME, npu_varlen_attention_forward)
        model.config._attn_implementation = ATTENTION_NAME
    else:
        ALL_ATTENTION_FUNCTIONS['sdpa'] = npu_sdpa_attention_forward
        model.config._attn_implementation = 'sdpa'

    summary['init_digest'] = state_digest(model)
    vocab = config.vocab_size
    generator = torch.Generator().manual_seed(args.seed)
    batches = []
    all_lengths = []
    fixed = [int(item) for item in args.lens.split(',') if item.strip()] if args.lens else None
    for _ in range(args.warmup + args.steps):
        lengths = list(fixed) if fixed else sample_lengths(args, generator)
        all_lengths.append(lengths)
        batches.append(build_batch(args, lengths, vocab, generator, 'npu:0'))
    summary['lengths'] = all_lengths
    summary['physical_tokens'] = sum(batch[3] for batch in batches[args.warmup:])
    summary['supervised_tokens'] = sum(batch[2] for batch in batches[args.warmup:])

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    def step(batch):
        inputs, targets, supervised, _ = batch
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
        return loss.detach().item()

    try:
        for batch in batches[:args.warmup]:
            step(batch)
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        started = time.perf_counter()
        losses = [step(batch) for batch in batches[args.warmup:]]
        torch.npu.synchronize()
        elapsed = time.perf_counter() - started
    except torch.OutOfMemoryError:
        summary['result'] = 'OUT_OF_MEMORY'
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return
    summary['result'] = 'ok'
    summary['step_ms'] = round(elapsed * 1000 / args.steps, 1)
    summary['tokens_per_s'] = round(summary['supervised_tokens'] / elapsed, 1)
    summary['peak_mb'] = round(torch.npu.max_memory_allocated() / 2**20, 1)
    summary['loss_curve'] = [round(value, 6) for value in losses]
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
