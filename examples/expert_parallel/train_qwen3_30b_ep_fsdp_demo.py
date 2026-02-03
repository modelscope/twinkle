# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import argparse
import os
from itertools import cycle
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from twinkle.model.transformers.moe import apply_expert_parallel
from twinkle.model.transformers.strategy import NativeFSDPStrategy
from twinkle.utils import DeviceMesh


def _init_dist(args) -> tuple[int, int, torch.device]:
    rank_env = os.environ.get("RANK")
    world_size_env = os.environ.get("WORLD_SIZE")
    local_rank_env = os.environ.get("LOCAL_RANK")

    rank = int(rank_env) if rank_env is not None else args.rank
    world_size = int(
        world_size_env) if world_size_env is not None else args.world_size
    local_rank = int(
        local_rank_env) if local_rank_env is not None else args.local_rank

    if world_size > 1 and rank_env is None and args.rank == 0 and world_size_env is None:
        raise RuntimeError(
            "Distributed training requires torchrun or env:// rendezvous. "
            "Please launch with: torchrun --nproc_per_node <N> examples/moe/train_qwen3_30b_ep_fsdp_demo.py ..."
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=device,
        )
    else:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=0,
            world_size=1,
            device_id=device,
        )
    return rank, world_size, device


def _load_qwen3_config(model_id: str):
    return AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
    )


def _build_ep_device_mesh(world_size: int, ep_size: int) -> DeviceMesh:
    if world_size % ep_size != 0:
        raise ValueError(
            f"world_size({world_size}) must be divisible by ep_size({ep_size})."
        )
    dp_size = world_size // ep_size
    mesh = np.arange(world_size).reshape(dp_size, ep_size)
    return DeviceMesh(
        device_type="cuda",
        mesh=mesh,
        mesh_dim_names=("dp", "ep"),
    )


def _maybe_set_num_layers(config, num_layers: Optional[int]) -> None:
    if num_layers is None:
        return
    if hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = int(num_layers)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-30B EP+FSDP2 training demo")
    parser.add_argument("--model-id", type=str,
                        required=True, help="Local path or HF ID")
    parser.add_argument("--dataset-id", type=str,
                        default=os.environ.get("QWEN3_DATASET_ID", "path/to/alpaca/dataset"))
    parser.add_argument("--num-layers", type=int, default=1,
                        help="Override num_hidden_layers (default: 1)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--ep-size", type=int, default=None,
                        help="Enable EP when > 1")
    parser.add_argument("--rank", type=int, default=0,
                        help="Rank for non-torchrun launches")
    parser.add_argument("--world-size", type=int, default=1,
                        help="World size for non-torchrun launches")
    parser.add_argument("--local-rank", type=int, default=0,
                        help="Local rank for non-torchrun launches")
    args = parser.parse_args()

    rank, world_size, device = _init_dist(args)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True

    config = _load_qwen3_config(args.model_id)
    _maybe_set_num_layers(config, args.num_layers)
    if hasattr(config, "use_cache"):
        config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.train()

    if args.ep_size is not None and args.ep_size > 1:
        device_mesh = _build_ep_device_mesh(world_size, args.ep_size)
        apply_expert_parallel(
            model.model,
            device_mesh,
            config={
                "enabled": True,
                "router_dtype": "fp32",
                "all_to_all": "torch",
                "keep_router_logits": False,
            },
        )
    else:
        device_mesh = DeviceMesh(
            device_type="cuda",
            mesh=np.arange(world_size),
            mesh_dim_names=("fsdp",),
        )

    strategy = NativeFSDPStrategy(
        device_mesh=device_mesh, mixed_precision="bf16", fsdp_config={})
    model.model, _ = strategy.wrap_model(model.model, optimizer=None)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, foreach=False)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def _tokenize_alpaca(example):
        instruction = example.get("instruction") or example.get("prompt") or ""
        inp = example.get("input") or ""
        output = example.get("output") or example.get("response") or ""
        if inp:
            prompt = (
                f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"

        full_text = prompt + output
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_tokens = tokenizer(full_text, add_special_tokens=False)
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens["attention_mask"]

        if tokenizer.eos_token_id is not None:
            input_ids = input_ids + [tokenizer.eos_token_id]
            attention_mask = attention_mask + [1]

        input_ids = input_ids[:args.seq_len]
        attention_mask = attention_mask[:args.seq_len]

        labels = input_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        pad_id = tokenizer.pad_token_id
        if len(input_ids) < args.seq_len:
            pad_len = args.seq_len - len(input_ids)
            input_ids = input_ids + [pad_id] * pad_len
            attention_mask = attention_mask + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    if os.path.exists(args.dataset_id):
        ext = os.path.splitext(args.dataset_id)[1].lstrip(".")
        file_type = {"jsonl": "json", "txt": "text"}.get(ext, ext)
        raw_dataset = load_dataset(
            file_type, data_files=args.dataset_id, split="train")
    else:
        raw_dataset = load_dataset(args.dataset_id, split="train")
    tokenized = raw_dataset.map(
        _tokenize_alpaca, remove_columns=raw_dataset.column_names)

    def _collate_fn(features):
        return {k: torch.tensor([f[k] for f in features], device=device)
                for k in features[0].keys()}

    dataloader = DataLoader(
        tokenized,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=_collate_fn,
        drop_last=True,
    )
    data_iter = cycle(dataloader)

    optimizer.zero_grad(set_to_none=True)
    for step, batch in zip(range(1, args.steps + 1), data_iter):
        outputs = model(**batch)
        loss = outputs.loss / args.grad_accum_steps
        loss.backward()

        if step % args.grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_val = (loss.detach() * args.grad_accum_steps)
            try:
                from torch.distributed.tensor import DTensor  # type: ignore

                if isinstance(loss_val, DTensor):
                    loss_val = loss_val.to_local()
            except Exception:
                pass
            loss_val = loss_val.float().to(device)
            dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
            loss_val = loss_val / world_size
            if rank == 0:
                print(
                    f"[step {step // args.grad_accum_steps}] loss={loss_val.item():.6f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
