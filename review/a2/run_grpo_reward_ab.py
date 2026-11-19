#!/usr/bin/env python3
"""在 A2 上用官方 GSM8K 数据比较故障恢复的训练收益。

两次运行使用同一 Qwen3-0.6B、同一 LoRA 初态、同一 GSM8K train 子集、同一 test
子集和同一采样预算。普通恢复在故障点丢掉当前完整组；durable 保留该组并训练。
奖励使用仓库标准 GSM8KAccuracyReward + GSM8KFormatReward。
"""
# ruff: noqa: E402, I001, T201

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "review" / "gate0"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import py310_compat  # noqa: E402,F401
import run_gate0  # noqa: E402,F401

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from twinkle.advantage.grpo import GRPOAdvantage  # noqa: E402
from twinkle.loss.grpo import GRPOLoss  # noqa: E402
from twinkle.reward import GSM8KAccuracyReward, GSM8KFormatReward  # noqa: E402


def _device():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    return torch.device("npu:0")


def _load_jsonl(path: str, limit: int, offset: int = 0) -> list[dict[str, str]]:
    records = []
    with open(path, encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index < offset:
                continue
            if line.strip():
                records.append(json.loads(line))
            if limit > 0 and len(records) >= limit:
                break
    if not records:
        raise ValueError(f"no GSM8K records loaded from {path}")
    return records


def _ground_truth(answer: str) -> str:
    if "####" not in answer:
        raise ValueError("GSM8K answer has no #### marker")
    return answer.rsplit("####", 1)[1].strip().replace(",", "")


def _render(tokenizer, question: str, *, max_length: int) -> str:
    messages = [
        {
            "role": "system",
            "content": "Solve the math problem with brief reasoning. Put the final answer within \\boxed{}.",
        },
        {"role": "user", "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
            truncation=True,
            max_length=max_length,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _tokenized_prompt(tokenizer, device, question: str, *, max_length: int):
    rendered = _render(tokenizer, question, max_length=max_length)
    encoded = tokenizer(rendered, return_tensors="pt", truncation=True, max_length=max_length)
    return {key: value.to(device) for key, value in encoded.items()}


def _tokenized_prompt_batch(tokenizer, device, questions: list[str], *, max_length: int):
    rendered = [_render(tokenizer, question, max_length=max_length) for question in questions]
    encoded = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def _trajectory(question: str, completion: str, target: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": completion},
        ],
        "user_data": [["ground_truth", target]],
    }


def _rewards(question: str, texts: list[str], target: str, *, format_weight: float) -> tuple[list[float], dict]:
    trajectories = [_trajectory(question, text, target) for text in texts]
    accuracy = GSM8KAccuracyReward()(trajectories)
    formatting = GSM8KFormatReward()(trajectories)
    rewards = [acc + format_weight * fmt for acc, fmt in zip(accuracy, formatting, strict=True)]
    return rewards, {
        "accuracy": sum(accuracy) / len(accuracy),
        "format": sum(formatting) / len(formatting),
        "reward": sum(rewards) / len(rewards),
    }


def _generate_group(model, tokenizer, device, prompt, *, group_size: int, max_new_tokens: int):
    prompt_ids = prompt["input_ids"]
    attention_mask = prompt["attention_mask"]
    with torch.no_grad():
        output = model.generate(
            input_ids=prompt_ids.repeat(group_size, 1),
            attention_mask=attention_mask.repeat(group_size, 1),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    prompt_length = int(prompt_ids.shape[1])
    generated = output.sequences[:, prompt_length:]
    old_logps: list[list[float]] = []
    for row in range(group_size):
        row_logps = []
        for step, score in enumerate(output.scores):
            token = int(generated[row, step])
            values = torch.log_softmax(score[row].float(), dim=-1)
            row_logps.append(float(values[token].detach().cpu()))
        old_logps.append(row_logps)
    texts = [tokenizer.decode(row, skip_special_tokens=True) for row in generated]
    return generated, old_logps, texts


def _train_group(model, optimizer, prompt, generated, old_logps, rewards):
    group_size = generated.shape[0]
    prompt_ids = prompt["input_ids"].repeat(group_size, 1)
    attention_mask = prompt["attention_mask"].repeat(group_size, 1)
    input_ids = torch.cat((prompt_ids, generated), dim=1)
    attention_mask = torch.cat((attention_mask, torch.ones_like(generated)), dim=1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.shape[1] :] = generated
    completion_mask = torch.zeros_like(input_ids)
    completion_mask[:, prompt_ids.shape[1] :] = 1

    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits.float()
    shift_logits = logits[:, :-1]
    shift_labels = labels[:, 1:]
    gathered = torch.log_softmax(shift_logits, dim=-1).gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    logps = torch.zeros_like(labels, dtype=torch.float32)
    logps[:, 1:] = gathered

    advantages = GRPOAdvantage()._per_sequence(rewards, num_generations=group_size, scale="group")
    result = GRPOLoss(epsilon=0.2, beta=0.0)(
        {"labels": labels, "completion_mask": completion_mask},
        {"logps": logps},
        old_logps=old_logps,
        advantages=advantages,
    )
    loss = result["loss"]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "nontrivial": bool(torch.any(torch.abs(advantages) > 1e-8)),
    }


def _evaluate(
    model,
    tokenizer,
    device,
    records,
    *,
    max_new_tokens: int,
    prompt_max_length: int,
    format_weight: float,
    eval_batch_size: int,
) -> dict:
    model.eval()
    samples = []
    with torch.no_grad():
        for offset in range(0, len(records), eval_batch_size):
            batch = records[offset : offset + eval_batch_size]
            prompt = _tokenized_prompt_batch(
                tokenizer,
                device,
                [record["question"] for record in batch],
                max_length=prompt_max_length,
            )
            output = model.generate(
                **prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            prompt_length = int(prompt["input_ids"].shape[1])
            for row, record in enumerate(batch):
                generated = output.sequences[row, prompt_length:]
                text = tokenizer.decode(generated, skip_special_tokens=True)
                target = _ground_truth(record["answer"])
                _, metrics = _rewards(record["question"], [text], target, format_weight=format_weight)
                samples.append({**metrics, "completion": text})
    return {
        "accuracy": sum(sample["accuracy"] for sample in samples) / len(samples),
        "format": sum(sample["format"] for sample in samples) / len(samples),
        "reward": sum(sample["reward"] for sample in samples) / len(samples),
        "samples": samples,
    }


def _reset_lora(model, initial: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in initial:
                parameter.copy_(initial[name])


def _run_variant(
    model,
    tokenizer,
    device,
    records,
    *,
    durable: bool,
    attempts: int,
    fault_period: int,
    group_size: int,
    max_new_tokens: int,
    prompt_max_length: int,
    format_weight: float,
    lr: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    random.seed(seed)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr)
    steps = 0
    nontrivial = 0
    lost_groups = 0
    metric_sums = {"accuracy": 0.0, "format": 0.0, "reward": 0.0}
    sampled = 0
    losses = []
    started = time.perf_counter()
    variant = "durable" if durable else "cursor_restart"
    for attempt in range(attempts):
        record = records[attempt % len(records)]
        target = _ground_truth(record["answer"])
        prompt = _tokenized_prompt(tokenizer, device, record["question"], max_length=prompt_max_length)
        generated, old_logps, texts = _generate_group(
            model, tokenizer, device, prompt, group_size=group_size, max_new_tokens=max_new_tokens
        )
        rewards, metrics = _rewards(record["question"], texts, target, format_weight=format_weight)
        sampled += len(texts)
        for key in metric_sums:
            metric_sums[key] += metrics[key] * len(texts)
        is_fault = fault_period > 0 and (attempt + 1) % fault_period == 0
        if is_fault and not durable:
            lost_groups += 1
            continue
        trained = _train_group(model, optimizer, prompt, generated, old_logps, rewards)
        steps += 1
        nontrivial += int(trained["nontrivial"])
        losses.append(trained["loss"])
        if (attempt + 1) % 8 == 0 or attempt + 1 == attempts:
            print(
                f"[{variant}] {attempt + 1}/{attempts} attempts, {steps} steps, {lost_groups} lost groups",
                file=sys.stderr,
                flush=True,
            )
    torch.npu.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "variant": variant,
        "seed": seed,
        "attempted_groups": attempts,
        "train_steps": steps,
        "lost_groups": lost_groups,
        "sampled_sequences": sampled,
        "nontrivial_advantage_groups": nontrivial,
        "sample_accuracy": metric_sums["accuracy"] / max(sampled, 1),
        "sample_format": metric_sums["format"] / max(sampled, 1),
        "sample_reward": metric_sums["reward"] / max(sampled, 1),
        "mean_loss": sum(losses) / max(len(losses), 1),
        "elapsed_s": elapsed,
    }


def run(args) -> dict:
    if args.deterministic_control:
        torch.use_deterministic_algorithms(True, warn_only=True)
    train = _load_jsonl(args.train_jsonl, args.train_limit, args.train_offset)
    test = _load_jsonl(args.test_jsonl, args.eval_limit, args.eval_offset)
    device = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to(device)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.0 if args.deterministic_control else 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters() if "lora_" in name}
    eval_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "prompt_max_length": args.prompt_max_length,
        "format_weight": args.format_weight,
        "eval_batch_size": args.eval_batch_size,
    }
    before = None if args.skip_before else _evaluate(model, tokenizer, device, test, **eval_kwargs)
    variant_kwargs = {
        "attempts": args.attempts,
        "fault_period": args.fault_period,
        "group_size": args.group_size,
        "lr": args.lr,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
        "prompt_max_length": args.prompt_max_length,
        "format_weight": args.format_weight,
    }
    _reset_lora(model, initial)
    baseline = _run_variant(model, tokenizer, device, train, durable=False, **variant_kwargs)
    baseline_eval = _evaluate(model, tokenizer, device, test, **eval_kwargs)
    _reset_lora(model, initial)
    durable = _run_variant(model, tokenizer, device, train, durable=True, **variant_kwargs)
    durable_eval = _evaluate(model, tokenizer, device, test, **eval_kwargs)
    return {
        "ok": True,
        "dataset": "openai/grade-school-math GSM8K official JSONL",
        "seed": args.seed,
        "config": {
            "train_limit": args.train_limit,
            "train_offset": args.train_offset,
            "eval_limit": args.eval_limit,
            "eval_offset": args.eval_offset,
            "attempts": args.attempts,
            "fault_period": args.fault_period,
            "group_size": args.group_size,
            "max_new_tokens": args.max_new_tokens,
            "prompt_max_length": args.prompt_max_length,
            "format_weight": args.format_weight,
            "eval_batch_size": args.eval_batch_size,
            "lora_rank": args.lora_rank,
            "lr": args.lr,
            "skip_before": args.skip_before,
            "deterministic_control": args.deterministic_control,
        },
        "train_records": len(train),
        "eval_records": len(test),
        "model": args.model,
        "device": str(device),
        "before": before,
        "baseline": baseline,
        "baseline_eval": baseline_eval,
        "durable": durable,
        "durable_eval": durable_eval,
        "accuracy_delta": durable_eval["accuracy"] - baseline_eval["accuracy"],
        "reward_delta": durable_eval["reward"] - baseline_eval["reward"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/workspace/shared_assets/models/Qwen/Qwen3-0.6B")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--test-jsonl", required=True)
    parser.add_argument("--train-limit", type=int, default=32)
    parser.add_argument("--train-offset", type=int, default=0)
    parser.add_argument("--eval-limit", type=int, default=32)
    parser.add_argument("--eval-offset", type=int, default=0)
    parser.add_argument("--attempts", type=int, default=32)
    parser.add_argument("--fault-period", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--prompt-max-length", type=int, default=768)
    parser.add_argument("--format-weight", type=float, default=0.2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--skip-before", action="store_true")
    parser.add_argument("--deterministic-control", action="store_true")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = run(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
