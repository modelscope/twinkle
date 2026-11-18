#!/usr/bin/env python3
"""在 A2 NPU 上验证 RolloutResume 的真实进程恢复。

第一进程在 Qwen3-0.6B 完成一轮真实采样、保存 token/logprob 和 LocalEnv workspace
后对自己发送 SIGKILL；父进程重新启动第二进程，恢复同一份状态并继续下一轮采样。
这不是训练收益实验，只验证跨进程断点恢复所依赖的真值没有被重编码或替换。
"""
# ruff: noqa: E402, I001, T201

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (SRC, ROOT / "review" / "gate0"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class _Template:
    max_length = 4096


def _import_runtime():
    # Gate0 installs import-only stubs for optional Ray/Swift pieces. The NPU probe uses
    # the real LocalEnv and TurnLedger, neither of which needs those services.
    import py310_compat  # noqa: F401
    import run_gate0  # noqa: F401

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from twinkle_agentic.async_rl.rollout_resume import ResumeStore
    from twinkle_agentic.envs.localenv import LocalEnv
    from twinkle_agentic.rollout.ledger import TurnLedger

    return torch, AutoModelForCausalLM, AutoTokenizer, ResumeStore, LocalEnv, TurnLedger


def _device(torch):
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    return "npu:0"


def _load_model(torch, model_cls, tokenizer_cls, model_path, device):
    tokenizer = tokenizer_cls.from_pretrained(model_path, trust_remote_code=True)
    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    return tokenizer, model


def _prompt_ids(tokenizer, device, text):
    messages = [{"role": "user", "content": text}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, return_tensors="pt").input_ids.to(device)


def _sample(torch, model, input_ids, *, seed, max_new_tokens=24):
    torch.manual_seed(seed)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=0,
        )
    prompt_length = int(input_ids.shape[1])
    generated = output.sequences[0, prompt_length:].detach().cpu().tolist()
    logprobs = []
    for token, score in zip(generated, output.scores, strict=True):
        values = torch.log_softmax(score[0].float(), dim=-1)
        logprobs.append(float(values[token].detach().cpu()))
    return generated, logprobs


def _first_phase(args):
    torch, model_cls, tokenizer_cls, ResumeStore, LocalEnv, TurnLedger = _import_runtime()
    device = _device(torch)
    tokenizer, model = _load_model(torch, model_cls, tokenizer_cls, args.model, device)
    input_ids = _prompt_ids(tokenizer, device, "Write one short sentence about durable rollouts.")
    generated, logprobs = _sample(torch, model, input_ids, seed=args.seed)

    state_key = f"{args.run_id}/group/0/attempt/0/member/0"
    ledger = TurnLedger(_Template(), label="a2-npu")
    prompt = input_ids[0].detach().cpu().tolist()
    ledger._pif = {
        "input_ids": prompt + generated,
        "labels": [-100] * len(prompt) + generated,
        "completion_mask": [0] * len(prompt) + [1] * len(generated),
        "response_positions": list(range(len(prompt) - 1, len(prompt) + len(generated) - 1)),
        "messages": [
            {"role": "user", "content": "durable resume"},
            {"role": "assistant", "content": tokenizer.decode(generated)},
        ],
    }
    ledger._logprobs = [[(token, value)] for token, value in zip(generated, logprobs, strict=True)]
    ledger._turns = 1

    env_root = Path(args.root) / "localenv"
    env = LocalEnv(root=str(env_root), memory_limit_gb=None)
    env.reset("episode-0")
    env.step("episode-0", "write_file", {"path": "turn.txt", "content": "turn-1-durable"})
    env_state = env.checkpoint_episode("episode-0", state_key)
    env.close("episode-0")

    store = ResumeStore(Path(args.root) / "resume", args.run_id)
    store.save_object(
        "npu-probe",
        "turn-1",
        {
            "format": "twinkle.npu-resume-probe.v1",
            "state_key": state_key,
            "input_ids": prompt,
            "generated": generated,
            "logprobs": logprobs,
            "ledger": ledger.state_dict(),
            "env_state": env_state,
            "model": args.model,
            "seed": args.seed,
        },
    )
    (Path(args.root) / "first-phase-ready.json").write_text(
        json.dumps({"state_key": state_key, "tokens": len(generated), "npu": device}, ensure_ascii=False),
        encoding="utf-8",
    )
    sys.stdout.flush()
    os.kill(os.getpid(), 9)


def _second_phase(args):
    torch, model_cls, tokenizer_cls, ResumeStore, LocalEnv, TurnLedger = _import_runtime()
    store = ResumeStore(Path(args.root) / "resume", args.run_id)
    saved = store.load_object("npu-probe", "turn-1")
    ledger = TurnLedger(_Template(), label="a2-npu")
    ledger.load_state_dict(saved["ledger"])
    assert ledger.input_feature["input_ids"] == saved["input_ids"] + saved["generated"]
    assert len(ledger.logprobs) == len(saved["generated"])

    env = LocalEnv(root=str(Path(args.root) / "localenv"), memory_limit_gb=None)
    env.restore_episode("episode-0", saved["state_key"], saved["env_state"])
    tool_result = env.step("episode-0", "read_file", {"path": "turn.txt"})
    assert tool_result.observation == "turn-1-durable"
    env.close("episode-0")

    device = _device(torch)
    tokenizer, model = _load_model(torch, model_cls, tokenizer_cls, args.model, device)
    restored_ids = torch.tensor([ledger.input_feature["input_ids"]], dtype=torch.long, device=device)
    second, second_logprobs = _sample(torch, model, restored_ids, seed=args.seed, max_new_tokens=16)
    report = {
        "format": "twinkle.npu-resume-probe-result.v1",
        "ok": True,
        "npu": device,
        "preserved_tokens": len(saved["generated"]),
        "first_logprobs_finite": all(abs(value) < float("inf") for value in saved["logprobs"]),
        "restored_exact": True,
        "tool_observation": tool_result.observation,
        "second_tokens": len(second),
        "second_logprobs_finite": all(abs(value) < float("inf") for value in second_logprobs),
        "second_text": tokenizer.decode(second),
    }
    (Path(args.root) / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _parent(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    first = subprocess.run(
        [
            sys.executable,
            __file__,
            "--phase",
            "first",
            "--root",
            args.root,
            "--model",
            args.model,
            "--run-id",
            args.run_id,
            "--seed",
            str(args.seed),
        ],
        check=False,
    )
    if first.returncode not in (-9, 137):
        raise SystemExit(f"first phase must die by SIGKILL, got {first.returncode}")
    second = subprocess.run(
        [
            sys.executable,
            __file__,
            "--phase",
            "second",
            "--root",
            args.root,
            "--model",
            args.model,
            "--run-id",
            args.run_id,
            "--seed",
            str(args.seed),
        ],
        check=False,
    )
    raise SystemExit(second.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("parent", "first", "second"), default="parent")
    parser.add_argument("--root", default="/workspace/rollout-resume-npu-run")
    parser.add_argument("--model", default="/workspace/shared_assets/models/Qwen/Qwen3-0.6B")
    parser.add_argument("--run-id", default="rollout-resume-a2")
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    if args.phase == "first":
        return _first_phase(args)
    if args.phase == "second":
        return _second_phase(args)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
