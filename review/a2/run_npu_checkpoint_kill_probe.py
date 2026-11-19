#!/usr/bin/env python3
"""SIGKILL a real Qwen LoRA training process around checkpoint boundaries."""

# ruff: noqa: E402, I001, T201

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "review" / "gate0", ROOT / "review" / "a2"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import py310_compat  # noqa: E402,F401
import run_gate0  # noqa: E402,F401

import torch  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import run_grpo_reward_ab as train_lib  # noqa: E402
from twinkle_agentic.async_rl.rollout_resume import (  # noqa: E402
    JournalEvent,
    ResumeStore,
)
from twinkle_agentic.async_rl.streaming_driver import StreamingDriver  # noqa: E402
from twinkle_agentic.rollout.multi_turn import RolloutStatus  # noqa: E402

RUN_ID = "npu-checkpoint-kill"
CASES = (
    "optimizer_step_uncommitted",
    "adapter_temp_uncommitted",
    "checkpoint_promoted_uncommitted",
    "checkpoint_committed",
)


class _Future:
    def __init__(self, value):
        self._value = value

    def exception(self):
        return None

    def result(self):
        return self._value

    def add_done_callback(self, callback):
        callback(self)


class _DoneRollout:
    def step(self, trajectory, episode_id=None, session_wait=None):
        row = dict(trajectory, turns=1)
        sample = RolloutStatus(
            episode_id=episode_id,
            turns=1,
            done=True,
            token_lengths=(1, 1),
            stop_reason="stop",
        )
        return _Future(([row], sample))

    def stop(self, trajectory, episode_id=None):
        return None


def _prompts():
    return [{"messages": [{"role": "user", "content": "npu-checkpoint-kill"}]}]


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    data = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(pending, "wb") as stream:
        stream.write(data.encode("utf-8"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _sync_tree(path: Path) -> None:
    for item in sorted(path.rglob("*")):
        if item.is_file():
            with open(item, "rb") as stream:
                os.fsync(stream.fileno())
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _lora_digest(model) -> str:
    digest = hashlib.sha256()
    parameters = 0
    for name, parameter in sorted(model.named_parameters()):
        if "lora_" not in name or not parameter.requires_grad:
            continue
        tensor = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
        parameters += tensor.numel()
    if parameters == 0:
        raise RuntimeError("model has no trainable LoRA parameters")
    return digest.hexdigest()


def _optimizer_steps(optimizer) -> list[int]:
    steps = []
    for state in optimizer.state_dict()["state"].values():
        step = state.get("step")
        if torch.is_tensor(step):
            steps.append(int(step.detach().cpu()))
        elif step is not None:
            steps.append(int(step))
    return sorted(set(steps))


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _write_checkpoint(
    model,
    optimizer,
    path: Path,
    *,
    step: int,
    model_path: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"checkpoint already exists: {path}")
    path.mkdir(parents=True)
    model.save_pretrained(str(path), safe_serialization=True)
    torch.save(_to_cpu(optimizer.state_dict()), path / "optimizer.pt")
    metadata = {
        "format": "twinkle.npu-checkpoint-kill.v1",
        "step": step,
        "model": model_path,
        "lora_digest": _lora_digest(model),
        "optimizer_steps": _optimizer_steps(optimizer),
    }
    _atomic_write_json(path / "metadata.json", metadata)
    _sync_tree(path)
    return {
        **metadata,
        "path": str(path),
        "checkpoint_digest": _checkpoint_digest(path),
    }


def _promote_checkpoint(pending: Path, committed: Path) -> None:
    if committed.exists():
        raise FileExistsError(f"checkpoint target already exists: {committed}")
    os.replace(pending, committed)
    descriptor = os.open(committed.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _device():
    import torch_npu  # noqa: F401

    torch.npu.set_device(0)
    return torch.device("npu:0")


def _load_base(model_path: str, device):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    return tokenizer, model


def _trainable_model(model_path: str, device, seed: int):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    tokenizer, base = _load_base(model_path, device)
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=8,
            lora_alpha=16,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        ),
    )
    return tokenizer, model


def _load_checkpoint_model(checkpoint: Path, model_path: str, device, seed: int):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    tokenizer, base = _load_base(model_path, device)
    model = PeftModel.from_pretrained(base, str(checkpoint), is_trainable=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-5,
        weight_decay=0.0,
    )
    optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=False))
    return tokenizer, model, optimizer


def _ready_driver(root: Path):
    store = ResumeStore(root / "resume", RUN_ID)
    driver = StreamingDriver(
        _prompts(),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        resume_store=store,
    )
    driver._start()
    deadline = time.monotonic() + 10
    while driver._ready_rows < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    if driver._ready_rows != 2:
        raise AssertionError("group did not become ready")
    return store, driver


def _response_logps(model, prompt, generated) -> list[list[float]]:
    group_size = generated.shape[0]
    prompt_ids = prompt["input_ids"].repeat(group_size, 1)
    attention_mask = prompt["attention_mask"].repeat(group_size, 1)
    input_ids = torch.cat((prompt_ids, generated), dim=1)
    full_mask = torch.cat((attention_mask, torch.ones_like(generated)), dim=1)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=full_mask).logits.float()
        offset = prompt_ids.shape[1] - 1
        response_logits = logits[:, offset : offset + generated.shape[1]]
        logps = (
            torch.log_softmax(response_logits, dim=-1)
            .gather(
                -1,
                generated.unsqueeze(-1),
            )
            .squeeze(-1)
        )
    return logps.detach().cpu().tolist()


def _tokenize_group(tokenizer, device, question: str, target: str):
    prompt = train_lib._tokenized_prompt(tokenizer, device, question, max_length=512)
    correct = f"The final answer is \\boxed{{{target}}}."
    wrong = "The final answer is \\boxed{0}."
    encoded = [
        tokenizer(text, add_special_tokens=False).input_ids + [tokenizer.eos_token_id] for text in (correct, wrong)
    ]
    width = max(len(tokens) for tokens in encoded)
    encoded = [tokens + [tokenizer.eos_token_id] * (width - len(tokens)) for tokens in encoded]
    generated = torch.tensor(encoded, dtype=torch.long, device=device)
    rewards, metrics = train_lib._rewards(question, [correct, wrong], target, format_weight=0.2)
    if rewards[0] <= rewards[1]:
        raise AssertionError(f"fixed GSM8K pair did not produce a nontrivial reward: {rewards}")
    return prompt, generated, rewards, {"correct": correct, "wrong": wrong, **metrics}


def _container_info() -> dict[str, Any]:
    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    return {
        "hostname": platform.node(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_npu": version("torch-npu"),
        "transformers": version("transformers"),
        "peft": version("peft"),
    }


def _worker(case: str, root: Path, model_path: str, train_jsonl: str, seed: int) -> int:
    device = _device()
    tokenizer, model = _trainable_model(model_path, device, seed)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=5e-5,
        weight_decay=0.0,
    )
    records = train_lib._load_jsonl(train_jsonl, 1, 0)
    record = records[0]
    target = train_lib._ground_truth(record["answer"])

    checkpoints = root / "checkpoints"
    initial_path = checkpoints / "step-0"
    initial = _write_checkpoint(model, optimizer, initial_path, step=0, model_path=model_path)
    initial_digest = initial["lora_digest"]

    store, driver = _ready_driver(root)
    rows, facts = driver._next_batch()
    lease_id = str(driver._lease["lease_id"])
    prompt, generated, rewards, trajectory = _tokenize_group(
        tokenizer,
        device,
        record["question"],
        target,
    )
    old_logps = _response_logps(model, prompt, generated)
    trained = train_lib._train_group(model, optimizer, prompt, generated, old_logps, rewards)
    torch.npu.synchronize()
    trained_digest = _lora_digest(model)
    if trained_digest == initial_digest:
        raise AssertionError("real NPU optimizer step did not change LoRA weights")
    trained_steps = _optimizer_steps(optimizer)
    if trained_steps != [1]:
        raise AssertionError(f"optimizer state was not advanced exactly once: {trained_steps}")

    final_path = checkpoints / "step-1"
    pending_path = checkpoints / f".step-1.tmp.{os.getpid()}"
    candidate = None
    committed_lease = None
    if case != "optimizer_step_uncommitted":
        candidate = _write_checkpoint(model, optimizer, pending_path, step=1, model_path=model_path)
        if candidate["lora_digest"] != trained_digest:
            raise AssertionError("saved adapter does not match the in-memory trained LoRA")
        if case in {"checkpoint_promoted_uncommitted", "checkpoint_committed"}:
            _promote_checkpoint(pending_path, final_path)
            candidate["promoted_path"] = str(final_path)
            candidate["promoted_checkpoint_digest"] = _checkpoint_digest(final_path)
        if case == "checkpoint_committed":
            committed_lease = driver.commit_batch(
                str(final_path),
                {
                    "optimizer_step": 1,
                    "adapter_path": str(final_path),
                    "lora_digest": trained_digest,
                },
            )
            if committed_lease != lease_id:
                raise AssertionError(f"commit returned {committed_lease!r}, expected {lease_id!r}")

    # Do not take a manual snapshot here.  The probe is meant to prove that
    # StreamingDriver persists its own stable boundaries; an explicit
    # state_dict() would hide a missing automatic write.
    stats = driver.stats()
    marker = {
        "format": "twinkle.npu-checkpoint-kill-marker.v1",
        "case": case,
        "root": str(root),
        "device": str(device),
        "seed": seed,
        "run_id": RUN_ID,
        "lease_id": lease_id,
        "batch_rows": len(rows),
        "batch_facts": _to_cpu(facts),
        "initial_checkpoint": initial,
        "candidate_checkpoint": candidate,
        "final_checkpoint_path": str(final_path),
        "pending_checkpoint_path": str(pending_path),
        "initial_lora_digest": initial_digest,
        "trained_lora_digest": trained_digest,
        "trained_optimizer_steps": trained_steps,
        "optimizer_loss": trained["loss"],
        "driver_state_path": str(store.root / "objects" / "driver"),
        "driver_handed": stats["handed"],
        "driver_persistence": "automatic",
        "trajectory": trajectory,
        "container": _container_info(),
        "marker_created_unix": time.time(),
    }
    _atomic_write_json(root / "ready.json", marker)
    print(json.dumps({"case": case, "ready": True, "lease_id": lease_id}, ensure_ascii=False))
    sys.stdout.flush()
    sys.stderr.flush()
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("SIGKILL returned unexpectedly")


def _last_commit(store: ResumeStore, lease_id: str):
    return store.journal.last(
        lease_id,
        events=(JournalEvent.MODEL_CHECKPOINT_COMMITTED,),
    )


def _verify(case: str, root: Path, model_path: str, seed: int) -> int:
    marker = json.loads((root / "ready.json").read_text(encoding="utf-8"))
    if marker["case"] != case:
        raise AssertionError(f"marker case {marker['case']!r} != {case!r}")
    device = _device()
    store = ResumeStore(root / "resume", RUN_ID)
    driver = StreamingDriver(
        _prompts(),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        resume_store=store,
    )
    driver.load_state_dict()
    recovered_batch = driver._next_batch()
    lease_id = marker["lease_id"]
    commit = _last_commit(store, lease_id)
    committed = case == "checkpoint_committed"

    if committed:
        if commit is None:
            raise AssertionError("checkpoint_committed case has no commit journal record")
        expected_checkpoint = Path(commit.payload["checkpoint"])
        expected_digest = marker["trained_lora_digest"]
        expected_steps = [1]
        if recovered_batch is not None:
            raise AssertionError("committed batch was delivered again")
        if driver._lease is not None:
            raise AssertionError("committed lease was not cleared")
    else:
        if commit is not None:
            raise AssertionError("uncommitted case unexpectedly has a checkpoint commit")
        expected_checkpoint = Path(marker["initial_checkpoint"]["path"])
        expected_digest = marker["initial_lora_digest"]
        expected_steps = []
        if recovered_batch is None:
            raise AssertionError("uncommitted batch was not redelivered")
        if driver._lease is None or str(driver._lease["lease_id"]) != lease_id:
            raise AssertionError("uncommitted lease identity was not restored")

    _, model, optimizer = _load_checkpoint_model(expected_checkpoint, model_path, device, seed)
    actual_digest = _lora_digest(model)
    actual_steps = _optimizer_steps(optimizer)
    if actual_digest != expected_digest:
        raise AssertionError("loaded LoRA checkpoint does not match the transaction decision")
    if actual_steps != expected_steps:
        raise AssertionError(f"loaded optimizer steps {actual_steps} != expected {expected_steps}")

    orphan_check = None
    if case == "adapter_temp_uncommitted":
        pending = Path(marker["pending_checkpoint_path"])
        final = Path(marker["final_checkpoint_path"])
        if not pending.is_dir():
            raise AssertionError("temporary adapter checkpoint was not preserved for the probe")
        if final.exists():
            raise AssertionError("temporary adapter unexpectedly reached its final path")
        orphan_check = {
            "pending_exists": True,
            "pending_matches_trained": _checkpoint_digest(pending)
            == marker["candidate_checkpoint"]["checkpoint_digest"],
            "final_absent": True,
        }
    elif case == "checkpoint_promoted_uncommitted":
        final = Path(marker["final_checkpoint_path"])
        if not final.is_dir():
            raise AssertionError("promoted checkpoint is missing")
        orphan_check = {
            "promoted_exists": True,
            "promoted_matches_trained": _checkpoint_digest(final)
            == marker["candidate_checkpoint"]["promoted_checkpoint_digest"],
            "journal_left_it_uncommitted": commit is None,
        }

    report = {
        "format": "twinkle.npu-checkpoint-kill-verify.v1",
        "ok": True,
        "case": case,
        "device": str(device),
        "expected_checkpoint": str(expected_checkpoint),
        "expected_lora_digest": expected_digest,
        "actual_lora_digest": actual_digest,
        "expected_optimizer_steps": expected_steps,
        "actual_optimizer_steps": actual_steps,
        "committed": committed,
        "commit_checkpoint": None if commit is None else commit.payload["checkpoint"],
        "redelivered_batch": recovered_batch is not None,
        "redelivered_rows": 0 if recovered_batch is None else len(recovered_batch[0]),
        "restored_lease": None if driver._lease is None else driver._lease["lease_id"],
        "orphan_check": orphan_check,
        "container": _container_info(),
    }
    driver.close()
    _atomic_write_json(root / "verify.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _safe_case_root(root: Path, case: str) -> Path:
    root = root.resolve()
    case_root = (root / case).resolve()
    if case_root.parent != root:
        raise ValueError(f"case root escaped output root: {case_root}")
    if case_root.exists():
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True)
    return case_root


def _wait_for_marker(process: subprocess.Popen, marker: Path, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if marker.is_file():
            return json.loads(marker.read_text(encoding="utf-8"))
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"worker exited {return_code} before writing {marker}")
        time.sleep(0.05)
    raise TimeoutError(f"worker {process.pid} did not reach {marker} within {timeout}s")


def _run_child(arguments: list[str]) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def run_case(
    case: str,
    root: Path,
    model_path: str,
    train_jsonl: str,
    seed: int,
    timeout: float,
) -> dict[str, Any]:
    case_root = _safe_case_root(root, case)
    marker_path = case_root / "ready.json"
    worker = _run_child(
        [
            "--mode",
            "worker",
            "--case",
            case,
            "--root",
            str(case_root),
            "--model",
            model_path,
            "--train-jsonl",
            train_jsonl,
            "--seed",
            str(seed),
        ]
    )
    started = time.time()
    try:
        marker = _wait_for_marker(worker, marker_path, timeout)
        output, _ = worker.communicate(timeout=20)
    except Exception:
        worker.kill()
        output, _ = worker.communicate(timeout=20)
        raise
    return_code = worker.returncode
    (case_root / "worker.log").write_text(output or "", encoding="utf-8")
    if return_code not in (-9, 137):
        raise AssertionError(f"worker was not killed by SIGKILL: returncode={return_code}\n{output}")

    verify = _run_child(
        [
            "--mode",
            "verify",
            "--case",
            case,
            "--root",
            str(case_root),
            "--model",
            model_path,
            "--seed",
            str(seed),
        ]
    )
    verify_output, _ = verify.communicate(timeout=timeout)
    (case_root / "verify.log").write_text(verify_output, encoding="utf-8")
    if verify.returncode != 0:
        raise AssertionError(f"verification failed with {verify.returncode}\n{verify_output}")
    verification = json.loads((case_root / "verify.json").read_text(encoding="utf-8"))
    return {
        "case": case,
        "ok": bool(verification["ok"]),
        "worker_returncode": return_code,
        "worker_killed_by_sigkill": return_code in (-9, 137),
        "marker": marker,
        "verification": verification,
        "elapsed_s": time.time() - started,
    }


def _parent(args) -> int:
    if args.cases:
        selected = [item.strip() for item in args.cases.split(",") if item.strip()]
    else:
        selected = list(CASES)
    invalid = sorted(set(selected) - set(CASES))
    if invalid:
        raise SystemExit(f"unknown cases: {invalid}")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "twinkle.npu-checkpoint-kill-report.v1",
        "ok": False,
        "started_unix": time.time(),
        "model": args.model,
        "train_jsonl": args.train_jsonl,
        "seed": args.seed,
        "container": _container_info(),
        "cases": [],
    }
    try:
        for case in selected:
            result = run_case(
                case,
                root,
                args.model,
                args.train_jsonl,
                args.seed,
                args.timeout,
            )
            report["cases"].append(result)
            print(
                f"[{case}] worker={result['worker_returncode']} "
                f"redelivered={result['verification']['redelivered_batch']} "
                f"steps={result['verification']['actual_optimizer_steps']}",
                file=sys.stderr,
                flush=True,
            )
        report["ok"] = all(item["ok"] for item in report["cases"])
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        report["finished_unix"] = time.time()
        _atomic_write_json(root / "report.json", report)
        if args.output:
            _atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("parent", "worker", "verify"), default="parent")
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--cases", default="")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model", default="/workspace/shared_assets/models/Qwen/Qwen3-0.6B")
    parser.add_argument("--train-jsonl", default="data/gsm8k/train.jsonl")
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.mode == "worker":
        if not args.case:
            parser.error("--case is required for worker mode")
        return _worker(args.case, args.root.resolve(), args.model, args.train_jsonl, args.seed)
    if args.mode == "verify":
        if not args.case:
            parser.error("--case is required for verify mode")
        return _verify(args.case, args.root.resolve(), args.model, args.seed)
    return _parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
