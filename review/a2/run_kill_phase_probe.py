#!/usr/bin/env python3
"""SIGKILL a rollout process at each durable transaction boundary."""

# ruff: noqa: E402, I001, T201

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import signal
import subprocess
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
from twinkle_agentic.async_rl.rollout_resume import (  # noqa: E402
    JournalEvent,
    ResumeStore,
    atomic_write_bytes,
)
from twinkle_agentic.async_rl.streaming_driver import StreamingDriver  # noqa: E402
from twinkle_agentic.rollout.multi_turn import RolloutStatus  # noqa: E402


class _Future:
    def __init__(self, value):
        self._value = value

    def exception(self):
        return None

    def result(self):
        return self._value

    def add_done_callback(self, callback):
        callback(self)


class _PendingFuture:
    def add_done_callback(self, callback):
        self.callback = callback


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


class _BlockingRollout(_DoneRollout):
    def step(self, trajectory, episode_id=None, session_wait=None):
        return _PendingFuture()


def _prompts():
    return [{"messages": [{"role": "user", "content": "kill-phase-probe"}]}]


def _ready_driver(root: Path):
    store = ResumeStore(root, "kill-phase")
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
    deadline = time.monotonic() + 5
    while driver._ready_rows < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    if driver._ready_rows != 2:
        raise AssertionError("group did not become ready")
    return store, driver


def _save_trainer_state(model, optimizer, path: Path):
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    atomic_write_bytes(path, buffer.getvalue())


def _load_trainer_state(model, optimizer, path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])


def _optimizer_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    (model.weight - 2.0).square().sum().backward()
    optimizer.step()


def _signal_ready(root: Path):
    (root / "ready").write_text("ready", encoding="utf-8")
    while True:
        time.sleep(60)


def _worker(phase: str, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if phase == "prompt_assigned":
        store = ResumeStore(root, "kill-phase")
        driver = StreamingDriver(
            _prompts(),
            _BlockingRollout(),
            batch_size=2,
            num_generations=2,
            max_staleness=0,
            max_steps=1,
            resume_store=store,
        )
        driver._admit()
        driver.state_dict()
        _signal_ready(root)

    if phase == "group_ready":
        _, driver = _ready_driver(root)
        driver.state_dict()
        _signal_ready(root)

    if phase == "batch_leased":
        _, driver = _ready_driver(root)
        driver._next_batch()
        driver.state_dict()
        _signal_ready(root)

    if phase in {"optimizer_step_uncommitted", "checkpoint_committed"}:
        _, driver = _ready_driver(root)
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        initial = root / "initial.pt"
        _save_trainer_state(model, optimizer, initial)
        driver._next_batch()
        _optimizer_step(model, optimizer)
        if phase == "checkpoint_committed":
            committed = root / "committed.pt"
            _save_trainer_state(model, optimizer, committed)
            driver.commit_batch(str(committed), {"optimizer_step": 1})
        driver.state_dict()
        _signal_ready(root)


def _wait_for_marker(root: Path, process: subprocess.Popen, timeout: float = 30.0):
    deadline = time.monotonic() + timeout
    marker = root / "ready"
    while time.monotonic() < deadline:
        if marker.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise RuntimeError(f"worker exited before marker: {stdout}\n{stderr}")
        time.sleep(0.01)
    raise TimeoutError(f"worker {process.pid} did not reach {marker}")


def _kill(process: subprocess.Popen):
    if os.name == "nt":
        process.kill()
    else:
        os.kill(process.pid, signal.SIGKILL)
    process.wait(timeout=10)


def _new_driver(root: Path, rollout=None):
    store = ResumeStore(root, "kill-phase")
    driver = StreamingDriver(
        _prompts(),
        rollout or _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        resume_store=store,
    )
    state = store.load_object("driver", "kill-phase")
    driver.load_state_dict(state)
    return store, driver, state


def _verify(phase: str, root: Path) -> dict:
    if phase == "prompt_assigned":
        _, driver, state = _new_driver(root)
        try:
            result = {
                "prompt_drawn": state["prompts_drawn"] == 1,
                "group_restored": list(driver._groups) == ["0"],
                "stable_state_key": driver._groups["0"]
                .members[0]
                .submit_traj["state_key"]
                .startswith("kill-phase/group/0/"),
            }
        finally:
            driver.close()
        return result

    if phase == "group_ready":
        _, driver, _ = _new_driver(root)
        try:
            rows, facts = driver._next_batch()
            return {"rows": len(rows), "policy_versions": facts["policy_versions"]}
        finally:
            driver.close()

    if phase == "batch_leased":
        _, driver, state = _new_driver(root)
        try:
            rows, facts = driver._next_batch()
            return {
                "rows": len(rows),
                "same_lease": facts == state["lease"]["facts"],
                "lease_id": driver._lease["lease_id"],
            }
        finally:
            driver.close()

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    checkpoint = root / ("committed.pt" if phase == "checkpoint_committed" else "initial.pt")
    _load_trainer_state(model, optimizer, checkpoint)
    if phase == "optimizer_step_uncommitted":
        _, driver, state = _new_driver(root)
        try:
            rows, _ = driver._next_batch()
            committed = driver.resume_store.journal.last(
                state["lease"]["lease_id"], events=(JournalEvent.MODEL_CHECKPOINT_COMMITTED,)
            )
            return {
                "recovered_weight": float(model.weight.detach()),
                "same_batch_redelivered": len(rows) == 2,
                "checkpoint_not_committed": committed is None,
            }
        finally:
            driver.close()

    store, driver, _ = _new_driver(root)
    try:
        commit_record = next(
            record
            for record in reversed(tuple(store.journal.records()))
            if record.event == JournalEvent.MODEL_CHECKPOINT_COMMITTED
        )
        return {
            "recovered_weight": float(model.weight.detach()),
            "lease_cleared": driver._lease is None,
            "no_redelivery": driver._next_batch() is None,
            "checkpoint_committed": commit_record.state_key.startswith("kill-phase/batch/"),
        }
    finally:
        driver.close()


def run(output_root: Path) -> dict:
    phases = [
        "prompt_assigned",
        "group_ready",
        "batch_leased",
        "optimizer_step_uncommitted",
        "checkpoint_committed",
    ]
    report = {"ok": True, "phases": {}}
    for phase in phases:
        root = output_root / phase
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker", phase, "--root", str(root)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        started = time.perf_counter()
        _wait_for_marker(root, process)
        _kill(process)
        killed_after = time.perf_counter() - started
        evidence = _verify(phase, root)
        checks = {
            "prompt_assigned": lambda item: all(
                item[key] for key in ("prompt_drawn", "group_restored", "stable_state_key")
            ),
            "group_ready": lambda item: item["rows"] == 2,
            "batch_leased": lambda item: item["rows"] == 2 and item["same_lease"],
            "optimizer_step_uncommitted": lambda item: (
                item["same_batch_redelivered"] and item["checkpoint_not_committed"]
            ),
            "checkpoint_committed": lambda item: (
                item["lease_cleared"] and item["no_redelivery"] and item["checkpoint_committed"]
            ),
        }
        ok = checks[phase](evidence)
        report["phases"][phase] = {
            "ok": ok,
            "kill_after_s": killed_after,
            "process_returncode": process.returncode,
            "evidence": evidence,
        }
        report["ok"] = report["ok"] and ok
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        choices=[
            "prompt_assigned",
            "group_ready",
            "batch_leased",
            "optimizer_step_uncommitted",
            "checkpoint_committed",
        ],
    )
    parser.add_argument("--root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.root is None:
            parser.error("--root is required in worker mode")
        _worker(args.worker, args.root)
        return 0
    if args.output_root is None or args.output is None:
        parser.error("--output-root and --output are required outside worker mode")
    report = run(args.output_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
