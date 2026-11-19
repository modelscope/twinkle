#!/usr/bin/env python3
"""Exercise a real torch optimizer step around the durable batch commit boundary."""

# ruff: noqa: E402, I001, T201

from __future__ import annotations

import argparse
import io
import json
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
from twinkle_agentic.async_rl.rollout_resume import ResumeStore, atomic_write_bytes  # noqa: E402
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
    return [{"messages": [{"role": "user", "content": "trainer-checkpoint-probe"}]}]


def _save_trainer_state(model, optimizer, path: Path):
    buffer = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        buffer,
    )
    atomic_write_bytes(path, buffer.getvalue())


def _load_trainer_state(model, optimizer, path: Path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])


def _ready_driver(root: Path):
    store = ResumeStore(root, "trainer-transaction")
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


def _optimizer_step(model, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss = (model.weight - 3.0).square().sum()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def run(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    store, driver = _ready_driver(root / "uncommitted")
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    initial_path = root / "initial.pt"
    _save_trainer_state(model, optimizer, initial_path)
    initial_weight = float(model.weight.detach())
    try:
        rows, facts = driver._next_batch()
        loss = _optimizer_step(model, optimizer)
        stepped_weight = float(model.weight.detach())
        uncommitted_state = driver.state_dict()
    finally:
        driver.close()

    recovered_store = ResumeStore(root / "uncommitted", "trainer-transaction")
    recovered = StreamingDriver(
        _prompts(),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        resume_store=recovered_store,
    )
    recovered_model = torch.nn.Linear(1, 1, bias=False)
    recovered_optimizer = torch.optim.SGD(recovered_model.parameters(), lr=0.1)
    _load_trainer_state(recovered_model, recovered_optimizer, initial_path)
    recovered.load_state_dict(uncommitted_state)
    redelivered_rows, redelivered_facts = recovered._next_batch()
    recovered.close()

    committed_store, committed_driver = _ready_driver(root / "committed")
    committed_model = torch.nn.Linear(1, 1, bias=False)
    committed_optimizer = torch.optim.SGD(committed_model.parameters(), lr=0.1)
    committed_path = root / "committed.pt"
    try:
        committed_rows, _ = committed_driver._next_batch()
        _optimizer_step(committed_model, committed_optimizer)
        _save_trainer_state(committed_model, committed_optimizer, committed_path)
        committed_weight = float(committed_model.weight.detach())
        lease_id = committed_driver.commit_batch(
            str(committed_path),
            {"optimizer_step": 1, "loss": loss},
        )
        committed_state = committed_driver.state_dict()
    finally:
        committed_driver.close()

    after_commit_store = ResumeStore(root / "committed", "trainer-transaction")
    after_commit = StreamingDriver(
        _prompts(),
        _DoneRollout(),
        batch_size=2,
        num_generations=2,
        max_staleness=0,
        max_steps=1,
        resume_store=after_commit_store,
    )
    restored_committed_model = torch.nn.Linear(1, 1, bias=False)
    restored_committed_optimizer = torch.optim.SGD(restored_committed_model.parameters(), lr=0.1)
    _load_trainer_state(restored_committed_model, restored_committed_optimizer, committed_path)
    after_commit.load_state_dict(committed_state)
    no_redelivery = after_commit._next_batch() is None
    after_commit.close()

    report = {
        "ok": True,
        "uncommitted": {
            "initial_weight": initial_weight,
            "stepped_weight": stepped_weight,
            "optimizer_changed_weights": stepped_weight != initial_weight,
            "recovered_weight": float(recovered_model.weight.detach()),
            "rollback_to_previous_checkpoint": float(recovered_model.weight.detach()) == initial_weight,
            "same_batch_redelivered": redelivered_rows == rows,
            "same_facts_redelivered": redelivered_facts == facts,
        },
        "committed": {
            "committed_weight": committed_weight,
            "restored_weight": float(restored_committed_model.weight.detach()),
            "checkpoint_restored": float(restored_committed_model.weight.detach()) == committed_weight,
            "lease_id": lease_id,
            "lease_cleared": after_commit._lease is None,
            "no_redelivery_after_commit": no_redelivery,
            "journal_tail": [record.event.value for record in after_commit_store.journal.records()][-2:],
            "committed_rows": len(committed_rows),
        },
    }
    report["ok"] = all(
        (
            report["uncommitted"]["optimizer_changed_weights"],
            report["uncommitted"]["rollback_to_previous_checkpoint"],
            report["uncommitted"]["same_batch_redelivered"],
            report["uncommitted"]["same_facts_redelivered"],
            report["committed"]["checkpoint_restored"],
            report["committed"]["lease_cleared"],
            report["committed"]["no_redelivery_after_commit"],
            report["committed"]["journal_tail"][-1] == "MODEL_CHECKPOINT_COMMITTED",
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
