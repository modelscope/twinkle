#!/usr/bin/env python3
"""Measure durable journal/object overhead and cross-process journal serialization."""

# ruff: noqa: E402, I001, T201

from __future__ import annotations

import argparse
import json
import multiprocessing
import pickle
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "review" / "gate0"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import py310_compat  # noqa: F401
import run_gate0  # noqa: F401

from twinkle_agentic.async_rl.rollout_resume import DurableJournal, JournalEvent, ResumeStore


def _payload():
    return {"tokens": list(range(64)), "state_key": "run/group/attempt/member"}


def _writer_worker(root: str, writer_index: int, count: int):
    journal = DurableJournal(Path(root) / "concurrent.jsonl")
    for item in range(count):
        journal.append(JournalEvent.PROMPT_ASSIGNED, f"run/{writer_index}/{item}", _payload())


def _journal_benchmark(root: Path, iterations: int) -> dict:
    payload = _payload()
    memory_records = []
    started = time.perf_counter()
    for index in range(iterations):
        memory_records.append(
            {
                "seq": index,
                "event": JournalEvent.PROMPT_ASSIGNED.value,
                "state_key": f"run/{index}",
                "payload": payload,
            }
        )
    memory_s = time.perf_counter() - started

    path = root / "journal" / "rollout.jsonl"
    journal = DurableJournal(path)
    started = time.perf_counter()
    for index in range(iterations):
        journal.append(JournalEvent.PROMPT_ASSIGNED, f"run/{index}", payload)
    durable_s = time.perf_counter() - started

    started = time.perf_counter()
    reopened = DurableJournal(path)
    reopen_s = time.perf_counter() - started
    if len(tuple(reopened.records())) != iterations:
        raise AssertionError("journal lost records")
    return {
        "iterations": iterations,
        "memory_s": memory_s,
        "durable_s": durable_s,
        "reopen_s": reopen_s,
        "durable_us_per_append": durable_s / iterations * 1_000_000,
        "durable_appends_per_s": iterations / durable_s,
        "slowdown_vs_memory": durable_s / max(memory_s, 1e-9),
        "journal_bytes": path.stat().st_size,
        "bytes_per_append": path.stat().st_size / iterations,
        "memory_records": len(memory_records),
    }


def _object_benchmark(root: Path, iterations: int) -> dict:
    value = {"tokens": list(range(256)), "messages": [{"role": "assistant", "content": "x" * 512}]}
    started = time.perf_counter()
    serialized = 0
    for _ in range(iterations):
        serialized += len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    pickle_s = time.perf_counter() - started

    store = ResumeStore(root / "objects", "objects")
    started = time.perf_counter()
    for index in range(iterations):
        store.save_object("episodes", f"episode-{index}", value)
    durable_s = time.perf_counter() - started
    object_bytes = sum(path.stat().st_size for path in (root / "objects" / "objects").rglob("*.resume"))
    return {
        "iterations": iterations,
        "pickle_s": pickle_s,
        "durable_s": durable_s,
        "durable_us_per_object": durable_s / iterations * 1_000_000,
        "objects_per_s": iterations / durable_s,
        "slowdown_vs_pickle": durable_s / max(pickle_s, 1e-9),
        "pickle_bytes": serialized,
        "object_bytes": object_bytes,
        "storage_amplification": object_bytes / max(serialized, 1),
    }


def _concurrent_process_benchmark(root: Path, writers: int, per_writer: int) -> dict:
    path = root / "concurrent.jsonl"
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_writer_worker, args=(str(root), index, per_writer)) for index in range(writers)
    ]
    started = time.perf_counter()
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        if process.exitcode != 0:
            raise RuntimeError(f"journal writer exited with {process.exitcode}")
    elapsed = time.perf_counter() - started
    records = tuple(DurableJournal(path).records())
    return {
        "writers": writers,
        "per_writer": per_writer,
        "elapsed_s": elapsed,
        "records": len(records),
        "contiguous_sequence": [record.seq for record in records] == list(range(writers * per_writer)),
        "journal_bytes": path.stat().st_size,
    }


def run(root: Path, iterations: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    journal = _journal_benchmark(root, iterations)
    objects = _object_benchmark(root, max(20, iterations // 10))
    concurrent = _concurrent_process_benchmark(root, writers=4, per_writer=max(10, iterations // 10))
    ok = (
        journal["durable_appends_per_s"] > 0
        and concurrent["records"] == concurrent["writers"] * concurrent["per_writer"]
        and concurrent["contiguous_sequence"]
    )
    return {
        "ok": ok,
        "python": sys.version,
        "platform": sys.platform,
        "journal": journal,
        "objects": objects,
        "concurrent_processes": concurrent,
        "summary": {
            "journal_overhead_factor": journal["slowdown_vs_memory"],
            "object_overhead_factor": objects["slowdown_vs_pickle"],
            "median_durable_events_per_s": statistics.median([journal["durable_appends_per_s"]]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    args = parser.parse_args()
    report = run(args.root, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
