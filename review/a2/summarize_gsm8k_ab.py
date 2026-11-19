#!/usr/bin/env python3
"""Summarize paired GSM8K A/B reports produced by run_grpo_reward_ab.py."""
# ruff: noqa: T201

from __future__ import annotations
import argparse
import math
import statistics
from pathlib import Path

import json


def _mcnemar_exact(baseline_only: int, durable_only: int) -> float:
    discordant = baseline_only + durable_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, count) for count in range(min(baseline_only, durable_only) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _summarize(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    baseline = [sample["accuracy"] for sample in report["baseline_eval"]["samples"]]
    durable = [sample["accuracy"] for sample in report["durable_eval"]["samples"]]
    if len(baseline) != len(durable):
        raise ValueError(f"{path}: baseline and durable evaluation sizes differ")

    baseline_only = sum(left == 1 and right == 0 for left, right in zip(baseline, durable, strict=True))
    durable_only = sum(left == 0 and right == 1 for left, right in zip(baseline, durable, strict=True))
    return {
        "file": path.name,
        "seed": report["seed"],
        "lr": report["config"]["lr"],
        "eval_records": len(baseline),
        "baseline_correct": int(sum(baseline)),
        "durable_correct": int(sum(durable)),
        "baseline_only": baseline_only,
        "durable_only": durable_only,
        "accuracy_delta": report["accuracy_delta"],
        "reward_delta": report["reward_delta"],
        "mcnemar_exact_p": _mcnemar_exact(baseline_only, durable_only),
        "baseline_steps": report["baseline"]["train_steps"],
        "durable_steps": report["durable"]["train_steps"],
        "baseline_lost_groups": report["baseline"]["lost_groups"],
        "durable_lost_groups": report["durable"]["lost_groups"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runs = [_summarize(path) for path in args.reports]
    result = {
        "runs": runs,
        "mean_accuracy_delta": statistics.mean(run["accuracy_delta"] for run in runs),
        "median_accuracy_delta": statistics.median(run["accuracy_delta"] for run in runs),
        "mean_reward_delta": statistics.mean(run["reward_delta"] for run in runs),
        "positive_accuracy_runs": sum(run["accuracy_delta"] > 0 for run in runs),
        "positive_reward_runs": sum(run["reward_delta"] > 0 for run in runs),
        "run_count": len(runs),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
