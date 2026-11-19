"""在 A2 的 Python 3.10 NPU 环境中运行 RolloutResume 聚焦测试。"""
# ruff: noqa: E402, I001

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src", ROOT / "review" / "gate0"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import py310_compat  # noqa: E402,F401
import run_gate0  # noqa: E402,F401

# Pytest 的 prepend import mode 会在收集阶段重排 sys.path。先把被测包固定到
# 当前源码树，避免 A2 上预装的其他 twinkle_agentic 抢占模块缓存。
import twinkle_agentic.async_rl  # noqa: E402,F401
import twinkle_agentic.envs  # noqa: E402,F401
import twinkle_agentic.rollout  # noqa: E402,F401
import pytest  # noqa: E402


TESTS = [
    "tests/twinkle_agentic/test_rollout_resume_store.py",
    "tests/twinkle_agentic/test_localenv_resume.py",
    "tests/twinkle_agentic/test_turn_ledger_resume.py",
    "tests/twinkle_agentic/test_multi_turn_resume.py",
    "tests/twinkle_agentic/test_group_policy_binding.py",
    "tests/twinkle_agentic/test_streaming_driver_resume.py",
    "tests/twinkle_agentic/test_resume_boundaries.py",
    "tests/twinkle_agentic/test_tail_prefetch.py",
    "tests/twinkle_client/test_checkpoint_policy_history.py",
]


if __name__ == "__main__":
    # tests/twinkle_agentic 与 src/twinkle_agentic 同名；prepend 模式会把测试目录
    # 当成生产包，导致子模块解析错位。
    raise SystemExit(pytest.main(["-q", "--import-mode=importlib", *TESTS]))
