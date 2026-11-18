"""Offline Gate0 probes for RolloutResume.

The probes use the real StreamingDriver, MultiTurnRollout and LocalEnv classes.  They
install import-only stubs for optional runtime dependencies that are irrelevant to the
state-resume question, so the same script can run on a bare Windows host and on A2.

Expected success means every current state-loss counterexample was reproduced, while
the LocalEnv workspace-copy feasibility probe succeeded.  This is evidence for the RFC,
not a production resume implementation.
"""
# ruff: noqa: E402, I001, T201

from __future__ import annotations

import dataclasses
import importlib.util
import inspect
import json
import shutil
import sys
import tempfile
import traceback
import types
from concurrent.futures import Future
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _install_import_stubs() -> None:
    try:
        available = importlib.util.find_spec("swift.dev.template_contract") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        available = False
    if not available:
        swift = types.ModuleType("swift")
        dev = types.ModuleType("swift.dev")
        contract = types.ModuleType("swift.dev.template_contract")

        @dataclasses.dataclass
        class RolloutState:
            feature: dict[str, Any] = dataclasses.field(default_factory=dict)
            messages: tuple[dict[str, Any], ...] = ()
            tools: tuple[dict[str, Any], ...] = ()
            engine_input_ids: tuple[int, ...] = ()

        contract.RolloutState = RolloutState
        contract.__path__ = []
        contract.TOKEN_FILL = ()
        contract.SHIFTED_FIELDS = ()
        for name in (
            "append_and_finalize append_input_order coerce_tool_call_arguments feature_has_media refresh render_delta "
            "shift shift_values splice splice_state split_reasoning split_reasoning_state strip_special_tokens "
            "unshift unshift_values"
        ).split():
            setattr(contract, name, lambda *args, **kwargs: None)
        for name in (
            "BatchHooks BatchTemplate CompletionEncoding Continuation ConversationSample DatasetTemplate DecodeContext "
            "DecodedResponse EncodedRow InputOrderCompletionMask InputOrderLabels InputOrderLossScale InputOrderRow "
            "KTOSample MediaBundle Message ModelHook ModelInfo ModelRef OutputOrderCompletionMask OutputOrderLabels "
            "OutputOrderLossScale PreferencePair PreferenceSample PreparedRow PromptEncoder PromptEncoding "
            "ResponseDecoder SamplerTemplate TemplateProtocol TemplateSpec Tool"
        ).split():
            setattr(contract, name, type(name, (), {}))
        swift.dev = dev
        dev.template_contract = contract
        encoding = types.ModuleType("swift.dev.template_contract.encoding")
        encoding.render_delta = lambda *args, **kwargs: None
        output = types.ModuleType("swift.dev.template_contract.output")
        output._reasoning_is_open = lambda *args, **kwargs: False
        output.decode_output = lambda *args, **kwargs: None
        contract.encoding = encoding
        contract.output = output
        sys.modules["swift"] = swift
        sys.modules["swift.dev"] = dev
        sys.modules["swift.dev.template_contract"] = contract
        sys.modules["swift.dev.template_contract.encoding"] = encoding
        sys.modules["swift.dev.template_contract.output"] = output

    if importlib.util.find_spec("resource") is None:
        resource = types.ModuleType("resource")
        resource.RLIMIT_AS = 0
        resource.setrlimit = lambda *args, **kwargs: None
        sys.modules["resource"] = resource

    if importlib.util.find_spec("ray") is None:
        ray = types.ModuleType("ray")
        ray.ObjectRef = type("ObjectRef", (), {})
        ray.is_initialized = lambda: False
        sys.modules["ray"] = ray


_install_import_stubs()

from twinkle_agentic.async_rl.streaming_driver import StreamingDriver  # noqa: E402
from twinkle_agentic.envs.base import Env, SessionNotFound, StepResult  # noqa: E402
from twinkle_agentic.envs.localenv import LocalEnv  # noqa: E402
from twinkle_agentic.rollout.ledger import TurnLedger  # noqa: E402
from twinkle_agentic.rollout.multi_turn import EpisodeError, MultiTurnRollout, RolloutStatus  # noqa: E402


class PendingRollout:
    def __init__(self) -> None:
        self.pending: dict[str, tuple[Any, Future]] = {}

    def step(self, traj, *, episode_id, session_wait):
        future = Future()
        self.pending[episode_id] = (traj, future)
        return future

    def finish(self, driver: StreamingDriver, episode) -> None:
        traj, future = self.pending.pop(episode.episode_id)
        future.set_result(([traj], RolloutStatus(episode_id=episode.episode_id, done=True)))
        driver._handle(*driver._events.get_nowait())

    def stop(self, traj, *, episode_id):
        self.pending.pop(episode_id, None)


class CountingEnv(Env):
    effects = 0

    def _open(self, trajectory):
        return None, StepResult()

    def _step(self, state, tool_name, arguments):
        type(self).effects += 1
        return StepResult(observation=f"effect-{type(self).effects}")

    def _run_script(self, state, source, interpreter="python", timeout=None):
        return 0, source

    def _close(self, state):
        return None

    def _workspace(self, state):
        return None


def probe_inflight_prompt_is_lost() -> dict[str, Any]:
    rows = [{"prompt": index} for index in range(4)]
    first = StreamingDriver(rows, PendingRollout(), batch_size=2, num_generations=2)
    with first._lock:
        first._admit()

    saved = first.state_dict()
    assert saved == {"handed": 0, "prompts_drawn": 1}, saved
    assert not hasattr(first, "load_state_dict")
    assert [group.row["prompt"] for group in first._groups.values()] == [0]

    # This is what a caller can do today with prompts_drawn: skip the in-flight row.
    restarted = StreamingDriver(rows[1:], PendingRollout(), batch_size=2, num_generations=2)
    with restarted._lock:
        restarted._admit()
    restarted_rows = [group.row["prompt"] for group in restarted._groups.values()]
    assert restarted_rows == [1], restarted_rows
    first.close()
    restarted.close()
    return {"saved": saved, "skipped_prompt": 0, "restart_rows": restarted_rows}


def probe_ready_batch_is_unrecoverable() -> dict[str, Any]:
    rollout = PendingRollout()
    driver = StreamingDriver([{"prompt": 0}], rollout, batch_size=2, num_generations=2)
    with driver._lock:
        driver._admit()
    group = next(iter(driver._groups.values()))
    for episode in group.members:
        rollout.finish(driver, episode)
    assert driver._ready_rows == 2

    rows, _ = driver._next_batch()
    assert len(rows) == 2
    saved = driver.state_dict()
    assert saved == {"handed": 1, "prompts_drawn": 1}, saved
    assert set(saved) == {"handed", "prompts_drawn"}
    assert not driver._ready and not driver._groups
    assert not hasattr(driver, "load_state_dict")
    driver.close()
    return {"batch_rows": len(rows), "saved": saved, "ready_after_lease": 0}


def probe_live_rollout_cannot_reattach() -> dict[str, Any]:
    rollout = MultiTurnRollout(response_callback=lambda *args: None)
    view = {"turns": 1, "user_data": {"episode_id": "ep-gate0"}}
    try:
        rollout._step(view, "ep-gate0")
    except EpisodeError as exc:
        message = str(exc)
        assert "not held by this rollout" in message
    else:
        raise AssertionError("a restarted rollout unexpectedly reattached a live view")

    assert not hasattr(TurnLedger, "state_dict")
    assert not hasattr(TurnLedger, "load_state_dict")
    return {"error": message, "ledger_checkpoint_api": False}


def probe_localenv_session_cannot_reattach() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gate0-env-") as root:
        first = LocalEnv(root=root)
        first.reset("ep-gate0")
        first.step("ep-gate0", "write_file", {"path": "state.txt", "content": "turn-boundary"})
        second = LocalEnv(root=root)
        try:
            second.step("ep-gate0", "read_file", {"path": "state.txt"})
        except SessionNotFound as exc:
            message = str(exc)
        else:
            raise AssertionError("a new LocalEnv instance unexpectedly reattached the old session")
        assert not hasattr(first, "checkpoint_episode")
        assert not hasattr(first, "restore_episode")
        first.close("ep-gate0")
        second.shutdown()
    return {"error": message, "capability_api": False}


def probe_tool_effect_has_no_journal() -> dict[str, Any]:
    signature = inspect.signature(Env.step)
    assert "effect_key" not in signature.parameters
    CountingEnv.effects = 0
    first = CountingEnv()
    first.reset("effect-ep")
    first.step("effect-ep", "charge", {"amount": 1})
    first.close("effect-ep")

    # A replacement process repeats the same state key and the same call.
    second = CountingEnv()
    second.reset("effect-ep")
    second.step("effect-ep", "charge", {"amount": 1})
    second.close("effect-ep")
    assert CountingEnv.effects == 2
    return {"effects": CountingEnv.effects, "effect_key_in_api": False}


def probe_workspace_snapshot_is_feasible() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gate0-workspace-") as root:
        source = LocalEnv(root=str(Path(root) / "source"))
        source.reset("source-ep")
        source.step("source-ep", "write_file", {"path": "state.txt", "content": "turn-boundary"})
        workspace = Path(source.workspace("source-ep"))

        staging = Path(root) / "snapshot.staging"
        snapshot = Path(root) / "snapshot"
        shutil.copytree(workspace, staging)
        staging.replace(snapshot)
        source.close("source-ep")
        assert snapshot.joinpath("state.txt").read_text(encoding="utf-8") == "turn-boundary"

        restored = LocalEnv(root=str(Path(root) / "restored"))
        restored.reset("restored-ep")
        target = Path(restored.workspace("restored-ep"))
        shutil.copytree(snapshot, target, dirs_exist_ok=True)
        result = restored.step("restored-ep", "read_file", {"path": "state.txt"})
        assert result.observation == "turn-boundary"
        restored.close("restored-ep")
    return {"atomic_snapshot": True, "new_session_rehydrated": True}


PROBES = (
    ("inflight_prompt_is_lost", probe_inflight_prompt_is_lost),
    ("ready_batch_is_unrecoverable", probe_ready_batch_is_unrecoverable),
    ("live_rollout_cannot_reattach", probe_live_rollout_cannot_reattach),
    ("localenv_session_cannot_reattach", probe_localenv_session_cannot_reattach),
    ("tool_effect_has_no_journal", probe_tool_effect_has_no_journal),
    ("workspace_snapshot_is_feasible", probe_workspace_snapshot_is_feasible),
)


def main() -> int:
    results = []
    failed = False
    for name, probe in PROBES:
        try:
            detail = probe()
            results.append({"name": name, "ok": True, "detail": detail})
            print(f"PASS {name}: {json.dumps(detail, ensure_ascii=False, sort_keys=True)}")
        except Exception as exc:
            failed = True
            results.append(
                {
                    "name": name,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    report = {"gate": "RolloutResume-Gate0", "ok": not failed, "results": results}
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
