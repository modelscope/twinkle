"""Deterministic adapter tests; real OpenEnv coverage is recorded separately."""

from __future__ import annotations
import ast
import threading
from types import SimpleNamespace

import json
import pytest

from .openenv_adapter import OpenEnvTaskEnv, OpenEnvTaskError


class Client:
    def __init__(self):
        self.client = self
        self.files = {}
        self.calls = []
        self.closed = False
        self.failure = None
        self.malformed = False

    def reset(self):
        self.files.clear()

    def state(self):
        return SimpleNamespace(episode_id="mock-server-episode")

    def close(self):
        self.closed = True

    def execute(self, action):
        source = action["code"]
        self.calls.append(source)
        if self.failure is not None:
            raise self.failure
        output, exit_code = "", 0
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript):
                key = ast.literal_eval(node.targets[0].slice)
                self.files[key] = ast.literal_eval(node.value)
        if "TWINKLE_VALUE_" in source or "TWINKLE_CASE_" in source:
            call = tree.body[-1].value
            marker = ast.literal_eval(call.args[0].left)
            if "TWINKLE_VALUE_" in source:
                key = ast.literal_eval(call.args[0].right.args[0].slice)
                if key not in self.files:
                    exit_code = 1
                else:
                    output = marker + json.dumps(self.files[key])
            else:
                output = marker + ("invalid" if self.malformed else "True")
        return SimpleNamespace(observation=SimpleNamespace(exit_code=exit_code, stdout=output, stderr=""))


def task():
    return SimpleNamespace(filename="solution.py", function="solve", hidden_cases=(([1], 1), ([2], 2)))


def environment():
    client = Client()
    env = OpenEnvTaskEnv(task(), base_url="http://not-used", client=client)
    env.reset()
    return env, client


def test_virtual_files_use_the_remote_session_and_server_identity():
    env, client = environment()
    env.step("write_file", {"path": "solution.py", "content": "def solve(values):\n    return values[0]"})
    assert env.read_file("/workspace/solution.py") == client.files["solution.py"]
    assert env.sandbox_id == "mock-server-episode"
    assert any("__twinkle_files" in source for source in client.calls)
    env.close()
    assert env.closed and client.closed


def test_local_refusals_do_not_submit_file_or_shell_operations():
    env, client = environment()
    before = len(client.calls)
    assert env.step("write_file", {"path": "../outside", "content": "x"}).observation.startswith("Error:")
    assert env.step("run_command", {"command": "echo unsafe"}).observation.startswith("Error:")
    assert len(client.calls) == before
    env.ensure_idle()
    env.close()


def test_transport_timeout_preserves_unknown_state_and_prevents_reuse():
    env, client = environment()
    client.failure = TimeoutError("unknown execution")
    with pytest.raises(OpenEnvTaskError):
        env.step("write_file", {"path": "solution.py", "content": "x"})
    client.failure = None
    count = len(client.calls)
    with pytest.raises(OpenEnvTaskError):
        env.ensure_idle()
    with pytest.raises(OpenEnvTaskError):
        env.step("read_file", {"path": "solution.py"})
    with pytest.raises(OpenEnvTaskError):
        env.close()
    assert len(client.calls) == count and not client.closed


def test_remote_interpreter_timeout_is_not_a_candidate_low_reward():
    env, client = environment()
    client.execute = lambda action: SimpleNamespace(
        observation=SimpleNamespace(exit_code=1, stdout="", stderr="TimeoutError: evaluation timed out")
    )
    with pytest.raises(OpenEnvTaskError, match="execution_timeout"):
        env.step("run_command", {"command": 'python -c "while True: pass"'})
    with pytest.raises(OpenEnvTaskError):
        env.ensure_idle()


def test_verifier_uses_fresh_callable_names_without_poisoning_smoke_state():
    env, client = environment()
    env.step("write_file", {"path": "solution.py", "content": "def solve(values):\n    return values[0]"})
    assert env.verify() == 1.0
    assert env.verify() == 1.0
    definitions = [source for source in client.calls if source.startswith("def twinkle_candidate_")]
    assert len(definitions) == len(set(definitions)) == 2
    assert not any("solve = None" in source for source in client.calls)
    env.step("write_file", {"path": "solution.py", "content": "unrelated = 1"})
    assert env.verify() == 0.0
    env.close()


def test_invalid_verifier_marker_is_not_a_low_reward():
    env, client = environment()
    env.step("write_file", {"path": "solution.py", "content": "def solve(values):\n    return values[0]"})
    client.malformed = True
    with pytest.raises(OpenEnvTaskError, match="verify_result"):
        env.verify()


def test_close_failure_is_visible_and_prevents_session_reuse():
    env, client = environment()

    def failed_close():
        raise ConnectionError("close acknowledgement lost")

    client.close = failed_close
    with pytest.raises(OpenEnvTaskError, match="close"):
        env.close()
    with pytest.raises(OpenEnvTaskError):
        env.ensure_idle()
    assert not env.closed


def test_close_waits_for_the_actual_blocking_operation():
    env, client = environment()
    started, release = threading.Event(), threading.Event()
    original = client.execute

    def blocked(action):
        started.set()
        assert release.wait(5)
        return original(action)

    client.execute = blocked
    operation = threading.Thread(target=lambda: env.step("write_file", {"path": "solution.py", "content": "x"}))
    operation.start()
    assert started.wait(5)
    closing = threading.Thread(target=env.close)
    closing.start()
    closing.join(0.2)
    assert closing.is_alive() and not client.closed
    release.set()
    operation.join(5)
    closing.join(5)
    assert not operation.is_alive() and not closing.is_alive()
    assert env.closed and client.closed
