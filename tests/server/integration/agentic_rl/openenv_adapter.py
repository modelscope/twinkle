"""OpenEnv functional acceptance using session-local virtual files and remote AST execution.

This is not an OS filesystem or microVM and does not certify AgentENV isolation.
"""

from __future__ import annotations
import ast
import copy
import posixpath
import shlex
import threading
import uuid

import json

from twinkle_agentic.envs import OpenEnvClient
from twinkle_agentic.envs.base import DEFAULT_TOOLS, Env, StepResult


class OpenEnvTaskError(RuntimeError):
    """Unconfirmed communication, execution, or verifier state must not become a low reward."""


class OpenEnvTaskEnv(Env):
    def __init__(self, task, *, base_url: str, env_name: str = "coding_env", client=None):
        self.task = task
        self._env = client or OpenEnvClient(env_name=env_name, base_url=base_url, message_timeout_s=120)
        self._lock = threading.RLock()
        self._uncertain: OpenEnvTaskError | None = None
        self._episode_id = None
        self.closed = False

    @property
    def sandbox_id(self):
        # Use the server's episode identity rather than fabricating an isolation receipt.
        return self._episode_id

    @property
    def workspace(self):
        return "/workspace"

    def tools(self):
        tools = copy.deepcopy(DEFAULT_TOOLS)
        for tool in tools:
            name = tool["function"]["name"]
            if name == "run_command":
                tool["function"]["description"] = (
                    "Run a Python script in the virtual workspace, e.g. python smoke_test.py. "
                    "Only python/python3 script commands and python -c are supported; no shell or OS access."
                )
            else:
                tool["function"]["description"] += " This is a virtual file inside the OpenEnv session."
        return tools

    def _fail(self, stage, error=None):
        if self._uncertain is None:
            detail = f" ({type(error).__name__})" if error is not None else ""
            self._uncertain = OpenEnvTaskError(f"OpenEnv {stage} unconfirmed, session={self.sandbox_id}{detail}")
        raise self._uncertain from error

    def _execute(self, source, stage):
        with self._lock:
            self.ensure_idle()
            if self.closed:
                raise OpenEnvTaskError("OpenEnv session is closed")
            try:
                result = self._env.execute({"code": source})
                observation = result.observation
                if type(observation.exit_code) is not int:
                    self._fail(stage)
                # A remote interpreter timeout may leave its evaluation thread running.
                if observation.exit_code and (
                    "TimeoutError" in observation.stderr or "timed out" in observation.stderr.lower()
                ):
                    self._fail(stage + "_execution_timeout")
                return observation
            except OpenEnvTaskError:
                raise
            except Exception as error:
                # An SDK timeout does not prove the remote AST execution stopped.
                self._fail(stage, error)

    def reset(self, trajectory=None):
        with self._lock:
            self.ensure_idle()
            try:
                self._env.reset()
                self._episode_id = self._env.client.state().episode_id
            except Exception as error:
                self._fail("reset", error)
            if not self.sandbox_id:
                self._fail("session_identity")
            result = self._execute('import json\n__twinkle_files = {}\nprint("workspace ready")', "setup")
            if result.exit_code:
                self._fail("setup")
            return StepResult(observation="Virtual workspace ready", done=False)

    def ensure_idle(self):
        # The lock drains synchronous calls; communication failures remain unconfirmed.
        with self._lock:
            if self._uncertain is not None:
                raise self._uncertain

    def close(self):
        with self._lock:
            self.ensure_idle()
            if not self.closed:
                # Bypass best-effort OpenEnvClient.close so close failures remain visible.
                try:
                    self._env.client.close()
                except Exception as error:
                    self._fail("close", error)
                self.closed = True

    def _path(self, value):
        if not isinstance(value, str) or not value:
            raise ValueError("path is required")
        path = posixpath.normpath(posixpath.join(self.workspace, value))
        if not path.startswith(self.workspace + "/"):
            raise ValueError("files must stay inside /workspace")
        return path[len(self.workspace) + 1 :]

    def _value(self, expression, stage):
        marker = "TWINKLE_VALUE_" + uuid.uuid4().hex + "="
        result = self._execute(f"import json\nprint({marker!r} + json.dumps({expression}))", stage)
        if result.exit_code:
            raise ValueError("virtual file is missing or cannot be read")
        rows = [row[len(marker) :] for row in result.stdout.splitlines() if row.startswith(marker)]
        try:
            if len(rows) != 1:
                raise ValueError("missing result marker")
            return json.loads(rows[0])
        except (ValueError, TypeError) as error:
            self._fail(stage, error)

    def read_file(self, path):
        value = self._value(f"__twinkle_files[{self._path(path)!r}]", "read_file")
        if not isinstance(value, str):
            self._fail("read_file_result")
        return value

    def _run_command(self, command):
        words = shlex.split(command)
        if words[:3] == ["cd", self.workspace, "&&"]:
            words = words[3:]
        if len(words) == 3 and words[0] in {"python", "python3"} and words[1] == "-c":
            return self._execute(words[2], "run_command")
        if len(words) != 2 or words[0] not in {"python", "python3"}:
            raise ValueError("use python smoke_test.py or python -c; shell commands are not available")
        path = self._path(words[1])
        source = self.read_file(path)
        if path == "smoke_test.py":
            # The AST interpreter has no file imports; execute current source with the smoke tests.
            module = self.task.filename.removesuffix(".py")
            source = "\n".join(line for line in source.splitlines() if not line.startswith(f"from {module} import "))
            source = self.read_file(self.task.filename) + "\n" + source
        return self._execute(source + '\nprint("script completed")', "run_command")

    def step(self, tool_name, arguments=None):
        arguments = arguments or {}
        try:
            if tool_name == "read_file":
                return StepResult(observation=self.read_file(arguments.get("path")), done=False)
            if tool_name == "write_file":
                path = self._path(arguments.get("path"))
                content = arguments.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                result = self._execute(f'__twinkle_files[{path!r}] = {content!r}\nprint("file written")', "write_file")
                if result.exit_code:
                    self._fail("write_file")
            elif tool_name == "run_command":
                result = self._run_command(arguments.get("command", ""))
            else:
                raise ValueError(f"unsupported task tool {tool_name!r}")
            output = (result.stdout + "\n" + result.stderr).strip()
            return StepResult(observation=output[-4096:], done=False)
        except ValueError as error:
            # Deterministic refusals and confirmed missing files are recoverable model errors.
            return StepResult(observation=f"Error: {error}", done=False)

    def verify(self):
        if not self.task.hidden_cases:
            self._fail("verify_cases")
        source = self.read_file(self.task.filename)
        function = self.task.function
        if not function.isidentifier():
            self._fail("function_name")
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return 0.0
        if not any(isinstance(node, ast.FunctionDef) and node.name == function for node in tree.body):
            return 0.0
        fresh_name = "twinkle_candidate_" + uuid.uuid4().hex

        class RenameCandidate(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name == function:
                    node.name = fresh_name
                return self.generic_visit(node)

            def visit_Name(self, node):
                if node.id == function:
                    node.id = fresh_name
                return node

        # Use a fresh callable instead of masking a name in the AST executor's persistent state.
        # Parsing/renaming is local; execution of the candidate still happens only on OpenEnv.
        source = ast.unparse(RenameCandidate().visit(tree))
        loaded = self._execute(source, "verify_candidate")
        if loaded.exit_code:
            return 0.0
        passed = 0
        for values, expected in self.task.hidden_cases:
            marker = "TWINKLE_CASE_" + uuid.uuid4().hex + "="
            result = self._execute(f"print({marker!r} + str({fresh_name}({values!r}) == {expected!r}))", "verify_case")
            if result.exit_code:
                # Candidate exceptions are completed failures; _execute raises transport errors.
                continue
            rows = [row[len(marker) :] for row in result.stdout.splitlines() if row.startswith(marker)]
            if len(rows) != 1 or rows[0] not in {"True", "False"}:
                self._fail("verify_result")
            passed += rows[0] == "True"
        return passed / len(self.task.hidden_cases)
