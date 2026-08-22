"""Training-side Env: ms-agent's tools, executed inside a remote sandbox.

Pairs with ``sandbox_server/tool_server.py``. That server runs in the microVM
and owns the real ms-agent ``ToolManager``; this class is the client. Nothing
here knows what ``edit_file`` or ``shell_executor`` do -- it forwards a tool
call and returns whatever ms-agent produced, so the behaviour the policy is
trained against is the behaviour it will meet at serving time.

Lifecycle mirrors :class:`twinkle_agentic.envs.AgentEnv`: one sandbox per
episode, created on ``reset`` and killed on ``close``. On top of that, ``reset``
uploads the agent yaml and the server script from the training host and waits
for the runtime to come up, so iterating on either one does not mean rebuilding
the template image.

Transport is HTTP, driven by ``curl`` over the sandbox's command channel rather
than a forwarded port. It costs one process spawn per turn -- noise next to a
shell command -- and in exchange depends only on ``commands.run`` and
``files.write``, which is the surface every e2b-compatible backend implements
the same way.
"""
import json
import os
import posixpath
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle import get_logger
from twinkle_agentic.envs.base import Env, StepResult

logger = get_logger()

__all__ = ['RemoteMsAgentToolEnv']

# Marker used to recover an exit status from a tool that only returns text.
_RC_MARK = '__TWINKLE_RC__'
_RC_RE = re.compile(rf'{_RC_MARK}:(-?\d+)')

_PY_WRAPPER = """\
import sys, traceback
try:
{body}
except SystemExit as _e:
    # Print the status and stop -- do NOT re-raise. SystemExit inherits from
    # BaseException, so ms-agent's `except Exception` around the exec does not
    # catch it; letting it escape kills the whole tool server process, and the
    # sandbox is shared by every task in the run. `else` is already skipped
    # because the exception was handled, so nothing further is needed.
    _c = _e.code
    print('{mark}:%d' % (0 if _c is None else _c if isinstance(_c, int) else 1))
except BaseException:
    traceback.print_exc()
    print('{mark}:1')
else:
    print('{mark}:0')
"""

_REMOTE_DIR = '/opt/rsi'
_LOCAL_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sandbox_server', 'tool_server.py')


class RemoteMsAgentToolEnv(Env):
    """Run one episode's ms-agent tool calls inside a dedicated sandbox.

    Args:
        template: AgentENV/e2b template name, built by ``sandbox_server/install.sh``.
        config_path: ms-agent yaml on the *training host*. Uploaded on every
            reset, so this file is the single source of truth for both sides.
        api_url: AgentENV server base URL. Falls back to ``E2B_API_URL``.
        api_key: API key; AgentENV accepts any non-empty string.
        port: port the tool server listens on inside the sandbox.
        workspace: ``config.output_dir`` inside the sandbox.
        sandbox_timeout: sandbox idle timeout, in seconds. Must outlast a whole
            episode plus the checks that run after it.
        command_timeout: per-request timeout for a tool call, in seconds.
        boot_timeout: how long to wait for the runtime to answer ``/health``.
            ms-agent's import plus tool construction dominates this.
        max_observation_chars: truncate a tool result before it becomes a
            message. A single ``grep`` can otherwise fill the context window.
    """

    def __init__(
        self,
        template: str,
        config_path: str,
        *,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        port: int = 8900,
        workspace: str = '/workspace',
        sandbox_timeout: int = 900,
        command_timeout: int = 180,
        boot_timeout: int = 300,
        max_observation_chars: int = 8000,
    ):
        if not template:
            raise ValueError("RemoteMsAgentToolEnv requires 'template'; build one with "
                             'sandbox_server/install.sh')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f'agent config not found: {config_path}')
        self._template = template
        self._config_path = config_path
        self._api_url = api_url
        self._api_key = api_key
        self._port = port
        self.workspace = workspace
        self._sandbox_timeout = sandbox_timeout
        self._command_timeout = command_timeout
        self._boot_timeout = boot_timeout
        self.max_observation_chars = max_observation_chars
        self._sandbox = None
        self._schemas: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------ Env

    def reset(self, trajectory: Optional[Dict[str, Any]] = None) -> StepResult:
        """Boot a sandbox and bring ms-agent's tool runtime up inside it."""
        self.close()
        self._sandbox = self._create_sandbox()
        self._upload()
        self._start_server()
        self._await_ready()
        self._schemas = None
        return StepResult(observation='')

    def step(self, tool_name: str, arguments: Dict[str, Any] = None) -> StepResult:
        return self.step_batch([(tool_name, arguments or {})])[0]

    def step_batch(self, calls: Sequence[Tuple[str, Dict[str, Any]]]) -> List[StepResult]:
        """Send a turn's calls as one request; the server runs them together.

        Batching matters twice over: it is one sandbox round trip instead of
        several, and it keeps ms-agent's own ``parallel_call_tool`` semantics
        rather than serialising what production would run concurrently.
        """
        calls = list(calls)
        if not calls:
            return []
        payload = {
            'calls': [{
                'tool_name': name,
                'arguments': args or {}
            } for name, args in calls],
            'timeout': self._command_timeout,
        }
        try:
            body = self._rpc('/call', payload)
            results = body.get('results') or []
        except Exception as e:  # noqa
            # A dead sandbox must not kill the training step: report it as an
            # observation and let the episode play out (and score zero).
            logger.warning(f'RemoteMsAgentToolEnv call failed: {e}')
            results = [{'observation': f'Tool runtime unreachable: {e}'} for _ in calls]
        if len(results) != len(calls):
            results = (results + [{'observation': 'Tool runtime returned no result'}] * len(calls))[:len(calls)]
        return [StepResult(observation=self._truncate(r.get('observation') or '')) for r in results]

    def close(self) -> None:
        if self._sandbox is None:
            return
        sandbox_id = getattr(self._sandbox, 'sandbox_id', None)
        try:
            self._sandbox.kill()
        except Exception as e:  # noqa # best-effort: the backend evicts on timeout anyway
            logger.warning(f'failed to kill sandbox {sandbox_id}: {e}')
        finally:
            self._sandbox = None

    # ------------------------------------------------------------ tool names

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Schemas from the runtime that will execute them.

        These go straight into the prompt. Sourcing them from the executor
        rather than from a second local ms-agent is what makes it impossible
        for the advertised contract and the running code to disagree.
        """
        if self._schemas is None:
            self._schemas = list((self._rpc('/tools', None) or {}).get('tools') or [])
        return list(self._schemas)

    def tool_names(self) -> List[str]:
        names = []
        for schema in self.tool_schemas():
            name = (schema.get('function') or {}).get('name')
            if name:
                names.append(str(name))
        return names

    def resolve_tool(self, name: str) -> str:
        """Map a plain tool name onto the runtime's own spelling.

        ms-agent namespaces its tools as ``{server}---{tool}``, so a caller that
        asks for ``shell_executor`` means ``code_executor---shell_executor``.
        An unknown name raises instead of being passed through: a mistyped tool
        comes back as a failed call, which for a checker is indistinguishable
        from a failed check, and a whole GRPO group would silently score zero.
        """
        names = self.tool_names()
        if name in names:
            return name
        matches = [n for n in names if n.rsplit('---', 1)[-1] == name]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise ValueError(f'no registered tool named {name!r}; available: {names}')
        raise ValueError(f'{name!r} is ambiguous across servers: {matches}')

    # ------------------------------------------------------- for the checker

    def runner(self, shell_tool: str = 'shell_executor', python_tool: str = 'python_executor'):
        """A ``result_check`` runner that executes inside this episode's sandbox.

        Verification has to see the filesystem the agent actually wrote to, so
        the check goes back through the same tools rather than a local
        subprocess. Those tools return prose, not an exit status, so the command
        is made to print a marker and the status is read back out of the output.
        """
        shell_name = self.resolve_tool(shell_tool)
        python_name = self.resolve_tool(python_tool)

        def _run(source: str, interpreter: str) -> Tuple[int, str]:
            if interpreter == 'python':
                body = '\n'.join('    ' + line for line in source.splitlines()) or '    pass'
                code = _PY_WRAPPER.format(body=body, mark=_RC_MARK)
                out = self.step(python_name, {'code': code}).observation
            else:
                out = self.step(shell_name, {'command': f'{source}\necho "{_RC_MARK}:$?"'}).observation
            match = _RC_RE.search(out or '')
            if match is None:
                # No marker means the tool itself failed (timeout, sandbox down)
                # rather than the check failing; report non-zero and keep output.
                return 1, out or 'check produced no output and no exit marker'
            return int(match.group(1)), _RC_RE.sub('', out or '').strip()

        return _run

    def download_workspace(self, dest: str, max_files: int = 200, max_bytes: int = 1 << 20) -> str:
        """Copy the episode's files out of the sandbox for the ``file_*`` checks.

        Those checks read from an ordinary local directory, which is the right
        interface for a generic verifier but cannot see inside a microVM. The
        episode is over by the time this runs, so a snapshot is equivalent to
        the live filesystem -- and the shell/python checks still go through
        :meth:`runner`, against the sandbox itself.

        Files above ``max_bytes`` are skipped: a check that needs to look at a
        100MB artifact wants a command, not a copy.
        """
        os.makedirs(dest, exist_ok=True)
        listing = self._sandbox.commands.run(
            f"find {self.workspace} -type f -size -{max(1, max_bytes // 1024)}k "
            f'-printf "%P\\n" 2>/dev/null | head -n {max_files}',
            timeout=60)
        for rel in (listing.stdout or '').splitlines():
            rel = rel.strip()
            if not rel:
                continue
            local = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            try:
                content = self._sandbox.files.read(posixpath.join(self.workspace, rel))
            except Exception as e:  # noqa # an unreadable file fails its own check, not the batch
                logger.debug(f'could not fetch {rel} from sandbox: {e}')
                continue
            mode = 'wb' if isinstance(content, (bytes, bytearray)) else 'w'
            with open(local, mode) as f:
                f.write(content)
        return dest

    # -------------------------------------------------------------- private

    def _create_sandbox(self):
        try:
            from e2b import Sandbox
        except ImportError as e:
            raise ImportError('RemoteMsAgentToolEnv needs the e2b SDK: pip install e2b') from e
        if self._api_url:
            os.environ['E2B_API_URL'] = self._api_url
            os.environ.setdefault('E2B_SANDBOX_URL', self._api_url)
        if self._api_key:
            os.environ['E2B_API_KEY'] = self._api_key
        os.environ.setdefault('E2B_API_KEY', 'dummy')
        os.environ.setdefault('E2B_ACCESS_TOKEN', 'dummy')
        # AgentENV issues no keys, but the SDK asserts the key looks like
        # ``e2b_[0-9a-f]+`` before sending anything. This is the SDK's own
        # opt-out for deployments that do not mint e2b-format keys.
        os.environ.setdefault('E2B_VALIDATE_API_KEY', 'false')
        # ``Sandbox.create``, not ``Sandbox(...)``: since e2b 2.x the constructor
        # takes connection options for an *existing* sandbox and rejects
        # ``template``, while the classmethod is what provisions a new one.
        return Sandbox.create(template=self._template, timeout=self._sandbox_timeout)

    def _upload(self) -> None:
        """Push the yaml and the server script into the sandbox.

        Uploading beats baking them into the image: the training host's copy is
        authoritative, so editing a tool line-up is a restart rather than a
        template rebuild, and the two halves cannot fall out of sync.
        """
        with open(self._config_path, encoding='utf-8') as f:
            self._sandbox.files.write(f'{_REMOTE_DIR}/rsi_agent.yaml', f.read())
        with open(_LOCAL_SERVER, encoding='utf-8') as f:
            self._sandbox.files.write(f'{_REMOTE_DIR}/tool_server.py', f.read())

    def _start_server(self) -> None:
        command = (f'python {_REMOTE_DIR}/tool_server.py '
                   f'--config {_REMOTE_DIR}/rsi_agent.yaml '
                   f'--workspace {self.workspace} --port {self._port}')
        try:
            self._sandbox.commands.run(command, background=True)
        except TypeError:
            # Older SDKs have no `background`; detach with setsid so the server
            # outlives the command that launched it.
            self._sandbox.commands.run(
                f'mkdir -p {self.workspace} && setsid nohup {command} '
                f'> /tmp/tool_server.log 2>&1 < /dev/null &',
                timeout=30)

    def _await_ready(self) -> None:
        """Poll ``/health`` until the runtime answers, then fail loudly.

        Silence here is worth an exception: a sandbox whose tools never came up
        answers every call with an error, the episode scores zero, and the whole
        GRPO group looks like a hard task rather than a broken environment.
        """
        deadline = time.time() + self._boot_timeout
        last = ''
        while time.time() < deadline:
            try:
                if (self._rpc('/health', None, timeout=10) or {}).get('status') == 'ok':
                    return
            except Exception as e:  # noqa
                last = str(e)
            time.sleep(2)
        log = ''
        try:
            log = (self._sandbox.commands.run('tail -n 40 /tmp/tool_server.log', timeout=20).stdout or '')
        except Exception:  # noqa
            pass
        raise RuntimeError(f'ms-agent tool runtime did not come up within {self._boot_timeout}s '
                           f'(last error: {last})\n{log}')

    def _rpc(self, path: str, payload: Optional[Dict[str, Any]], timeout: Optional[int] = None) -> Dict[str, Any]:
        """One request to the in-sandbox server, via curl on the command channel.

        The body is written to a file rather than inlined: tool arguments carry
        arbitrary source code, and no amount of shell quoting survives that
        reliably.
        """
        seconds = timeout or self._command_timeout
        if payload is None:
            command = f'curl -sS -m {seconds} http://127.0.0.1:{self._port}{path}'
        else:
            request = f'{_REMOTE_DIR}/request.json'
            self._sandbox.files.write(request, json.dumps(payload, ensure_ascii=False))
            command = (f'curl -sS -m {seconds} -X POST -H "Content-Type: application/json" '
                       f'--data-binary @{request} http://127.0.0.1:{self._port}{path}')
        result = self._sandbox.commands.run(command, timeout=seconds + 30)
        stdout = (getattr(result, 'stdout', '') or '').strip()
        if not stdout:
            raise RuntimeError(f'empty response from {path}: {getattr(result, "stderr", "")}')
        return json.loads(stdout)

    def _truncate(self, text: str) -> str:
        limit = self.max_observation_chars
        if limit and len(text) > limit:
            return f'{text[:limit]}\n...[truncated {len(text) - limit} chars]'
        return text
