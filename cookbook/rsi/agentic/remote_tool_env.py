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
import copy
import json
import os
import posixpath
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle import get_logger
from twinkle_agentic.envs.base import Env, StepResult

logger = get_logger()

__all__ = ['RemoteMsAgentToolEnv']

# Marker used to recover an exit status from a tool that only returns text.
_RC_MARK = '__TWINKLE_RC__'
_RC_RE = re.compile(rf'{_RC_MARK}:(-?\d+)')

# The check script is passed through as a string and compiled under its own
# filename rather than indented into the `try` below. Indenting shifted every
# line by the two lines of preamble, so a traceback said "line 27" about a
# 25-line script and "line 15" about a comment -- the one piece of information a
# reader needs to see which assertion failed pointed at the wrong assertion, or
# past the end of the file. `<check>` in the traceback is that script, line for
# line, and the frames above it are this wrapper's.
_PY_WRAPPER = """\
import sys, io, traceback
_tw_check_src = {body}
_tw_buf = io.StringIO()
_tw_out = sys.stdout
sys.stdout = _tw_buf
_tw_rc = 0
try:
    _tw_ns = {{'__name__': '__main__'}}
    exec(compile(_tw_check_src, '<check>', 'exec'), _tw_ns, _tw_ns)
except SystemExit as _e:
    # Print the status and stop -- do NOT re-raise. SystemExit inherits from
    # BaseException, so ms-agent's `except Exception` around the exec does not
    # catch it; letting it escape kills the whole tool server process, and the
    # sandbox is shared by every task in the run.
    _c = _e.code
    _tw_rc = 0 if _c is None else _c if isinstance(_c, int) else 1
except BaseException:
    traceback.print_exc(file=_tw_buf)
    _tw_rc = 1
finally:
    sys.stdout = _tw_out
# Marker FIRST, then the body. The executor truncates a tool observation at
# ~8KB, counted from the start; a marker printed after a large body (a rich
# workspace snapshot, say) is silently cut off, `runner` then finds no marker
# and reports exit 1 -- which read as an empty workspace and threw the task
# away. Emitted before the body, the marker always survives; only the tail of
# the body is ever lost.
print('{mark}:%d' % _tw_rc)
sys.stdout.write(_tw_buf.getvalue())
"""

_REMOTE_DIR = '/opt/rsi'
# Where the in-sandbox runtime's stdout/stderr goes. Read back by `server_log`.
SERVER_LOG = '/tmp/tool_server.log'
# Seconds the transport gets beyond the server's own budget, so that a slow call
# is answered by the layer that knows which call was slow. curl waits this much
# longer than the server may spend, and the command channel that much again.
# Anything smaller than the gap between two deadlines is a race, and the client
# wins it -- which turns one slow call into "runtime unreachable" for the whole
# turn.
_RPC_HEADROOM = 60
_LOCAL_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sandbox_server', 'tool_server.py')

# No ms-agent code is uploaded. ``rsi_agent.yaml`` turns on two safety switches
# -- ``safety_rules.unrestricted_removal`` and ``safety_rules.allow_write_globs``
# -- that ms-agent's ``SafetyConfig.from_dict`` does not implement and silently
# ignores. This used to be handled by copying a patched ``ms_agent/permission``
# package into every sandbox, which meant carrying a fork of a dependency that
# twinkle supports as a harness, and re-merging it forever. ``tool_server.py``
# now applies the same two relaxations as a runtime patch inside the sandbox
# (``_patch_permission``), next to the ``python_executor`` patch that was already
# there, so the released ms-agent is used as-is on both sides.


def tool_payload(observation: str) -> str:
    """The command output inside an ms-agent tool observation, or the text as-is.

    The executor tools answer with a JSON envelope
    (``{"success": ..., "output": ..., "error": ...}``). Fed to a model as-is it
    reads as a wall of metadata around the one part that matters, and a model
    asked to describe a directory from it tends to trust its own recollection
    instead. Callers that want the output *as data* -- a file listing, a computed
    value -- go through here; callers that only want an exit status do not need it.
    """
    text = (observation or '').strip()
    if not text.startswith('{'):
        return observation or ''
    try:
        body = json.loads(text)
    except ValueError:
        return observation or ''
    if not isinstance(body, dict) or 'output' not in body:
        return observation or ''
    payload = body.get('output') or ''
    error = body.get('error')
    if error:
        payload = f'{payload}\n{error}'.strip()
    return payload


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
        command_timeout: how long the in-sandbox server may spend on a turn's
            tool calls, in seconds. The transport around it is given headroom on
            top -- see :meth:`_rpc`.
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
        # Short advertised name -> the runtime's namespaced one. Filled by
        # _load_schemas, which is the only thing that knows which short names are
        # unambiguous.
        self._short_to_full: Dict[str, str] = {}
        self._deadline = 0.0
        self.n_recoveries = 0

    # ------------------------------------------------------------------ Env

    def reset(self, trajectory: Optional[Dict[str, Any]] = None) -> StepResult:
        """Boot a sandbox and bring ms-agent's tool runtime up inside it."""
        self.close()
        self._sandbox = self._create_sandbox()
        self._deadline = time.time() + self._sandbox_timeout
        self._upload()
        self._start_server()
        self._await_ready()
        self._schemas = None
        self._short_to_full = {}
        return StepResult(observation='')

    def healthy(self) -> bool:
        """Does the tool runtime answer right now?"""
        if self._sandbox is None:
            return False
        try:
            return (self._rpc('/health', None, timeout=10) or {}).get('status') == 'ok'
        except Exception:  # noqa
            return False

    def ensure_ready(self) -> bool:
        """Re-establish the sandbox if its runtime has gone away. True if it did.

        For the callers that hold one sandbox across many episodes, losing it --
        evicted, timed out, runtime crashed -- otherwise ends the whole run. This
        is safe to call only where the workspace is about to be discarded
        anyway: a mid-episode rebuild would silently swap the state the episode
        is being judged on for an empty directory, so recovery is offered as an
        explicit call rather than a retry hidden inside every tool dispatch.

        Recoveries are counted in ``n_recoveries`` so a run can report how often
        this happened instead of hiding it.
        """
        if self.healthy():
            return False
        self.n_recoveries += 1
        logger.warning(f'tool runtime unreachable; rebuilding the sandbox '
                       f'(recovery #{self.n_recoveries})')
        self.reset()
        return True

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
                'tool_name': self._dispatch_name(name),
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
            self._deadline = 0.0

    def _keep_alive(self) -> None:
        """Push the sandbox's expiry back while it is still being used.

        ``sandbox_timeout`` is a lifetime from creation, not an idle timer, so a
        caller that keeps one sandbox for a long run would lose it mid-run no
        matter how busy it was. Extended once past the halfway mark rather than on
        every call: this is an extra HTTP round trip, and tool dispatch is already
        the slow part of a turn.
        """
        if self._sandbox is None:
            return
        if time.time() < self._deadline - self._sandbox_timeout / 2:
            return
        try:
            self._sandbox.set_timeout(self._sandbox_timeout)
            self._deadline = time.time() + self._sandbox_timeout
        except Exception as e:  # noqa
            logger.warning(f'failed to extend sandbox timeout: {e}')

    # ------------------------------------------------------------ tool names

    def _load_schemas(self) -> None:
        """Fetch the runtime's schemas and shorten the names it advertises.

        ms-agent namespaces every tool as ``{server}---{tool}``, and a 4B policy
        spends calls on that prefix: across three arms it wrote a bare
        ``shell_executor`` 7 times, each one refused with "unknown tool ... Did
        you mean 'code_executor---shell_executor'?" -- a whole turn burnt on
        punctuation. Since the prefix carries no information the model can act
        on (nothing here has two servers offering the same tool), the advertised
        name drops it, and :meth:`step_batch` puts it back before dispatch.

        This is not the same as accepting a wrong name and fixing it up: the
        model is shown ``shell_executor`` and calls ``shell_executor``, so what
        it learns to emit is what the schema promised. A name that would collide
        keeps its prefix, in both directions, rather than becoming ambiguous.
        """
        raw = list((self._rpc('/tools', None) or {}).get('tools') or [])
        full_names = [(t.get('function') or {}).get('name') for t in raw]
        counts: Dict[str, int] = {}
        for name in full_names:
            if name:
                counts[str(name).rsplit('---', 1)[-1]] = counts.get(
                    str(name).rsplit('---', 1)[-1], 0) + 1
        self._short_to_full = {}
        schemas = []
        for schema in raw:
            schema = copy.deepcopy(schema)
            fn = schema.get('function') or {}
            full = str(fn.get('name') or '')
            short = full.rsplit('---', 1)[-1]
            if full and counts.get(short) == 1 and short != full:
                fn['name'] = short
                self._short_to_full[short] = full
            schemas.append(schema)
        self._schemas = schemas

    def _dispatch_name(self, name: str) -> str:
        """The runtime's own spelling for a name taken from a tool call.

        Usually the map is already there, because the schemas were advertised
        before anything could be called. When it is not, fetching it must not be
        able to raise: a dead sandbox has to come back through ``step_batch`` as
        an observation the episode survives, not as an exception from name
        lookup. An unmapped name passes through as-is, which is also what a
        caller using the runtime's full spelling wants.
        """
        if not self._short_to_full and self._schemas is None:
            try:
                self._load_schemas()
            except Exception as e:  # noqa
                logger.warning(f'could not load tool names for dispatch: {e}')
                return name
        return self._short_to_full.get(name, name)

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """Schemas from the runtime that will execute them.

        These go straight into the prompt. Sourcing them from the executor
        rather than from a second local ms-agent is what makes it impossible
        for the advertised contract and the running code to disagree.
        """
        if self._schemas is None:
            self._load_schemas()
        return list(self._schemas)

    def tool_names(self) -> List[str]:
        names = []
        for schema in self.tool_schemas():
            name = (schema.get('function') or {}).get('name')
            if name:
                names.append(str(name))
        return names

    def resolve_tool(self, name: str) -> str:
        """Map any spelling of a tool onto the one this Env advertises.

        Advertised names are short (see :meth:`_load_schemas`), so this returns
        ``shell_executor``, not ``code_executor---shell_executor``. Both spellings
        go in: callers written before the names were shortened pass the
        namespaced one, and a stale spelling should not be the thing that fails.

        An unknown name raises instead of being passed through: a mistyped tool
        comes back as a failed call, which for a checker is indistinguishable
        from a failed check, and a whole GRPO group would silently score zero.
        """
        names = self.tool_names()
        if name in names:
            return name
        # A namespaced name for a tool advertised short.
        suffix = name.rsplit('---', 1)[-1]
        if suffix in names and self._short_to_full.get(suffix) == name:
            return suffix
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
                code = _PY_WRAPPER.format(body=repr(source), mark=_RC_MARK)
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

        Only twinkle's own two files travel. The safety relaxations
        ``rsi_agent.yaml`` asks for are applied by ``tool_server.py`` at runtime,
        so no ms-agent source is shipped or overwritten here.
        """
        with open(self._config_path, encoding='utf-8') as f:
            self._sandbox.files.write(f'{_REMOTE_DIR}/rsi_agent.yaml', f.read())
        with open(_LOCAL_SERVER, encoding='utf-8') as f:
            self._sandbox.files.write(f'{_REMOTE_DIR}/tool_server.py', f.read())

    def _start_server(self) -> None:
        """Launch the tool runtime in the background, with its output on disk.

        ``background=True`` is what detaches it; the redirect is what makes a
        later death diagnosable. Without the redirect the output lives on a
        command handle nobody keeps, so a runtime that dies mid-run reads only as
        a refused connection. Do not swap the redirect for a trailing ``&``:
        ``commands.run`` then waits out its own timeout instead of returning.

        ``-u`` rather than relying on the image: the template is built from a
        snapshot of a live sandbox, which keeps the filesystem but not the image
        config, so the Dockerfile's ``ENV PYTHONUNBUFFERED=1`` is not there. An
        unflushed buffer is the difference between a readable log and an empty
        one when the runtime dies.

        ``cd`` into the workspace, because ``python_executor`` runs ``exec()``
        inside this process (ms-agent's local_code_executor.py:657) rather than in
        a subprocess with its own cwd. Started from ``/``, as it was, a relative
        path in model code resolved against ``/`` while every other tool resolves
        against the workspace: measured in a live sandbox, ``write_file
        'a.txt'`` answered "Save file successfully" and the next python call got
        ``[Errno 2] No such file or directory: 'a.txt'``, with the file sitting in
        ``/workspace`` and python looking in ``/``. That single mismatch is 41 of
        ex7's 58 such failures, and it also hid files from the end-of-episode
        snapshot, which only lists the workspace. The python_executor patch in
        tool_server.py chdirs per call as well, so the two do not depend on each
        other.
        """
        command = (f'mkdir -p {self.workspace} && cd {self.workspace} && '
                   f'python -u {_REMOTE_DIR}/tool_server.py '
                   f'--config {_REMOTE_DIR}/rsi_agent.yaml '
                   f'--workspace {self.workspace} --port {self._port} '
                   f'> {SERVER_LOG} 2>&1')
        self._sandbox.commands.run(command, background=True)

    def server_log(self, lines: int = 40) -> str:
        """Tail the in-sandbox runtime log; '' if it cannot be read.

        Used when the runtime stops answering, which is the one moment its own
        output matters and the one moment an RPC cannot fetch it.
        """
        try:
            return (self._sandbox.commands.run(f'tail -n {lines} {SERVER_LOG}',
                                               timeout=20).stdout or '')
        except Exception:  # noqa
            return ''

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
            log = self.server_log(40)
        except Exception:  # noqa
            pass
        raise RuntimeError(f'ms-agent tool runtime did not come up within {self._boot_timeout}s '
                           f'(last error: {last})\n{log}')

    def _rpc(self, path: str, payload: Optional[Dict[str, Any]], timeout: Optional[int] = None) -> Dict[str, Any]:
        """One request to the in-sandbox server, via curl on the command channel.

        The body is written to a file rather than inlined: tool arguments carry
        arbitrary source code, and no amount of shell quoting survives that
        reliably.

        The file name carries a nonce because two threads can be in here at once.
        A fixed ``request.json`` made them overwrite each other between the write
        and the curl, so every concurrent call executed whichever payload landed
        last and each caller filed that one answer under its own call. That is
        how ex4's episode 8 came back with a glob listing as the result of a
        python script it never ran.

        curl is given ``_RPC_HEADROOM`` seconds more than the server is allowed to
        spend, and the command channel more again. They used to share one number,
        which meant that when a call ran long the client gave up in the same
        second the server was formulating its answer -- and the client wins that
        race, so a turn holding one slow call came back as "Tool runtime
        unreachable" for *every* call in it, including the ones that had finished.
        ex8's episode 23 is that: a shell command started an HTTP server, and the
        write_file beside it was reported as an unreachable runtime. With headroom
        the server's own per-call timeout message arrives instead.
        """
        seconds = timeout or self._command_timeout
        self._keep_alive()
        if payload is None:
            command = f'curl -sS -m {seconds + _RPC_HEADROOM} http://127.0.0.1:{self._port}{path}'
        else:
            request = f'{_REMOTE_DIR}/request-{uuid.uuid4().hex}.json'
            self._sandbox.files.write(request, json.dumps(payload, ensure_ascii=False))
            command = (f'curl -sS -m {seconds + _RPC_HEADROOM} -X POST -H "Content-Type: application/json" '
                       f'--data-binary @{request} http://127.0.0.1:{self._port}{path}; '
                       f'rm -f {request}')
        result = self._sandbox.commands.run(command, timeout=seconds + 2 * _RPC_HEADROOM)
        stdout = (getattr(result, 'stdout', '') or '').strip()
        if not stdout:
            raise RuntimeError(f'empty response from {path}: {getattr(result, "stderr", "")}')
        return json.loads(stdout)

    def _truncate(self, text: str) -> str:
        limit = self.max_observation_chars
        if limit and len(text) > limit:
            return f'{text[:limit]}\n...[truncated {len(text) - limit} chars]'
        return text
