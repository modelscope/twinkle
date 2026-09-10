# Copyright (c) ModelScope Contributors. All rights reserved.
"""A tool line-up that lives inside the sandbox, served by an uploaded runtime.

The client half of ``sandbox_server/server.py``. It knows how to put that server
in a machine, ask it what tools there are and forward calls to it -- and nothing
whatsoever about any agent framework: which tools exist, what they are called and
what an observation looks like are all answered by the runtime the server loads.

Adding a framework is therefore a ``sandbox_server/runtime_<name>.py`` and no
change here::

    env = AgentEnv(template='rsi-msagent',
                   tool_backend=RemoteTools(runtime='msagent', config='rsi_agent.yaml'))

Transport is HTTP over the sandbox's command channel via ``curl``, not a
forwarded port. It costs one process spawn per turn -- noise next to a shell
command -- and in exchange depends only on ``commands.run`` and ``files.write``,
which is the surface every e2b-compatible backend implements the same way.
"""
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from twinkle.data_format.message import Tool as ToolInfo
from twinkle.utils import get_logger
from .base import Env, ToolBackend, truncate_observation

logger = get_logger()

__all__ = ['RemoteTools']

# Where the uploaded files land inside the sandbox: never the workspace, which is
# read back as the episode's own output.
_REMOTE_DIR = '/opt/twinkle_tools'

# Where the in-sandbox server's stdout/stderr goes. Read back by `server_log`.
SERVER_LOG = '/tmp/twinkle_tool_server.log'

# Seconds the transport gets beyond the server's own budget, so that a slow call
# is answered by the layer that knows which call was slow. curl waits this much
# longer than the server may spend, and the command channel that much again.
# Anything smaller than the gap between two deadlines is a race, and the client
# wins it -- which turns one slow call into "runtime unreachable" for the whole
# turn.
_RPC_HEADROOM = 60

_LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sandbox_server')


class RemoteTools(ToolBackend):
    """Tools executed by a framework running inside the environment's machine.

    Args:
        runtime: which ``sandbox_server/runtime_<name>.py`` serves the tools.
        config: the framework's config file *on the training host*. Uploaded on
            every install, so this file is the single source of truth for both
            sides -- editing a tool line-up is a restart, not an image rebuild.
        port: port the server listens on inside the sandbox.
        call_timeout: how long the server may spend on one turn's calls, in
            seconds. The transport around it is given headroom on top.
        boot_timeout: how long to wait for the runtime to answer ``/health``.
            Importing a framework and constructing its tools dominates this.
        max_observation_chars: truncate a result before it becomes a message. A
            single ``grep`` can otherwise fill the context window.

    One of these belongs to one environment. It caches the schemas the machine
    reported, and a second environment installing over that would leave both
    advertising one machine's tools while calls went to the other's.
    """

    def __init__(
        self,
        runtime: str,
        config: str,
        *,
        port: int = 8900,
        call_timeout: int = 180,
        boot_timeout: int = 300,
        max_observation_chars: int = 8000,
    ):
        if not os.path.exists(config):
            raise FileNotFoundError(f'tool config not found: {config}')
        self._server = os.path.join(_LOCAL_DIR, 'server.py')
        self._runtime_file = os.path.join(_LOCAL_DIR, f'runtime_{runtime}.py')
        if not os.path.exists(self._runtime_file):
            available = sorted(f[8:-3] for f in os.listdir(_LOCAL_DIR) if f.startswith('runtime_'))
            raise ValueError(f'no runtime named {runtime!r}; available: {available}')
        self._runtime = runtime
        self._config = config
        self._port = port
        self._call_timeout = call_timeout
        self._boot_timeout = boot_timeout
        self.max_observation_chars = max_observation_chars
        self._schemas: Optional[List[ToolInfo]] = None
        self._owner: Optional[int] = None

    # --------------------------------------------------------- ToolBackend

    def install(self, env: Env) -> None:
        """Upload the server and its runtime, start it, wait for it to answer.

        Uploading beats baking the files into the image: the host's copy is
        authoritative, so iterating on either the config or the runtime is a
        restart rather than a template rebuild, and the two halves cannot fall
        out of sync.
        """
        if self._owner is not None and self._owner != id(env):
            raise RuntimeError(f'this {type(self).__name__} is already installed in another environment; '
                               'give each environment its own backend, or its tool schemas and its '
                               'tool calls will describe different machines')
        self._owner = id(env)
        self._schemas = None
        sandbox = self._require_sandbox(env)
        for local, remote in ((self._config, os.path.basename(self._config)),
                              (self._server, 'server.py'),
                              (self._runtime_file, os.path.basename(self._runtime_file))):
            with open(local, encoding='utf-8') as handle:
                sandbox.files.write(f'{_REMOTE_DIR}/{remote}', handle.read())
        self._start(env)
        self._await_ready(env)

    def tools(self) -> List[ToolInfo]:
        """Schemas from the runtime that will execute them.

        These go straight into the prompt. Sourcing them from the executor
        rather than from a second copy of the framework on the host is what
        makes it impossible for the advertised contract and the running code to
        disagree.
        """
        if self._schemas is None:
            raise RuntimeError(f'{type(self).__name__}.tools() before install(): '
                               'the tools are the ones the machine reports, so there is '
                               'nothing to advertise until a machine is up')
        return list(self._schemas)

    def call(self, env: Env, calls: Sequence[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """Send a turn's calls as one request; the runtime runs them together."""
        return self._dispatch(env, calls, self._call_timeout)

    def healthy(self, env: Env) -> bool:
        """Does the runtime answer right now?"""
        try:
            return (self._rpc(env, '/health', None, timeout=10) or {}).get('status') == 'ok'
        except Exception:  # noqa
            return False

    # -------------------------------------------------------------- public

    def server_log(self, env: Env, lines: int = 40) -> str:
        """Tail the in-sandbox server log; '' if it cannot be read.

        Used when the runtime stops answering, which is the one moment its own
        output matters and the one moment an RPC cannot fetch it.
        """
        try:
            return (self._require_sandbox(env).commands.run(f'tail -n {lines} {SERVER_LOG}',
                                                            timeout=20).stdout or '')
        except Exception:  # noqa
            return ''

    # ------------------------------------------------------------- private

    def _dispatch(self, env: Env, calls: Sequence[Tuple[str, Dict[str, Any]]], timeout: int) -> List[str]:
        """One request, with the per-call budget stated.

        A failure comes back as one observation per call rather than an
        exception: a dead machine must not end the training step, so the episode
        plays out (and scores zero) with the reason in the transcript.
        """
        calls = list(calls)
        if not calls:
            return []
        payload = {
            'calls': [{
                'tool_name': name,
                'arguments': args or {}
            } for name, args in calls],
            'timeout': timeout,
        }
        try:
            results = (self._rpc(env, '/call', payload, timeout=timeout) or {}).get('results') or []
        except Exception as e:  # noqa
            logger.warning(f'{type(self).__name__} call failed: {e}')
            results = [{'observation': f'Tool runtime unreachable: {e}'} for _ in calls]
        if len(results) != len(calls):
            results = (results + [{'observation': 'Tool runtime returned no result'}] * len(calls))[:len(calls)]
        return [self._truncate(r.get('observation') or '') for r in results]

    def _start(self, env: Env) -> None:
        """Launch the server in the background, with its output on disk.

        ``background=True`` is what detaches it; the redirect is what makes a
        later death diagnosable -- without it the output lives on a command
        handle nobody keeps, and a runtime that dies mid-run reads only as a
        refused connection. Do not swap the redirect for a trailing ``&``:
        ``commands.run`` then waits out its own timeout instead of returning.

        ``-u`` rather than relying on the image: a template built from a
        snapshot of a live sandbox keeps the filesystem but not the image
        config, so a Dockerfile's ``ENV PYTHONUNBUFFERED=1`` is not there. An
        unflushed buffer is the difference between a readable log and an empty
        one when the runtime dies.

        Started *in* the workspace, because a framework tool that runs code
        in-process inherits this cwd. From ``/``, a relative path in model code
        resolved against ``/`` while every other tool resolved against the
        workspace: measured in a live sandbox, ``write_file 'a.txt'`` answered
        "Save file successfully" and the next python call got ``[Errno 2] No
        such file or directory: 'a.txt'``, with the file sitting in
        ``/workspace`` and python looking in ``/``.
        """
        workspace = env.workspace
        command = (f'mkdir -p {workspace} && cd {workspace} && '
                   f'python -u {_REMOTE_DIR}/server.py --runtime {self._runtime} '
                   f'--config {_REMOTE_DIR}/{os.path.basename(self._config)} '
                   f'--workspace {workspace} --port {self._port} '
                   f'> {SERVER_LOG} 2>&1')
        self._require_sandbox(env).commands.run(command, background=True)

    def _await_ready(self, env: Env) -> None:
        """Poll ``/health`` until the runtime answers, then fail loudly.

        Silence here is worth an exception: a machine whose tools never came up
        answers every call with an error, the episode scores zero, and the whole
        GRPO group looks like a hard task rather than a broken environment.

        The schemas are read once it does answer, because that is the first
        moment there is anything to read -- and asking now means a boot that
        half-succeeded is reported here rather than by the first prompt that
        needed a tool list.
        """
        deadline = time.time() + self._boot_timeout
        last = ''
        while time.time() < deadline:
            try:
                if (self._rpc(env, '/health', None, timeout=10) or {}).get('status') == 'ok':
                    self._schemas = list((self._rpc(env, '/tools', None) or {}).get('tools') or [])
                    return
            except Exception as e:  # noqa
                last = str(e)
            time.sleep(2)
        raise RuntimeError(f'{self._runtime} tool runtime did not come up within {self._boot_timeout}s '
                           f'(last error: {last})\n{self.server_log(env)}')

    def _rpc(self, env: Env, path: str, payload: Optional[Dict[str, Any]],
             timeout: Optional[int] = None) -> Dict[str, Any]:
        """One request to the in-sandbox server, via curl on the command channel.

        The body is written to a file rather than inlined: tool arguments carry
        arbitrary source code, and no amount of shell quoting survives that
        reliably.

        The file name carries a nonce because two threads can be in here at once.
        A fixed ``request.json`` made them overwrite each other between the write
        and the curl, so every concurrent call executed whichever payload landed
        last and each caller filed that one answer under its own call -- one
        episode came back with a glob listing as the result of a python script it
        never ran.

        curl is given ``_RPC_HEADROOM`` seconds more than the server is allowed to
        spend, and the command channel more again. They used to share one number,
        which meant that when a call ran long the client gave up in the same
        second the server was formulating its answer -- and the client wins that
        race, so a turn holding one slow call came back as "Tool runtime
        unreachable" for *every* call in it, including the ones that had
        finished. With headroom the server's own per-call timeout message
        arrives instead.
        """
        seconds = timeout or self._call_timeout
        sandbox = self._require_sandbox(env)
        if payload is None:
            command = f'curl -sS -m {seconds + _RPC_HEADROOM} http://127.0.0.1:{self._port}{path}'
        else:
            request = f'{_REMOTE_DIR}/request-{uuid.uuid4().hex}.json'
            sandbox.files.write(request, json.dumps(payload, ensure_ascii=False))
            command = (f'curl -sS -m {seconds + _RPC_HEADROOM} -X POST -H "Content-Type: application/json" '
                       f'--data-binary @{request} http://127.0.0.1:{self._port}{path}; '
                       f'rm -f {request}')
        result = sandbox.commands.run(command, timeout=seconds + 2 * _RPC_HEADROOM)
        stdout = (getattr(result, 'stdout', '') or '').strip()
        if not stdout:
            raise RuntimeError(f'empty response from {path}: {getattr(result, "stderr", "")}')
        return json.loads(stdout)

    @staticmethod
    def _require_sandbox(env: Env):
        """The machine to talk to, or a message naming what this needs.

        A sandbox handle and a workspace, which is the whole dependency: an env
        that runs its tools in this process has nothing for a server to be
        uploaded into, and finding that out as ``AttributeError: 'LocalEnv'
        object has no attribute 'sandbox'`` names the symptom rather than the
        mistake.
        """
        sandbox = getattr(env, 'sandbox', None)
        if sandbox is None:
            raise RuntimeError(f'RemoteTools needs a sandboxed environment that has been reset; '
                               f'{type(env).__name__} offers no sandbox to install into')
        return sandbox

    def _truncate(self, text: str) -> str:
        limit = self.max_observation_chars
        return truncate_observation(text, limit) if limit else text
