"""ms-agent's tool runtime, served over HTTP from inside the sandbox.

This is the half of the RSI setup that runs *in* the microVM. It builds a real
``LLMAgent`` from the same ``rsi_agent.yaml`` the training host reads, lets
ms-agent prepare its own tools, and exposes two things over HTTP:

* ``GET /tools`` -- the tool schemas, taken from the runtime that will execute
  them. The training host advertises these to the model verbatim, so the
  contract in the prompt and the code behind it cannot drift apart.
* ``POST /call`` -- dispatch, through ms-agent's own ``single_call_tool`` /
  ``parallel_call_tool``.

Nothing here reimplements a tool. That is the whole point: the policy is
trained against the same ``edit_file`` / ``grep`` / ``shell_executor`` behaviour
it will meet at serving time, down to the output formatting. A reimplementation
would be cheaper, but in RL any divergence gets actively exploited by the policy
and only shows up after deployment.

The server is deliberately stdlib-only so the sandbox image stays close to
ms-agent's own dependency set.

Run inside the sandbox::

    python tool_server.py --config /opt/rsi/rsi_agent.yaml --workspace /workspace
"""
import argparse
import asyncio
import copy
import json
import os
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

DEFAULT_PORT = 8900

# read_file's only LLM-backed argument: it summarises a file instead of
# returning it verbatim. Without a reachable LLM the tool cannot honour it, so
# it is also removed from the advertised schema -- see `_usable_llm`.
_LLM_BACKED_ARGS = {'file_system---read_file': ('abbreviate', )}


def _usable_llm(cfg) -> bool:
    """Whether the declared ``llm`` section can actually serve a request.

    ms-agent merges its own ``agent.yaml`` underneath the user's, and that
    default declares ``service: modelscope``. So an absent ``llm:`` section in
    rsi_agent.yaml does not mean "no LLM" -- it means "modelscope, with no
    credentials", which asserts as soon as FileSystemTool is constructed. The
    presence of a key is what decides it.
    """
    llm = getattr(cfg, 'llm', None)
    if llm is None:
        return False
    service = str(getattr(llm, 'service', '') or '')
    key_fields = (f'{service}_api_key', 'api_key', 'openai_api_key')
    return any(getattr(llm, f, None) or os.environ.get(f.upper()) for f in key_fields)


def _without_llm_args(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Drop arguments this deployment cannot serve from a tool schema.

    Everything reachable from ``/tools`` has to be executable, or the model
    spends the episode learning that a documented argument is broken and carries
    that lesson to a deployment where it works.
    """
    fn = schema.get('function') or {}
    drop = _LLM_BACKED_ARGS.get(fn.get('name'))
    properties = ((fn.get('parameters') or {}).get('properties') or {})
    if not drop or not any(arg in properties for arg in drop):
        return schema
    schema = copy.deepcopy(schema)
    for arg in drop:
        schema['function']['parameters']['properties'].pop(arg, None)
    return schema


class _LoopThread:
    """A single long-lived asyncio loop, owned by a background thread.

    ms-agent's tools bind state to the loop that created them: the notebook
    kernel, MCP client sessions and subprocess transports all hold references
    to it. Running ``asyncio.run`` per request would strand that state -- the
    notebook would lose its variables between turns -- so one loop is created
    at startup and every request is submitted onto it.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._serve, name='ms-agent-loop', daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coro, timeout: Optional[float] = None):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)


class ToolRuntime:
    """Owns the ms-agent agent and answers tool queries against it."""

    def __init__(self, config_path: str, workspace: str) -> None:
        from omegaconf import OmegaConf, open_dict

        from ms_agent.agent.llm_agent import LLMAgent

        cfg = OmegaConf.load(config_path)
        with open_dict(cfg):
            cfg.output_dir = workspace
            # Same non-interactive stubs MsAgentHarness applies on the training
            # host: nothing here can answer a TUI permission prompt, and a tool
            # blocking on stdin would hang the episode until the sandbox timeout.
            cfg.interactive = False
            cfg.permission_mode = 'auto'
            self.has_llm = _usable_llm(cfg)
            if not self.has_llm:
                # Leaving an unusable section in place is not an option:
                # FileSystemTool builds a client from it eagerly and asserts on
                # the missing key, so no tool at all would come up.
                cfg.pop('llm', None)
        self.agent = LLMAgent(cfg)
        self.agent._interactive = False
        self.agent._event_sink = None
        self.agent._input_source = None
        self.workspace = workspace
        self._loop = _LoopThread()
        self._loop.run(self._prepare())

    async def _prepare(self) -> None:
        self.agent.prepare_runtime()
        await self.agent.prepare_tools()

    @property
    def _tm(self):
        return self.agent.tool_manager

    def tools(self) -> List[Dict[str, Any]]:
        """Tool schemas, flattened to a plain OpenAI-shaped list.

        ``get_tools`` groups by server; the model only ever sees the flat list,
        and the names are already namespaced as ``{server}---{tool}``.
        """
        raw = self._loop.run(self._tm.get_tools())
        if isinstance(raw, dict):
            flat: List[Any] = []
            for value in raw.values():
                flat.extend(value if isinstance(value, list) else [value])
        else:
            flat = list(raw or [])
        return [t if self.has_llm else _without_llm_args(t) for t in flat if isinstance(t, dict)]

    def call(self, calls: List[Dict[str, Any]], timeout: Optional[float]) -> List[Dict[str, Any]]:
        """Dispatch a turn's tool calls, mirroring how ms-agent itself does it.

        A single call goes through ``single_call_tool`` and a batch through
        ``parallel_call_tool``, matching LLMAgent, so concurrency-sensitive
        tools behave in training exactly as they do in production.
        """
        payload = [{'tool_name': c.get('tool_name'), 'arguments': c.get('arguments') or {}} for c in calls]
        try:
            if len(payload) == 1:
                results = [self._loop.run(self._tm.single_call_tool(payload[0]), timeout)]
            else:
                results = self._loop.run(self._tm.parallel_call_tool(payload), timeout)
        except Exception as e:  # noqa
            # One failing tool must not take down the server: the episode can
            # still recover, and a dead server would fail every later step of
            # every trajectory sharing this sandbox.
            detail = f'{type(e).__name__}: {e}'
            return [{'observation': f'Tool call failed. {detail}', 'ok': False} for _ in payload]
        return [{'observation': _as_text(r), 'ok': True} for r in list(results)]


def _as_text(result: Any) -> str:
    if result is None:
        return ''
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(result)


class _Handler(BaseHTTPRequestHandler):
    runtime: ToolRuntime = None  # set on the class before the server starts
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path.startswith('/health'):
            self._reply(200, {'status': 'ok', 'workspace': self.runtime.workspace})
        elif self.path.startswith('/tools'):
            self._guarded(lambda: {'tools': self.runtime.tools()})
        else:
            self._reply(404, {'error': f'no such endpoint: {self.path}'})

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.startswith('/call'):
            self._reply(404, {'error': f'no such endpoint: {self.path}'})
            return
        length = int(self.headers.get('Content-Length') or 0)
        try:
            body = json.loads(self.rfile.read(length) or b'{}')
        except ValueError as e:
            self._reply(400, {'error': f'malformed request body: {e}'})
            return
        calls = body.get('calls') or []
        if not isinstance(calls, list) or not calls:
            self._reply(400, {'error': "'calls' must be a non-empty list"})
            return
        self._guarded(lambda: {'results': self.runtime.call(calls, body.get('timeout'))})

    def _guarded(self, produce) -> None:
        """Answer with ``produce()``, turning a crash into a 500 with a traceback.

        The client surfaces the body as the observation, so a bug in here shows
        up in the trajectory instead of as an opaque connection reset.
        """
        try:
            self._reply(200, produce())
        except Exception:  # noqa
            self._reply(500, {'error': traceback.format_exc()})

    def _reply(self, code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write('[tool_server] %s\n' % (fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='ms-agent yaml, the same one the training host loads')
    parser.add_argument('--workspace', default='/workspace', help='config.output_dir for this episode')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    runtime = ToolRuntime(args.config, args.workspace)
    _Handler.runtime = runtime
    # Threading, because a turn's tool calls arrive as one request but the
    # health poll must stay answerable while a long shell command runs.
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    names = [t.get('function', {}).get('name') for t in runtime.tools()]
    llm_note = 'llm configured' if runtime.has_llm else 'no llm (read_file.abbreviate withdrawn)'
    sys.stderr.write(f'[tool_server] ready on {args.host}:{args.port}, {llm_note}, '
                     f'{len(names)} tools: {names}\n')
    sys.stderr.flush()
    server.serve_forever()


if __name__ == '__main__':
    main()
