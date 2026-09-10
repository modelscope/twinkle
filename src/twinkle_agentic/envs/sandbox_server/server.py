# Copyright (c) ModelScope Contributors. All rights reserved.
"""An agent framework's own tools, served over HTTP from inside a sandbox.

This is the half that runs *in* the machine. It is a transport and nothing else:
the tools belong to a runtime module loaded by name, so training a policy behind
one more framework is one more ``runtime_<name>.py`` beside this file and no
change here.

Three endpoints, which is the whole protocol
:class:`~twinkle_agentic.envs.remote_tools.RemoteTools` speaks:

* ``GET  /health`` -- is the runtime up.
* ``GET  /tools``  -- the schemas, taken from the runtime that will execute them,
  so the contract in the prompt and the code behind it cannot drift apart.
* ``POST /call``   -- ``{'calls': [{'tool_name', 'arguments'}], 'timeout': s}``,
  answered with ``{'results': [{'observation'}]}``, one per call and in order.

A runtime is a class named ``Runtime`` in ``runtime_<name>.py``::

    class Runtime:
        def __init__(self, config_path: str, workspace: str) -> None: ...
        def tools(self) -> List[dict]: ...                  # OpenAI-shaped
        def call(self, calls: List[dict], timeout) -> List[dict]: ...
        note: str = ''                                      # optional, logged once

The names in ``tools`` are the names ``call`` is handed back. A framework that
namespaces its tools decides in its own runtime whether the model ever sees the
namespace -- nothing outside that file knows the convention.

Stdlib-only on purpose, so a sandbox image stays close to the framework's own
dependency set.

Run inside the sandbox::

    python server.py --runtime msagent --config /opt/rsi/agent.yaml --workspace /workspace
"""
import argparse
import importlib.util
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict

DEFAULT_PORT = 8900


def load_runtime(name: str, config_path: str, workspace: str):
    """Build the ``Runtime`` from ``runtime_<name>.py`` next to this file.

    Loaded by path rather than imported: these two files are uploaded into a
    sandbox as loose files, so there is no package for an import statement to
    resolve against. Failing here is worth the exit -- a server answering
    ``/health`` with no runtime behind it would let a whole run's episodes fail
    one call at a time instead of once, loudly, at boot.
    """
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'runtime_{name}.py')
    if not os.path.exists(path):
        raise SystemExit(f'[server] no runtime named {name!r}: {path} was not uploaded')
    spec = importlib.util.spec_from_file_location(f'twinkle_runtime_{name}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, 'Runtime'):
        raise SystemExit(f'[server] {path} defines no Runtime class')
    return module.Runtime(config_path, workspace)


class _Handler(BaseHTTPRequestHandler):
    runtime: Any = None  # set on the class before the server starts
    workspace: str = ''
    protocol_version = 'HTTP/1.1'

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        if self.path.startswith('/health'):
            self._reply(200, {'status': 'ok', 'workspace': self.workspace})
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

        The client surfaces the body as the observation, so a bug in a runtime
        shows up in the trajectory instead of as an opaque connection reset.
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
        sys.stderr.write('[server] %s\n' % (fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', default='msagent', help='runtime_<name>.py to serve the tools of')
    parser.add_argument('--config', required=True, help='the framework config, as uploaded by the host')
    parser.add_argument('--workspace', default='/workspace', help='the directory an episode acts in')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    _Handler.runtime = load_runtime(args.runtime, args.config, args.workspace)
    _Handler.workspace = args.workspace
    # Threading, because a turn's tool calls arrive as one request but the health
    # poll must stay answerable while a long shell command runs.
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    names = [(t.get('function') or {}).get('name') for t in _Handler.runtime.tools()]
    note = getattr(_Handler.runtime, 'note', '') or ''
    sys.stderr.write(f'[server] {args.runtime} ready on {args.host}:{args.port}'
                     f'{", " + note if note else ""}, {len(names)} tools: {names}\n')
    sys.stderr.flush()
    server.serve_forever()


if __name__ == '__main__':
    main()
