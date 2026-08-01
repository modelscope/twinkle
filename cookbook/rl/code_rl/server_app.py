"""OpenEnv server for the code-writing RL task — server mode, no Docker needed.

Wraps OpenEnv's ``PythonCodeActEnv`` (``OpenEnv/envs/coding_env``) with the
three changes a training workload needs. Each one is a deliberate deviation
from upstream defaults:

1. ``SUPPORTS_CONCURRENT_SESSIONS = True``. Upstream leaves this at the
   conservative default, which caps the server at ONE session
   (``create_app`` raises ``ConcurrencyConfigurationError`` for
   ``max_concurrent_envs > 1`` otherwise). The class is in fact session-safe:
   ``__init__`` builds a private executor and state and shares nothing, and
   ``create_app`` receives the class as a *factory*, so every WebSocket
   connection gets a fresh instance.

2. A wider import whitelist. smolagents' ``LocalPythonExecutor`` authorises
   only ``json`` by default, so ``import math`` / ``collections`` — which many
   MBPP solutions need — would fail.

3. No reward transforms. ``coding_env.create_safe_coding_transform()``
   overwrites ``observation.reward`` with code-style heuristics (-1.0 when the
   code matches ``open(`` / ``import os``, +0.1 for short code). For this task
   the reward must come from unit tests, computed by the training script, so a
   style score on the same channel is noise.

Prerequisites::

    pip install openenv
    pip install -e /path/to/OpenEnv/envs/coding_env   # brings in smolagents

Run::

    sh serve.sh
    # or explicitly:
    MAX_CONCURRENT_ENVS=64 uvicorn server_app:app --host 0.0.0.0 --port 8000 --workers 4

Environment variables:
    MAX_CONCURRENT_ENVS: concurrent sessions per worker process (default 64).
"""
import os

from coding_env.models import CodeAction, CodeObservation
from coding_env.server.python_codeact_env import PythonCodeActEnv
from coding_env.server.python_executor import PyExecutor
from openenv.core.env_server import create_app

# Modules the sandboxed executor may import. Keep this list tight: it is the
# only thing standing between model-generated code and this process, since
# LocalPythonExecutor is an AST interpreter, not an OS-level sandbox.
ALLOWED_IMPORTS = [
    'math',
    're',
    'collections',
    'itertools',
    'functools',
    'operator',
    'heapq',
    'bisect',
    'string',
    'statistics',
    'fractions',
    'decimal',
    'datetime',
    'copy',
    'json',
]

MAX_CONCURRENT_ENVS = int(os.environ.get('MAX_CONCURRENT_ENVS', '64'))


class ConcurrentCodeEnv(PythonCodeActEnv):
    """Session-isolated Python executor with a task-appropriate config."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._configure()

    def reset(self, **kwargs):
        # The parent's reset() rebuilds the executor and the transform with
        # upstream defaults, so re-apply our config afterwards. **kwargs
        # absorbs the seed / episode_id the server forwards from the client.
        observation = super().reset()
        self._configure()
        return observation

    def _configure(self) -> None:
        self._executor = PyExecutor(additional_imports=list(ALLOWED_IMPORTS))
        # Drop the style/safety reward heuristics; reward comes from the tests.
        self.transform = None


app = create_app(
    ConcurrentCodeEnv,
    CodeAction,
    CodeObservation,
    env_name='twinkle_code_env',
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)


def main():
    """Entry point for ``python server_app.py`` (single worker)."""
    import uvicorn
    uvicorn.run(
        app,
        host=os.environ.get('HOST', '0.0.0.0'),
        port=int(os.environ.get('PORT', '8000')),
    )


if __name__ == '__main__':
    main()
