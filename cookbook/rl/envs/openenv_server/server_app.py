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
