"""OpenEnv server-mode backend: one WebSocket session per trajectory.

Code runs in a remote OpenEnv ``coding_env`` interpreter session. Two properties
shape the prompt below:

  * The session namespace PERSISTS across ``run_python`` calls, so the model can
    define a function in one turn and probe it in the next.
  * ``coding_env`` executes through an AST interpreter, not real CPython. There
    is no file or network access, decorators are silently ignored, and only
    authorised modules can be imported (see ``server_app.py``).
"""
import os
from typing import Any, Dict, List, Tuple

from twinkle_agentic.envs import OpenEnvClient

NAME = 'openenv'

BASE_URL = os.environ.get('OPENENV_BASE_URL', 'http://127.0.0.1:8000')
ENV_NAME = os.environ.get('OPENENV_ENV_NAME', 'coding_env')
# Code execution can be slow; keep the per-message timeout generous. Note the
# executor's own caps still apply (operation count, while-loop iterations, and
# a wall-clock limit in newer smolagents releases).
MESSAGE_TIMEOUT_S = float(os.environ.get('OPENENV_MESSAGE_TIMEOUT_S', '120'))

SYSTEM_PROMPT = """You are an expert Python programmer with access to a Python interpreter.

Solve the task by writing a Python function.

- Use `run_python` to define and test your function. The interpreter keeps its
  state between calls, so a function defined in one call stays available.
- Available modules: math, re, collections, itertools, functools, operator,
  heapq, bisect, string, statistics, fractions, decimal, datetime, copy, json.
  There is no file or network access.
- When you are confident, call `submit_solution` with the complete final source
  (imports plus the function definition).

Submit exactly once, and only after the code runs correctly."""

TOOL_SCHEMA: List[Dict[str, Any]] = [
    {
        'type': 'function',
        'function': {
            'name': 'run_python',
            'description': 'Execute a Python snippet in the interpreter and return its stdout/stderr.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'Python source to execute. Use print(...) to inspect values.',
                    },
                },
                'required': ['code'],
            },
        },
    },
    {
        'type': 'function',
        'function': {
            'name': 'submit_solution',
            'description': 'Submit the final solution source and end the coding phase.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'Complete final source: imports plus the function definition.',
                    },
                },
                'required': ['code'],
            },
        },
    },
]


def _submit_solution(env: OpenEnvClient, arguments: Dict[str, Any]) -> str:
    """Record the solution on the env; the training loop scores it later."""
    code = (arguments.get('code') or '').strip()
    if not code:
        return "Error: 'code' argument is required."
    env.submitted_code = code
    return 'Solution submitted.'


def make_env() -> OpenEnvClient:
    """Open one session per trajectory, exposing only the two task tools.

    ``run_python`` stays server-backed: the default action mapper turns
    ``{'code': ...}`` straight into the env's ``CodeAction``. Only
    ``submit_solution`` needs a client-side handler.
    """
    env = OpenEnvClient(
        env_name=ENV_NAME,
        base_url=BASE_URL,
        tools=[TOOL_SCHEMA[0]],
        message_timeout_s=MESSAGE_TIMEOUT_S,
    )
    env.submitted_code = None
    return env.register_tool(TOOL_SCHEMA[1], _submit_solution)


def _ok(result) -> bool:
    return not (getattr(result.observation, 'exit_code', 0) or 0)


def _last_line(text: str) -> str:
    lines = [line for line in (text or '').splitlines() if line.strip()]
    return lines[-1].strip() if lines else ''


def run_tests(env: OpenEnvClient, test_list: List[str], setup_code: str = '') -> Tuple[int, int]:
    """Run the hidden tests against the submitted solution in the same session.

    Each assertion is executed as its own ``print(<expr>)`` step rather than as
    one block of ``assert`` statements. Two reasons: printing the boolean tells
    "the assertion evaluated to False" apart from "the code blew up", whereas a
    failed ``assert`` surfaces as an exception indistinguishable from a crash
    inside the solution; and one step per test isolates a test that raises, so
    the remaining ones still run.

    The executor does support ``assert``/``try``, so this is a diagnosability
    choice, not a capability workaround.

    Returns:
        ``(n_passed, n_total)``. ``(0, n)`` when nothing was submitted or the
        solution itself fails to execute.
    """
    total = len(test_list)
    solution = getattr(env, 'submitted_code', None)
    if not solution:
        return 0, total

    # Define the submitted solution in the session namespace.
    if not _ok(env.execute({'code': solution})):
        return 0, total
    if setup_code and not _ok(env.execute({'code': setup_code})):
        return 0, total

    passed = 0
    for test in test_list:
        expr = test.strip()
        if expr.startswith('assert '):
            expr = expr[len('assert '):]
        result = env.execute({'code': f'print({expr})'})
        if _ok(result) and _last_line(getattr(result.observation, 'stdout', '')) == 'True':
            passed += 1
    return passed, total


def describe() -> str:
    return f'OpenEnv server mode: base_url={BASE_URL}, env={ENV_NAME}'
