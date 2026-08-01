"""Tool definitions for OpenEnv-backed code-writing RL (MBPP).

Same role as ``cookbook/rl/agentenv/tools.py``, but the tools run against a
remote OpenEnv **server session** instead of a Firecracker sandbox. The pair of
examples is intentionally symmetric: identical task, identical tool names,
different execution backend.

Two tools are exposed to the model:
  * ``run_python``      — execute a snippet in the session (server-backed).
  * ``submit_solution`` — hand in the final function (recorded client-side).

The session namespace persists across ``run_python`` calls, so the model can
define a function in one turn and probe it in the next.
"""
from typing import Any, Dict, List, Tuple

from twinkle_agentic.envs import OpenEnvClient

SYSTEM_PROMPT = """You are an expert Python programmer with access to a Python interpreter.

Write a function that solves the given task, verify it, then submit it.

Rules:
- Use `run_python` to define and test your function. The interpreter keeps its
  state between calls, so a function defined in one call stays available.
- Available modules: math, re, collections, itertools, functools, operator,
  heapq, bisect, string, statistics, fractions, decimal, datetime, copy, json.
  There is no file or network access.
- Match the function name and signature implied by the task description and
  the example call, otherwise the hidden tests cannot find your function.
- When your function works, call `submit_solution` with the COMPLETE final
  source (imports plus the function definition).
- After submitting, reply with one short sentence and do NOT call any more tools.
- You have a limited number of turns, so do not run redundant code."""

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


def register_tools(env: OpenEnvClient) -> OpenEnvClient:
    """Attach the task tools to a fresh OpenEnvClient instance.

    ``run_python`` stays server-backed: the default action mapper turns
    ``{'code': ...}`` straight into the env's ``CodeAction``.
    """
    env.submitted_code = None
    return env.register_tool(TOOL_SCHEMA[1], _submit_solution)


def _last_line(text: str) -> str:
    lines = [line for line in (text or '').splitlines() if line.strip()]
    return lines[-1].strip() if lines else ''


def _ok(result) -> bool:
    return not (getattr(result.observation, 'exit_code', 0) or 0)


def run_tests(env: OpenEnvClient, test_list: List[str], setup_code: str = '') -> Tuple[int, int]:
    """Run the hidden tests against the submitted solution in the same session.

    Each assertion is executed as its own ``print(<expr>)`` step rather than as
    one block of ``assert`` statements. Two reasons: printing the boolean tells
    "the assertion evaluated to False" apart from "the code blew up", whereas a
    failed ``assert`` surfaces as an exception indistinguishable from a crash
    inside the solution; and one step per test isolates a test that raises, so
    the remaining ones still run.

    The executor does support ``assert``/``try`` (smolagents implements both),
    so this is a diagnosability choice, not a capability workaround.

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
