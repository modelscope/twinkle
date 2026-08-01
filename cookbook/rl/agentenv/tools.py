"""Tool definitions for AgentENV-backed code-writing RL (MBPP).

Mirrors ``cookbook/rl/openenv_code/tools.py`` — identical task and tool names —
but the tools execute inside a real Firecracker microVM instead of a remote
OpenEnv interpreter session. The differences that matter to the prompt:

  * Full CPython, so ``assert`` / ``try`` / imports / files all work, and the
    hidden tests can be replayed as one ordinary script.
  * Each ``run_python`` call is a FRESH process, so snippets must be
    self-contained (an OpenEnv session, by contrast, keeps its namespace).

Two tools are exposed to the model:
  * ``run_python``      — write a snippet to /workspace/scratch.py and run it.
  * ``submit_solution`` — hand in the final function (recorded client-side).
"""
import textwrap
from typing import Any, Dict, List, Tuple

from twinkle_agentic.envs import AgentEnv

SYSTEM_PROMPT = """You are an expert Python programmer with access to a Linux sandbox.

Write a function that solves the given task, verify it, then submit it.

Rules:
- Use `run_python` to try out your function. Each call runs in a FRESH process,
  so every snippet must be self-contained (include the imports and the function
  definition, then call it) and must `print(...)` what you want to see.
- The full Python standard library is available, plus `numpy` and `sympy`.
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
            'description': 'Run a self-contained Python snippet in the sandbox and return its stdout/stderr.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'Python source to execute. Must print what you want to inspect.',
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


def _run_python(env: AgentEnv, arguments: Dict[str, Any]) -> str:
    """Write the snippet to a file and execute it, avoiding shell quoting issues."""
    code = arguments.get('code')
    if not code:
        return "Error: 'code' argument is required."
    env.sandbox.files.write('/workspace/scratch.py', code)
    return env.run_command({'command': 'python /workspace/scratch.py', 'cwd': '/workspace'})


def _submit_solution(env: AgentEnv, arguments: Dict[str, Any]) -> str:
    """Record the solution on the env; the training loop scores it later."""
    code = (arguments.get('code') or '').strip()
    if not code:
        return "Error: 'code' argument is required."
    env.submitted_code = code
    return 'Solution submitted.'


def register_tools(env: AgentEnv) -> AgentEnv:
    """Attach the task tools to a fresh AgentEnv instance."""
    env.submitted_code = None
    return (env.register_tool(TOOL_SCHEMA[0], _run_python).register_tool(TOOL_SCHEMA[1], _submit_solution))


def _build_test_script(solution: str, test_list: List[str], setup_code: str) -> str:
    """Build a script that runs each assertion independently and prints a tally.

    Every test is wrapped in its own ``try`` so that one failing assertion (or
    one that raises) does not hide the rest — the reward uses the pass rate.
    """
    parts = [solution, '']
    if setup_code:
        parts += [setup_code, '']
    parts.append('_passed = 0')
    for test in test_list:
        body = textwrap.indent(test.strip(), '    ')
        parts += ['try:', body, '    _passed += 1', 'except Exception:', '    pass']
    parts.append(f"print('TESTS_PASSED', _passed, {len(test_list)})")
    return '\n'.join(parts) + '\n'


def run_tests(env: AgentEnv, test_list: List[str], setup_code: str = '') -> Tuple[int, int]:
    """Replay the hidden tests against the submitted solution inside the sandbox.

    Returns:
        ``(n_passed, n_total)``. ``(0, n)`` when nothing was submitted, or when
        the script never reaches its tally line (syntax error, timeout, ...).
    """
    total = len(test_list)
    solution = getattr(env, 'submitted_code', None)
    if not solution:
        return 0, total

    script = _build_test_script(solution, test_list, setup_code)
    env.sandbox.files.write('/workspace/run_tests.py', script)
    output = env.run_command({'command': 'python /workspace/run_tests.py', 'cwd': '/workspace'})

    for line in reversed(output.splitlines()):
        if line.startswith('TESTS_PASSED'):
            fields = line.split()
            if len(fields) >= 3 and fields[1].isdigit():
                return int(fields[1]), total
    return 0, total
