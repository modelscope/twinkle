import os
import textwrap
from typing import Any, Dict, List, Tuple

from twinkle_agentic.envs import AgentEnv

NAME = 'agentenv'

API_URL = os.environ.get('AENV_API_URL', 'http://127.0.0.1:8000')
TEMPLATE = os.environ.get('AENV_TEMPLATE', 'twinkle-code')
SANDBOX_TIMEOUT = int(os.environ.get('SANDBOX_TIMEOUT', '600'))
COMMAND_TIMEOUT = int(os.environ.get('AENV_COMMAND_TIMEOUT', '60'))

SYSTEM_PROMPT = """You are an expert Python programmer with access to a Linux sandbox.

Solve the task by writing a Python function.

- Use `run_python` to try out your function. Each call runs in a FRESH process,
  so every snippet must be self-contained (include the imports and the function
  definition, then call it) and must `print(...)` what you want to see.
- The full Python standard library is available, plus `numpy` and `sympy`.
- When you are confident, call `submit_solution` with the complete final source
  (imports plus the function definition).

Submit exactly once, and only after the code runs correctly."""

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


def make_env() -> AgentEnv:
    """Boot one sandbox per trajectory, exposing only the two task tools.

    ``include_default_tools=False`` hides AgentEnv's built-ins (raw command
    execution, file read/write) so the action space matches the task exactly and
    reward attribution stays clean.
    """
    env = AgentEnv(
        template=TEMPLATE,
        api_url=API_URL,
        sandbox_timeout=SANDBOX_TIMEOUT,
        command_timeout=COMMAND_TIMEOUT,
        include_default_tools=False,
    )
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

    Real CPython means the tests can run as one ordinary script, unlike the
    OpenEnv backend which drives them one expression at a time.

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


def describe() -> str:
    return f'AgentENV microVM: api_url={API_URL}, template={TEMPLATE}'
