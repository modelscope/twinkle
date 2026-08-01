"""Execution backends for the code-writing RL task.

``train.py`` is backend-agnostic: it owns the dataset, the GRPO loop and the
reward shaping, and delegates everything that depends on *where code runs* to
one of the modules here.

A backend module must expose:

``NAME``
    Short identifier used in logs and checkpoint names.
``SYSTEM_PROMPT``
    Task prompt. This is NOT shared, because the execution semantics differ in
    ways the model must know about — most importantly whether state survives
    between ``run_python`` calls, and which modules are importable. A prompt
    that describes the wrong backend sends the policy down the wrong path.
``TOOL_SCHEMA``
    OpenAI-format tool list advertised to the model. Both backends use the same
    two tool *names* (``run_python``, ``submit_solution``) so trajectories stay
    comparable; only the descriptions differ.
``make_env() -> Env``
    Build one environment for one trajectory, with tools registered. Must set
    ``env.submitted_code = None``.
``run_tests(env, test_list, setup_code) -> tuple[int, int]``
    Replay the hidden tests against ``env.submitted_code`` and return
    ``(n_passed, n_total)``. Returns ``(0, n)`` when nothing was submitted.
``describe() -> str``
    One line of connection info for the startup log.

Env lifecycle (``reset``/``close``) is part of the ``Env`` interface, so
``train.py`` calls those directly.
"""
from importlib import import_module
from types import ModuleType

AVAILABLE = ('openenv', 'agentenv')


def get_backend(name: str) -> ModuleType:
    """Import a backend module by short name.

    Args:
        name: One of ``AVAILABLE``.

    Raises:
        ValueError: On an unknown name, listing the valid ones — a typo in
            ``CODE_RL_BACKEND`` should fail immediately rather than silently
            falling back to a default and training the wrong thing.
    """
    if name not in AVAILABLE:
        raise ValueError(f'Unknown backend {name!r}. Available: {", ".join(AVAILABLE)}')
    return import_module(f'backends.{name}')
