"""Test cleanup compatibility without starting vLLM workers."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('use_core', [False, True])
@pytest.mark.parametrize('asynchronous', [False, True])
def test_shutdown_compatibility(use_core, asynchronous):
    root = Path(__file__).resolve().parents[2]
    source = root / 'src/twinkle/sampler/vllm_sampler/vllm_engine.py'
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'VLLMEngine'
    )
    method = next(
        node for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'shutdown'
    )

    calls, warnings = [], []
    env = {
        'logger': SimpleNamespace(
            info=lambda *a, **k: None,
            warning=lambda *a, **k: warnings.append((a, k)),
        ),
        'Torch': SimpleNamespace(
            empty_cache=lambda: None,
            ipc_collect=lambda: None,
        ),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'),
        env,
    )

    def sync_shutdown():
        calls.append('shutdown')

    async def async_shutdown():
        calls.append('shutdown')

    backend = SimpleNamespace(
        shutdown=async_shutdown if asynchronous else sync_shutdown,
    )
    engine = SimpleNamespace(engine_core=backend) if use_core else backend
    wrapper = SimpleNamespace(
        engine=engine,
        _lora_request_cache={'old': 1},
    )

    asyncio.run(env['shutdown'](wrapper))

    assert calls == ['shutdown']
    assert warnings == []
    assert wrapper.engine is None
    assert wrapper._lora_request_cache == {}
