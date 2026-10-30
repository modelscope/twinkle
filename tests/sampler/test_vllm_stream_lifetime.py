# Copyright (c) ModelScope Contributors. All rights reserved.
"""The vLLM stream iterator owns its engine task until actual completion."""
import ast
import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace


def test_stream_iterator_close_joins_the_real_engine_loop_task():
    # Execute the repository method without constructing a GPU-backed vLLM engine.
    path = Path('src/twinkle/sampler/vllm_sampler/vllm_sampler.py')
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '_iter_in_loop')
    namespace = {'asyncio': asyncio}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    loop = asyncio.new_event_loop()
    running, finalizing, release, closed = [threading.Event() for _ in range(4)]

    def run_loop():
        asyncio.set_event_loop(loop)
        running.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop)
    thread.start()
    running.wait(1)

    async def generation():
        try:
            yield 'first'
        finally:
            finalizing.set()
            await asyncio.to_thread(release.wait)

    iterator = namespace['_iter_in_loop'](SimpleNamespace(_async_loop=loop), generation())
    closer = None
    try:
        assert next(iterator) == 'first'
        assert finalizing.wait(1)

        def close():
            iterator.close()
            closed.set()

        closer = threading.Thread(target=close)
        closer.start()
        assert not closed.wait(0.05)
        release.set()
        assert closed.wait(2)
    finally:
        release.set()
        if closer:
            closer.join(2)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(2)
        loop.close()
