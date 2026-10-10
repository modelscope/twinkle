import asyncio
from unittest.mock import MagicMock
from weakref import WeakValueDictionary

from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine


def _engine():
    engine = VLLMEngine.__new__(VLLMEngine)
    engine._lora_request_cache = {}
    engine._lora_revisions = {}
    engine._lora_path_locks = WeakValueDictionary()
    engine.engine = MagicMock()
    return engine


def test_concurrent_lora_requests_load_once():
    async def run():
        engine = _engine()
        request = object()
        load_count = 0

        async def load_lora(_path):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(.01)
            return request

        engine._load_lora = load_lora
        results = await asyncio.gather(*(engine._get_or_load_lora('/adapter') for _ in range(8)))

        assert load_count == 1
        assert results == [request] * 8
        assert engine._lora_request_cache == {'/adapter': request}
        assert not engine._lora_path_locks

    asyncio.run(run())


def test_unload_lora_accepts_synchronous_engine_api():
    async def run():
        engine = _engine()
        request = MagicMock(lora_int_id=7)
        engine._lora_request_cache = {'/adapter': request}
        engine.engine.remove_lora.return_value = True

        await engine.unload_lora_paths(['/adapter'])

        engine.engine.remove_lora.assert_called_once_with(7)
        assert engine._lora_request_cache == {}

    asyncio.run(run())


def test_unload_lora_waits_for_an_inflight_load():
    async def run():
        engine = _engine()
        request = MagicMock(lora_int_id=9)
        started, finish = asyncio.Event(), asyncio.Event()
        async def load(_):
            started.set()
            await finish.wait()
            return request
        engine._load_lora = load
        load_task = asyncio.create_task(engine._get_or_load_lora('/adapter'))
        await started.wait()
        unload = asyncio.create_task(engine.unload_lora_paths(['/adapter']))
        await asyncio.sleep(0)
        assert not unload.done()
        finish.set()
        assert await load_task is request
        await unload

        engine.engine.remove_lora.assert_called_once_with(9)
        assert engine._lora_request_cache == {}
        assert engine._lora_revisions == {}
        assert not engine._lora_path_locks

    asyncio.run(run())


def test_named_checkpoint_overwrite_reloads_cached_lora(tmp_path):
    import json
    from types import SimpleNamespace

    async def run():
        engine = _engine()
        metadata = tmp_path / 'checkpoint_metadata.json'
        loaded = []

        async def load(path):
            await asyncio.sleep(.01)
            request = SimpleNamespace(lora_int_id=len(loaded) + 1, weights=(tmp_path / 'weights').read_text())
            loaded.append(request)
            return request

        engine._load_lora = load
        metadata.write_text(json.dumps({'weights_revision': 'first'}))
        (tmp_path / 'weights').write_text('first')
        first = await engine._get_or_load_lora(str(tmp_path))
        assert await engine._get_or_load_lora(str(tmp_path)) is first
        metadata.write_text(json.dumps({'weights_revision': 'second'}))
        (tmp_path / 'weights').write_text('second')
        results = await asyncio.gather(*(engine._get_or_load_lora(str(tmp_path)) for _ in range(8)))
        assert len(loaded) == 2
        assert all(result.weights == 'second' for result in results)
        engine.engine.remove_lora.assert_called_once_with(first.lora_int_id)
        assert not engine._lora_path_locks

    asyncio.run(run())
