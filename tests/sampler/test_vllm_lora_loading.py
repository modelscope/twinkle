import asyncio
from unittest.mock import MagicMock

from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine


def test_concurrent_lora_requests_share_one_load_task():
    async def run():
        engine = VLLMEngine.__new__(VLLMEngine)
        engine._lora_request_cache = {}
        engine._lora_load_tasks = {}
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
        assert engine._lora_load_tasks == {}

    asyncio.run(run())


def test_unload_lora_accepts_synchronous_engine_api():
    async def run():
        engine = VLLMEngine.__new__(VLLMEngine)
        request = MagicMock(lora_int_id=7)
        engine._lora_request_cache = {'/adapter': request}
        engine._lora_load_tasks = {}
        engine.engine = MagicMock()
        engine.engine.remove_lora.return_value = True

        await engine.unload_lora_paths(['/adapter'])

        engine.engine.remove_lora.assert_called_once_with(7)
        assert engine._lora_request_cache == {}

    asyncio.run(run())


def test_unload_lora_removes_a_just_completed_load():
    async def run():
        engine = VLLMEngine.__new__(VLLMEngine)
        request = MagicMock(lora_int_id=9)
        load_task = asyncio.create_task(asyncio.sleep(0, result=request))
        await load_task
        engine._lora_request_cache = {}
        engine._lora_load_tasks = {'/adapter': load_task}
        engine.engine = MagicMock()
        engine.engine.remove_lora.return_value = None

        await engine.unload_lora_paths(['/adapter'])

        engine.engine.remove_lora.assert_called_once_with(9)
        assert engine._lora_load_tasks == {}

    asyncio.run(run())
