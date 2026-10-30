"""SDK operations shared by the mock and real-device acceptance harnesses."""
import time


def unload_tinker_model(service, model_id):
    from tinker import types
    from tinker.lib.api_future_impl import _APIFuture
    from tinker.lib.client_connection_pool_type import ClientConnectionPoolType

    async def unload():
        started = time.time()
        with service.holder.aclient(ClientConnectionPoolType.TRAIN) as client:
            future = await client.models.unload(request=types.UnloadModelRequest(model_id=model_id))
        return await _APIFuture(
            types.UnloadModelResponse, service.holder, future, request_start_time=started,
            request_type='UnloadModel').result_async()

    return service.holder.run_coroutine_threadsafe(unload()).result(timeout=180)
