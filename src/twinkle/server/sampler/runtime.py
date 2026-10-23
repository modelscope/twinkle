# Copyright (c) ModelScope Contributors. All rights reserved.
"""Request ownership inside one allocated sampling replica, independent of HTTP waiters."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from twinkle.server.deployment import build_deployment_app
from twinkle.server.exceptions import ResourceNotFoundError
from twinkle.server.lifecycle.proxy import ForwardedReply, ForwardedRequest
from twinkle.server.task_queue.types import ExecutionScope


def request_execution_scope(request) -> ExecutionScope | None:
    """Internal ASGI metadata, absent for direct standalone endpoint calls."""
    return request.scope.get('state', {}).get('execution_scope')


@dataclass
class SamplingOperation:
    scope: ExecutionScope
    task: asyncio.Task | None = None


class SamplerRuntime:

    def __init__(self, owner):
        self.owner = owner
        self._operations: dict[str, SamplingOperation] = {}
        self._stream_tasks = set()
        self._rpc_app = None

    def start_stream_job(self, factory, *, execution_scope: ExecutionScope | None = None):
        job = asyncio.create_task(factory())
        if execution_scope is not None:
            execution_scope.track(job)
        self._stream_tasks.add(job)

        def done(task):
            self._stream_tasks.discard(task)
            if not task.cancelled():
                task.exception()

        job.add_done_callback(done)
        return job

    async def shutdown(self):
        await asyncio.gather(*tuple(self._stream_tasks), return_exceptions=True)

    async def invoke(self, ticket: str, payload: ForwardedRequest, stream_queue=None) -> ForwardedReply:
        """Return the HTTP reply early while retaining every actual generation job."""
        if ticket in self._operations:
            raise ResourceNotFoundError('Sampling request was already dispatched')
        if self._rpc_app is None:
            from .tinker_handlers import _register_tinker_sampler_routes
            from .twinkle_handlers import _register_twinkle_sampler_routes

            def routes(app, get_self):
                _register_tinker_sampler_routes(app, lambda: self.owner)
                _register_twinkle_sampler_routes(app, lambda: self.owner)

            self._rpc_app = build_deployment_app('Sampler', routes)
        loop = asyncio.get_running_loop()
        reply = loop.create_future()
        record = SamplingOperation(ExecutionScope())
        self._operations[ticket] = record

        async def dispatch():
            response: ForwardedReply = {'status': 500, 'headers': [], 'body': b''}
            sent = False

            async def receive():
                nonlocal sent
                if not sent:
                    sent = True
                    return {'type': 'http.request', 'body': payload['body'], 'more_body': False}
                await asyncio.Event().wait()

            async def send(message):
                if message['type'] == 'http.response.start':
                    response['status'] = message['status']
                    response['headers'] = message['headers']
                    if stream_queue is not None and not reply.done():
                        reply.set_result(dict(response))
                elif message['type'] == 'http.response.body':
                    chunk = message.get('body', b'')
                    if stream_queue is None:
                        response['body'] += chunk
                    elif chunk:
                        try:
                            await stream_queue.put_async(chunk)
                        except Exception:
                            pass  # Keep draining generation after an HTTP consumer disconnects.

            scope = {
                'type': 'http',
                'state': {
                    'execution_scope': record.scope
                },
                'asgi': {
                    'version': '3.0'
                },
                'http_version': '1.1',
                'scheme': 'http',
                'method': payload['method'],
                'path': '/' + payload['endpoint'],
                'raw_path': ('/' + payload['endpoint']).encode(),
                'root_path': '',
                'query_string': payload.get('query_string', b''),
                'headers': payload['headers'],
                'server': ('sampler', 80),
                'client': ('controller', 0)
            }
            try:
                await self._rpc_app(scope, receive, send)
                if not reply.done():
                    reply.set_result(response)
            except BaseException as error:
                if not reply.done():
                    reply.set_exception(error)
                raise
            finally:
                # ComputeWorker supplied completion handles, not a terminal status read.
                await record.scope.wait()
                if stream_queue is not None:
                    try:
                        await stream_queue.put_async(None)
                    except Exception:
                        pass

        record.task = asyncio.create_task(dispatch())
        record.task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        return await asyncio.shield(reply)

    async def wait_operation(self, ticket):
        record = self._operations.get(ticket)
        if record is None:
            raise ResourceNotFoundError('Unknown sampling operation')
        await asyncio.shield(record.task)
        self.owner.sampling_guard.check_available()
        self._operations.pop(ticket, None)

    async def install_base_weights(self, path):
        from .weights import resolve_sampler_weights
        await self.owner.sampling_guard.run(
            lambda: resolve_sampler_weights(self.owner, path, full_checkpoint=True), exclusive=True)

    async def pending_futures(self):
        return list({identity for record in self._operations.values() for identity in record.scope.request_ids})

    async def runtime_identity(self):
        import ray
        actors = getattr(self.owner.sampler, '_actors', ())
        states = await asyncio.to_thread(
            lambda: [ray._private.state.actors(actor._actor_id.hex()).get('State') for actor in actors])
        return {
            'unhealthy':
            self.owner.sampling_guard._closed or self.owner.sampling_guard._weights_error is not None
            or any(state != 'ALIVE' for state in states)
        }
