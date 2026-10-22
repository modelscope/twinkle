# Copyright (c) ModelScope Contributors. All rights reserved.
"""Processor management routes for the Processor deployment.

Registered by ``_register_processor_routes(app, self_fn)`` -- module-level route
registration closing over ``self_fn`` via ``Depends``, not a mixin: there is no
inheritance relationship with the deployment class. All endpoints are prefixed
/twinkle/... and handle processor lifecycle (create, call). ``self_fn`` is injected via
FastAPI Depends to obtain the ProcessorManagement instance at request time.
"""
from __future__ import annotations

import asyncio
import importlib
import uuid
from collections.abc import Callable
from fastapi import Depends, FastAPI, HTTPException, Request
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import ProcessorManagement

import twinkle.protocol.types as types
from twinkle.protocol.serialize import deserialize_object
from twinkle.server.middleware.auth import get_session_id_from_request, get_token_from_request
from twinkle.server.telemetry.correlation import SESSION_ID, TOKEN_ID
from twinkle.server.telemetry.tracing import traced_operation
from twinkle.utils.logger import get_logger

logger = get_logger()

_PROCESSOR_TYPES = ['dataset', 'dataloader', 'preprocessor', 'processor', 'reward', 'template', 'weight_loader']


def _register_processor_routes(app: FastAPI, self_fn: Callable[[], ProcessorManagement]) -> None:
    """Register all /twinkle/* processor routes on the given FastAPI app.

    self_fn is a zero-argument callable that returns the current ProcessorManagement
    replica instance. It is wired in via Depends so it is resolved lazily at request time.
    """

    @app.post('/twinkle/create', response_model=types.ProcessorCreateResponse)
    async def create(
        request: Request, body: types.ProcessorCreateRequest,
        self: ProcessorManagement = Depends(self_fn)) -> types.ProcessorCreateResponse:
        await self._ensure_sticky()

        processor_type_name = body.processor_type
        class_type = body.class_type
        _kwargs = dict(body.init_kwargs)

        assert processor_type_name in _PROCESSOR_TYPES, f'Invalid processor type: {processor_type_name}'
        processor_module = importlib.import_module(f'twinkle.{processor_type_name}')
        assert hasattr(processor_module, class_type), f'Class {class_type} not found in {processor_type_name}'

        token = get_token_from_request(request)
        session_id = get_session_id_from_request(request)
        processor_id = str(uuid.uuid4().hex)

        # Reject malformed registrations before touching shared quota state.
        self._validate_registration(processor_id, token, session_id)
        await self.state.reserve_processor_quota(
            token,
            processor_id,
            session_id,
            limit=self._per_token_processor_limit,
            lease_seconds=self._processor_quota_lease_seconds,
        )

        try:
            self.register_resource(processor_id, token, session_id)

            _kwargs.pop('remote_group', None)
            _kwargs.pop('device_mesh', None)

            resolved_kwargs = {}
            for key, value in _kwargs.items():
                if isinstance(value, str) and value.startswith('pid:'):
                    ref_id = value[4:]
                    resolved_kwargs[key] = self.resource_dict[ref_id]
                else:
                    value = deserialize_object(value)
                    resolved_kwargs[key] = value

            # Run processor instantiation in a thread to avoid blocking the event loop,
            # which would starve the session-liveness coroutines submitted by the
            # countdown thread via asyncio.run_coroutine_threadsafe.
            _remote_group = self.device_group.name
            _device_mesh = self.device_mesh

            def _do_create():
                return getattr(processor_module, class_type)(
                    remote_group=_remote_group, device_mesh=_device_mesh, instance_id=processor_id, **resolved_kwargs)

            # Span the primary processor.create op with token + session correlation.
            with traced_operation(
                    f'processor.create.{processor_type_name}.{class_type}',
                    attrs={
                        TOKEN_ID: token,
                        SESSION_ID: session_id,
                    }):
                processor = await asyncio.get_running_loop().run_in_executor(None, _do_create)
            self.resource_dict[processor_id] = processor
        except Exception:
            self.resource_dict.pop(processor_id, None)
            self.unregister_resource(processor_id)
            try:
                await self.state.release_processor_quota(
                    token,
                    processor_id,
                    lease_seconds=self._processor_quota_lease_seconds,
                )
            except Exception as release_error:
                logger.warning('Failed to release processor quota after create failure for %s: %r', processor_id,
                               release_error)
            raise
        return types.ProcessorCreateResponse(processor_id='pid:' + processor_id)

    @app.post('/twinkle/call', response_model=types.ProcessorCallResponse)
    async def call(
        request: Request, body: types.ProcessorCallRequest,
        self: ProcessorManagement = Depends(self_fn)) -> types.ProcessorCallResponse:
        await self._ensure_sticky()

        processor_id = body.processor_id
        function_name = body.function
        _kwargs = dict(body.call_kwargs)
        processor_id = processor_id[4:]
        self.assert_resource_exists(processor_id)
        processor = self.resource_dict.get(processor_id)
        function = getattr(processor, function_name, None)

        assert function is not None, f'`{function_name}` not found in {processor.__class__}'
        assert hasattr(function, '_execute'), f'Cannot call inner method of {processor.__class__}'

        resolved_kwargs = {}
        for key, value in _kwargs.items():
            if isinstance(value, str) and value.startswith('pid:'):
                ref_id = value[4:]
                resolved_kwargs[key] = self.resource_dict[ref_id]
            else:
                value = deserialize_object(value)
                resolved_kwargs[key] = value

        # Run the processor function in a thread to avoid blocking the event loop.
        # StopIteration cannot propagate through asyncio coroutine boundaries
        # (Python 3.7+ converts it to RuntimeError), so capture it as a sentinel tuple.
        def _do_call():
            try:
                result = function(**resolved_kwargs)
                return False, result
            except StopIteration:
                return True, None

        # Span the primary processor.call op so each invocation is observable.
        with traced_operation(f'processor.call.{function_name}', attrs={TOKEN_ID: get_token_from_request(request)}):
            is_exhausted, result = await asyncio.get_running_loop().run_in_executor(None, _do_call)

        if function_name == '__next__':
            if is_exhausted:
                # HTTP 410 Gone signals iterator exhausted
                raise HTTPException(status_code=410, detail='Iterator exhausted')
            return types.ProcessorCallResponse(result=result)

        if function_name == '__iter__':
            return types.ProcessorCallResponse(result='ok')
        return types.ProcessorCallResponse(result=result)
