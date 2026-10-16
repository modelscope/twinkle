# Copyright (c) ModelScope Contributors. All rights reserved.
"""Managed training operations on the existing model queue and backend gate."""
from __future__ import annotations

import asyncio
import os
import uuid
from contextvars import ContextVar
from datetime import datetime

from twinkle.protocol.serialize import deserialize_object
from twinkle.protocol.types import model as native
from twinkle.protocol.types.training import CreateModelRequest
from twinkle.protocol.types.training import LoraConfig as RunLoraConfig
from twinkle.server.checkpoint import create_training_run_manager
from twinkle.server.exceptions import ResourceNotFoundError
from twinkle.server.lifecycle.submit import backend_kwargs
from twinkle.server.middleware.auth import get_session_id_from_request, get_token_from_request
from twinkle.server.state.models import FutureFailureRecord
from twinkle.server.utils import get_template_for_model
from twinkle.utils.logger import get_logger
from .records import CreateTraining, TrainingRecord

_tinker_operation = ContextVar('managed_tinker_operation', default=None)
logger = get_logger()


class ManagedTrainingRuntime:

    async def wait_for_training_workers(self):
        # Constructing a remote_class returns actor handles before its worker
        # constructors (including model loading) finish. Full creation otherwise
        # performs no backend call to establish this readiness barrier.
        actors = getattr(self.model, '_actors', None) or []
        await asyncio.gather(*(actor.__ray_ready__.remote() for actor in actors))

    async def initialize_managed_runtime(self, context, placement_group, devices):
        import ray

        from twinkle.infra._ray.ray_helper import RayHelper
        from twinkle.infra._ray.resource_manager import ResourceManager
        self._training_controller = ray.get_actor(context['controller_name'], namespace=context['namespace'])
        await self._training_controller.register_runtime.remote(context['instance_id'], context['epoch'],
                                                                uuid.uuid4().hex)
        os.environ['TWINKLE_RAY_REGISTRY_NAME'] = context['registry_name']
        os.environ['TWINKLE_RAY_NAMESPACE'] = context['namespace']
        RayHelper.resource_manager = ResourceManager.from_allocation(self.device_group, placement_group, devices)

    async def managed_request_start(self, request):
        token = get_token_from_request(request)
        if self._model_unhealthy:
            raise ResourceNotFoundError('Training runtime is unavailable')
        body = await request.json()
        training_id = body.get('training_id') or body.get('model_id')
        session = get_session_id_from_request(request) if '/twinkle/' in request.url.path else None
        if '/twinkle/' in request.url.path and not session:
            raise ResourceNotFoundError('Training requests require the creating session')
        route = await self._training_controller.route.remote(training_id, token, session)
        if route['instance_id'] != self.training_context['instance_id'] or route['epoch'] != self.training_context[
                'epoch']:
            raise ResourceNotFoundError('Runtime does not own this training handle')
        _tinker_operation.set((training_id, body.get('seq_id')) if '/tinker/' in request.url.path else None)
        return token

    async def schedule_task(self, coro_factory, **kwargs):
        operation = _tinker_operation.get()
        if operation is None or operation[1] is None:
            return await super().schedule_task(coro_factory, **kwargs)
        training_id, seq_id = operation
        task_type = kwargs.get('task_type')
        stream = f'tinker::{task_type}'
        request_id = 'req_' + uuid.uuid4().hex
        retention = self.sequence_retention_s
        previous = await self.state.claim_training_seq(training_id, stream, seq_id, request_id, retention)
        if previous is not None:
            return {'request_id': previous, 'model_id': training_id}
        try:
            return await super().schedule_task(coro_factory, request_id=request_id, **kwargs)
        except Exception:
            if await self.state.get_future(request_id) is None:
                await self.state.release_training_seq(training_id, stream, seq_id, request_id)
            raise

    async def _prepare_queued_task(self, task):
        if task.model_id is None:
            return
        context = self.training_context
        await self._training_controller.admit.remote(
            task.model_id,
            task.token,
            None,
            task.request_id,
            renew=task.task_type not in ('get_info', 'get_train_configs'))

        async def enqueued():
            await self._training_controller.commit_operation.remote(task.model_id, task.request_id, context['epoch'])

        async def finished(executed, unsafe):
            if unsafe:
                self._model_unhealthy = True
                self._training_controller.fail_instance.remote(
                    context['instance_id'],
                    FutureFailureRecord(
                        reason_code='execution_timeout',
                        attribution='server',
                        message='Backend execution timed out; runtime is being terminated.'))
            else:
                try:
                    await self._training_controller.finish_operation.remote(
                        task.model_id, task.request_id, context['epoch'], executed=executed)
                except Exception:
                    # The terminal future and queue membership allow the controller
                    # to reconcile a lost notification without stopping compute.
                    logger.exception('Training completion notification failed for %s', task.request_id)

        task.on_enqueued, task.on_finished = enqueued, finished
        if task.task_type == 'apply_patch':
            await self._training_controller.mark_dirty.remote(context['instance_id'], context['epoch'])

    async def _managed_serial(self, factory, task_type):
        ref = await self.schedule_task(factory, task_type=task_type)
        deadline = asyncio.get_running_loop().time() + self.task_queue_config.absolute_future_ttl(1)
        while True:
            future = await self.state.get_future(ref['request_id'])
            if future and future['status'] in ('failed', 'cancelled'):
                raise RuntimeError(future['failure']['message'])
            if future and future['status'] == 'completed':
                return future['result']
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError('Training lifecycle operation has not finished')
            await asyncio.sleep(self.task_queue_config.inline_fast_path_timeout)

    async def initialize_training(self, training_data, request_data):
        training = TrainingRecord.model_validate(training_data)
        request = CreateTraining.model_validate(request_data)

        async def initialize():
            adapter = training.training_id
            backend_adapter = self.resolve_model_adapter_name(adapter)
            self.register_resource(adapter, training.token, training.session_id)
            self.set_resource_state(adapter, 'grad_ready', False)
            if request.client_type == 'twinkle':
                body = native.AddAdapterRequest.model_validate(request.payload)
                config = deserialize_object(body.config)
                if not self.is_full_mode:
                    await self.call_backend(self.model.add_adapter_to_model, adapter, config, **backend_kwargs(body))
                lora = None if config is None else RunLoraConfig(
                    rank=config.r, train_unembed=False, train_mlp=True, train_attn=True)
                run_config = CreateModelRequest(
                    base_model=training.base_model,
                    lora_config=lora,
                    save_dir=body.save_dir,
                    user_metadata={'adapter_name': training.adapter_label})
            else:
                from peft import LoraConfig
                from tinker import types
                run_config = types.CreateModelRequest.model_validate(request.payload)
                if not self.is_full_mode:
                    await self.call_backend(
                        self.model.add_adapter_to_model,
                        adapter_name=adapter,
                        config_or_dir=LoraConfig(r=run_config.lora_config.rank, target_modules='all-linear'))
                await self.call_backend(
                    self.model.set_template,
                    get_template_for_model(self.base_model),
                    adapter_name=backend_adapter,
                    model_id=self.base_model)
                await self.call_backend(self.model.set_processor, 'InputProcessor', adapter_name=backend_adapter)
                await self.call_backend(self.model.set_optimizer, 'Adam', adapter_name=backend_adapter)
            create_training_run_manager(training.token, client_type=request.client_type).save(adapter, run_config)

        await self._managed_serial(initialize, 'initialize_training')

    async def release_training(self, training_data, reset_full):
        training = TrainingRecord.model_validate(training_data)

        async def release():
            adapter = training.training_id
            if self.get_resource_info(adapter) is None:
                return
            if self.is_full_mode:
                if reset_full:
                    await self.call_backend(self.model.reload_initial_weights, reset_training_state=True)
            else:
                await self.call_backend(self.model.remove_adapter, adapter)
            self.clear_resource_state(adapter)
            self.unregister_resource(adapter)

        await self._managed_serial(release, 'release_training')

    async def runtime_identity(self):
        import ray
        actors = getattr(self.model, '_actors', None) or []
        states = await asyncio.to_thread(
            lambda: [ray._private.state.actors(actor._actor_id.hex()).get('State') for actor in actors])
        return {
            'pid': os.getpid(),
            'instance_id': self.training_context['instance_id'],
            'unhealthy': self._model_unhealthy or any(state != 'ALIVE' for state in states)
        }

    async def verify_operation(self, request_id):
        if self._model_unhealthy:
            return {'state': 'unsafe'}
        if self._compute_worker.has_task(request_id):
            return {'state': 'pending'}
        future = await self.state.get_future(request_id)
        if future and future['status'] in ('completed', 'failed', 'cancelled'):
            return {
                'state': 'finished',
                'executed': future['status'] != 'cancelled',
                'finished_at': datetime.fromisoformat(future['updated_at']).timestamp()
            }
        return {'state': 'missing'}
