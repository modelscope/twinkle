# Copyright (c) ModelScope Contributors. All rights reserved.
"""Single-writer training lifecycle; slow runtime work stays outside the lock."""
from __future__ import annotations

import asyncio
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from datetime import datetime
from typing import Any, Protocol

from twinkle.server.config.training import TrainingConfig
from twinkle.server.exceptions import RequestRejectedError, ResourceNotFoundError, ResourceQuotaExceededError
from twinkle.server.state.models import FutureFailureRecord
from .catalog import TrainingCatalog, TrainingSpec
from .errors import CapacityUnavailableError, TrainingExpiredError
from .records import LIVE_INSTANCES, TERMINAL_TRAINING, CreateTraining, InstanceRecord, TrainingRecord
from .store import TrainingStore

logger = logging.getLogger(__name__)


class InstanceDriver(Protocol):

    async def start(self, instance: InstanceRecord, spec: TrainingSpec) -> dict[str, Any]:
        ...

    async def create_training(self, instance: InstanceRecord, training: TrainingRecord,
                              request: CreateTraining) -> None:
        ...

    async def release_training(self, instance: InstanceRecord, training: TrainingRecord) -> None:
        ...

    async def shutdown(self, instance: InstanceRecord) -> None:
        ...

    async def verify_operation(self, instance: InstanceRecord, request_id: str) -> dict[str, Any]:
        ...

    async def recover_orphans(self) -> None:
        ...


def failure_for(exc: Exception) -> FutureFailureRecord:
    if isinstance(exc, CapacityUnavailableError):
        reason, attribution = 'capacity_unavailable', 'user'
    elif isinstance(exc, ResourceQuotaExceededError):
        reason, attribution = 'resource_quota_exceeded', 'user'
    elif isinstance(exc, TimeoutError):
        reason, attribution = 'startup_timeout', 'server'
    elif isinstance(exc, RequestRejectedError):
        reason, attribution = 'invalid_request', 'user'
    else:
        reason, attribution = 'startup_failed', 'server'
    return FutureFailureRecord(
        reason_code=reason,
        message=str(exc)[:1024],
        attribution=attribution,
        diagnostic=traceback.format_exc() if attribution == 'server' else None)


class TrainingController:

    def __init__(self,
                 config: TrainingConfig,
                 store: TrainingStore,
                 driver: InstanceDriver,
                 *,
                 clock: Callable[[], float] = time.monotonic):
        self.config = config
        self.catalog = TrainingCatalog(config)
        self.store = store
        self.driver = driver
        self.clock = clock
        self.epoch = 'controller_' + uuid.uuid4().hex
        self.instances: dict[str, InstanceRecord] = {}
        self.trainings: dict[str, TrainingRecord] = {}
        self._startup: dict[str, asyncio.Task] = {}
        self._jobs: set[asyncio.Task] = set()
        self._cleaning: set[str] = set()
        self._closing: set[str] = set()
        self._reconciling: set[str] = set()
        self._lock = asyncio.Lock()
        self._ready = False

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._jobs.add(task)

        def done(completed):
            self._jobs.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(done)
        return task

    async def initialize(self) -> None:
        """Discard old state only after owned compute has been confirmed dead."""
        old_instances = await self.store.instances()
        old_trainings = await self.store.trainings()
        old_controller = await self.store.backend.get(self.store.key('controller', 'active'))
        await self.store.begin_epoch(self.epoch)
        failure = FutureFailureRecord(
            reason_code='orphaned_replica',
            message='Training controller restarted; create a new training handle.',
            attribution='server')
        for training in old_trainings:
            for request_id in (training.create_request_id, *training.pending_operation_ids):
                await self.store.future(request_id, 'failed', training.training_id, failure=failure)
        await self.driver.recover_orphans()
        for instance in old_instances:
            if instance.state != 'STOPPED':
                await asyncio.wait_for(self.driver.shutdown(instance), self.config.shutdown_timeout_s)
                instance.state = 'STOPPED'
                instance.stopped_at = time.time()
                await self.store.save_instance(instance)
        for training in old_trainings:
            training.state = 'FAILED'
            training.terminal_reason = 'orphaned_replica'
            training.terminal_at = time.time()
            await self.store.retire(training)
        await self.store.state.rebuild_model_indexes()
        if old_controller:
            await self.store.state.unregister_replica(old_controller['epoch'])
        await self.store.state.register_replica(self.epoch, 0)
        await self.store.state.touch_replica_last_seen(self.epoch)
        try:
            for spec in self.catalog.specs():
                for _ in range(spec.config.min_replicas):
                    async with self._lock:
                        instance = await self._new_instance(spec, resident=True)
                    await asyncio.shield(self._startup[instance.instance_id])
        except Exception:
            await self.shutdown()
            raise
        self._ready = True

    @property
    def residents_ready(self) -> bool:
        return all(self.residents_ready_for(spec) for spec in self.catalog.specs())

    def residents_ready_for(self, spec: TrainingSpec) -> bool:
        return sum(
            instance.resident and (instance.base_model, instance.mode) == spec.key and instance.state in ('READY',
                                                                                                          'RESETTING')
            for instance in self.instances.values()) >= spec.config.min_replicas

    async def register_runtime(self, instance_id: str, epoch: str, incarnation: str) -> None:
        async with self._lock:
            await self._assert_epoch()
            instance = self.instances.get(instance_id)
            if (epoch != self.epoch or instance is None or instance.incarnation is not None
                    or instance.state not in ('ALLOCATING', 'STARTING')):
                raise ResourceNotFoundError('Runtime incarnation is no longer allowed')
            instance.incarnation = incarnation
            instance.state = 'STARTING'
            await self.store.save_instance(instance)

    async def mark_dirty(self, instance_id: str, epoch: str) -> None:
        async with self._lock:
            await self._assert_epoch()
            instance = self.instances.get(instance_id)
            if epoch != self.epoch or instance is None:
                raise ResourceNotFoundError('Unknown runtime')
            instance.dirty = True
            await self.store.save_instance(instance)

    async def _assert_epoch(self) -> None:
        active = await self.store.backend.get(self.store.key('controller', 'active'))
        if not active or active['epoch'] != self.epoch:
            raise ResourceNotFoundError('Training controller epoch is no longer active')

    async def _new_instance(self, spec: TrainingSpec, *, resident: bool = False) -> InstanceRecord:
        instance = InstanceRecord(
            instance_id='instance_' + uuid.uuid4().hex,
            controller_epoch=self.epoch,
            base_model=spec.model.name,
            mode=spec.mode,
            resident=resident,
            startup_deadline=self.clock() + self.config.startup_timeout_s)
        await self.store.save_instance(instance)
        self.instances[instance.instance_id] = instance
        self._startup[instance.instance_id] = self._spawn(self._start_instance(instance))
        return instance

    async def _start_instance(self, instance: InstanceRecord) -> None:
        spec = self.catalog.get(instance.base_model, instance.mode)
        try:
            details = await asyncio.wait_for(self.driver.start(instance, spec), self.config.startup_timeout_s)
            async with self._lock:
                await self._assert_epoch()
                if self.clock() >= instance.startup_deadline:
                    raise TimeoutError('Training runtime became ready after its startup deadline')
                if instance.state not in ('ALLOCATING', 'STARTING'):
                    raise RuntimeError('Instance became unavailable during startup')
                for name, value in details.items():
                    if name not in ('application_name', 'placement_group_id', 'node_id', 'devices'):
                        raise ValueError(f'Unknown runtime allocation field: {name}')
                    setattr(instance, name, value)
                instance.state = 'READY'
                await self.store.save_instance(instance)
        except Exception as exc:
            await self.fail_instance(instance.instance_id, failure_for(exc))
            raise

    def _select(self, spec: TrainingSpec) -> InstanceRecord | None:
        candidates = [
            instance for instance in self.instances.values()
            if (instance.base_model, instance.mode) == spec.key and instance.state in (
                'READY', 'ALLOCATING', 'STARTING') and len(instance.training_ids) < spec.capacity
        ]
        return min(
            candidates,
            key=lambda instance: (instance.state != 'READY', -len(instance.training_ids), instance.instance_id),
            default=None)

    async def create(self, request: CreateTraining) -> dict[str, str]:
        # Catalog/mode errors are rejected before a future or any resource exists.
        spec = self.catalog.get(request.base_model, request.mode)
        identity, fingerprint = self.store.create_identity(request)
        async with self._lock:
            await self._assert_epoch()
            if not self._ready:
                raise RequestRejectedError('Training service is not ready')
            previous = await self.store.get_create(identity)
            if previous is not None:
                if previous['fingerprint'] != fingerprint:
                    raise RequestRejectedError('Create identity was reused with different parameters')
                return {'request_id': previous['request_id']}
            request_id = 'req_' + uuid.uuid4().hex
            training_id = 'training_' + uuid.uuid4().hex
            await self.store.future(
                request_id,
                'pending',
                training_id,
                epoch=self.epoch,
                deadline=time.time() + 2 * self.config.startup_timeout_s)
            await self.store.own_future(request_id, request.token)
            await self.store.save_create(identity, fingerprint, request_id, request.session_id)
            training = None
            try:
                # Reserve quota before starting an instance, but leave no usable route.
                training = TrainingRecord(
                    training_id=training_id,
                    token=request.token,
                    session_id=request.session_id,
                    base_model=request.base_model,
                    mode=request.mode,
                    controller_epoch=self.epoch,
                    training_namespace=self.store.prefix,
                    instance_id='',
                    adapter_label=request.adapter_label,
                    create_request_id=request_id,
                    creation_deadline=self.clock() + self.config.startup_timeout_s,
                    last_activity_at=self.clock(),
                    lora_config=request.payload.get('lora_config'),
                    user_metadata=request.payload.get('user_metadata') or {})
                await self.store.reserve(training)
                instance = self._select(spec)
                if instance is None:
                    if not self.residents_ready_for(spec):
                        raise CapacityUnavailableError(
                            'Resident training capacity is recovering; retry a new create later.')
                    count = sum((item.base_model, item.mode) == spec.key and item.state != 'STOPPED'
                                for item in self.instances.values())
                    if spec.config.max_replicas is not None and count >= spec.config.max_replicas:
                        raise CapacityUnavailableError('Training replica limit reached; retry a new create later.')
                    instance = await self._new_instance(spec)
                training.instance_id = instance.instance_id
                training.replica_id = instance.instance_id
                instance.training_ids.append(training_id)
                self.trainings[training_id] = training
                await self.store.save_training(training)
                await self.store.save_instance(instance)
                self._spawn(self._create_user(training, request))
            except Exception as exc:
                if training is not None:
                    self.trainings.pop(training_id, None)
                    if training.instance_id in self.instances:
                        instance = self.instances[training.instance_id]
                        if training_id in instance.training_ids:
                            instance.training_ids.remove(training_id)
                        await self.store.save_instance(instance)
                        if not instance.training_ids and not instance.resident:
                            self._spawn(self.fail_instance(instance.instance_id, failure_for(exc)))
                    training.state = 'FAILED'
                    training.terminal_reason = failure_for(exc).reason_code
                    await self.store.retire(training)
                await self.store.future(request_id, 'failed', training_id, failure=failure_for(exc))
            return {'request_id': request_id}

    async def _create_user(self, training: TrainingRecord, request: CreateTraining) -> None:
        instance = self.instances[training.instance_id]
        try:
            await asyncio.shield(self._startup[instance.instance_id])
            async with self._lock:
                future = await self.store.state.get_future(training.create_request_id)
                if future and future['status'] == 'cancelled':
                    training.state = 'CLOSING'
                    training.terminal_reason = 'cancelled'
                if training.state != 'PENDING' or instance.state != 'READY':
                    raise RequestRejectedError('Training creation is no longer pending')
                training.creation_deadline = self.clock() + self.config.startup_timeout_s
                await self.store.save_training(training)
                await self.store.future(training.create_request_id, 'running', training.training_id)
            remaining = training.creation_deadline - self.clock()
            await asyncio.wait_for(self.driver.create_training(instance, training, request), max(0, remaining))
            # Driver includes training-run metadata commit, before public success.
            async with self._lock:
                await self._assert_epoch()
                if self.clock() >= training.creation_deadline:
                    raise TimeoutError('Training initialization completed after its creation deadline')
                if training.state != 'PENDING' or instance.state != 'READY':
                    raise RuntimeError('Training became unavailable during initialization')
                training.state = 'ACTIVE'
                training.last_activity_at = self.clock()
                await self.store.save_training(training)
                result = ({
                    'model_id': training.training_id
                } if request.client_type == 'tinker' else {
                    'training_id': training.training_id,
                    'adapter_name': training.adapter_label
                })
                await self.store.future(training.create_request_id, 'completed', training.training_id, result=result)
        except Exception as exc:
            if training.state not in TERMINAL_TRAINING:
                await self.store.future(
                    training.create_request_id, 'failed', training.training_id, failure=failure_for(exc))
                async with self._lock:
                    training.state = 'CLOSING'
                    training.terminal_reason = ('cancelled' if training.terminal_reason == 'cancelled' else
                                                failure_for(exc).reason_code)
                    await self.store.save_training(training)
                await self._cleanup_user(training)

    async def _expired(self, training: TrainingRecord) -> bool:
        spec = self.catalog.get(training.base_model, training.mode)
        config, legacy = spec.config, spec.model.legacy_expiration
        heartbeat = await self.store.state.get_session_last_heartbeat(training.session_id)
        if heartbeat is None:
            return True
        if legacy is not None:
            now = time.time()
            if legacy.session_timeout_s is not None and now - heartbeat >= legacy.session_timeout_s:
                return True
            if legacy.max_lifetime_s is not None:
                created_at = datetime.fromisoformat(training.created_at).timestamp()
                if now - created_at >= legacy.max_lifetime_s:
                    return True
        return config.idle_timeout_s is not None and self.clock() - training.last_activity_at >= config.idle_timeout_s

    async def _authorized(self, training_id: str, token: str, session_id: str | None) -> TrainingRecord:
        await self._assert_epoch()
        training = self.trainings.get(training_id)
        if training is None or training.token != token or (session_id is not None
                                                           and session_id != training.session_id):
            raise ResourceNotFoundError('Unknown training handle')
        if training.state == 'EXPIRED':
            raise TrainingExpiredError('Training handle expired; create a new handle and explicitly load a checkpoint.')
        if training.state != 'ACTIVE':
            raise ResourceNotFoundError('Training handle is no longer active')
        if await self.store.state.get_session_last_heartbeat(training.session_id) is None:
            raise TrainingExpiredError('Training session expired; create a new training handle')
        return training

    async def admit(self,
                    training_id: str,
                    token: str,
                    session_id: str | None,
                    request_id: str,
                    *,
                    renew: bool = True) -> dict[str, str]:
        async with self._lock:
            training = await self._authorized(training_id, token, session_id)
            if not training.pending_operation_ids and await self._expired(training):
                training.state = 'CLOSING'
                training.terminal_reason = 'expired'
                await self.store.save_training(training)
                self._spawn(self._cleanup_user(training))
                raise TrainingExpiredError(
                    'Training handle expired; create a new handle and explicitly load a checkpoint.')
            instance = self.instances[training.instance_id]
            if instance.state != 'READY':
                raise ResourceNotFoundError('Training runtime is unavailable')
            if request_id not in training.pending_operation_ids:
                training.pending_operation_ids[request_id] = {
                    'accepted_at': self.clock(),
                    'committed': False,
                    'renew': renew
                }
                await self.store.save_training(training)
                await self.store.own_future(request_id, token)
            return {'instance_id': instance.instance_id, 'epoch': self.epoch}

    async def commit_operation(self, training_id: str, request_id: str, epoch: str) -> None:
        async with self._lock:
            await self._assert_epoch()
            training = self.trainings.get(training_id)
            if epoch != self.epoch or training is None:
                raise ResourceNotFoundError('Operation admission no longer exists')
            if request_id not in training.pending_operation_ids:
                # A very short task can finish while the enqueue acknowledgment
                # is in flight. Its original terminal future is conclusive.
                future = await self.store.state.get_future(request_id)
                if future and future['status'] in ('completed', 'failed', 'cancelled'):
                    return
                raise ResourceNotFoundError('Operation admission no longer exists')
            training.pending_operation_ids[request_id]['committed'] = True
            await self.store.save_training(training)

    async def finish_operation(self,
                               training_id: str,
                               request_id: str,
                               epoch: str,
                               *,
                               executed: bool,
                               finished_at: float | None = None) -> None:
        async with self._lock:
            try:
                await self._assert_epoch()
            except ResourceNotFoundError:
                return
            training = self.trainings.get(training_id)
            if epoch != self.epoch or training is None:
                return
            ticket = training.pending_operation_ids.pop(request_id, None)
            if ticket is None:
                return
            if executed and ticket['renew']:
                age = max(0, time.time() - finished_at) if finished_at is not None else 0
                training.last_activity_at = max(training.last_activity_at, self.clock() - age)
            await self.store.save_training(training)
            cleanup = training.state == 'CLOSING' and not training.pending_operation_ids
        if cleanup:
            self._spawn(self._cleanup_user(training))

    async def route(self, training_id: str, token: str, session_id: str | None = None) -> dict[str, str]:
        async with self._lock:
            training = await self._authorized(training_id, token, session_id)
            return {'instance_id': training.instance_id, 'base_model': training.base_model, 'epoch': self.epoch}

    async def release(self, training_id: str, token: str, session_id: str | None = None) -> None:
        async with self._lock:
            training = await self._authorized(training_id, token, session_id)
            training.state = 'CLOSING'
            training.terminal_reason = 'released'
            await self.store.save_training(training)
            cleanup = not training.pending_operation_ids
        if cleanup:
            await self._cleanup_user(training)
        config = self.catalog.get(training.base_model, training.mode).config.queue_config
        deadline = asyncio.get_running_loop().time() + config.absolute_future_ttl(1) + self.config.shutdown_timeout_s
        while training.state == 'CLOSING' or self.instances[training.instance_id].state == 'DRAINING':
            if self.instances[training.instance_id].state == 'QUARANTINED':
                raise RuntimeError('Runtime cleanup is quarantined; device release is not confirmed')
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError('Training cleanup is still pending; its slot remains reserved')
            await asyncio.sleep(config.inline_fast_path_timeout)

    async def _cleanup_user(self, training: TrainingRecord) -> None:
        instance = self.instances[training.instance_id]
        # Each training must have exactly one cleanup, including concurrent finish callbacks.
        async with self._lock:
            if (instance.state not in ('READY', 'RESETTING') or training.state != 'CLOSING'
                    or training.pending_operation_ids or training.training_id in self._cleaning):
                return
            self._cleaning.add(training.training_id)
            resetting = instance.resident and instance.mode == 'full' and not instance.dirty
            if resetting:
                instance.state = 'RESETTING'
                await self.store.save_instance(instance)
            closing_full = instance.mode == 'full' and (not instance.resident or instance.dirty)
            if closing_full:
                instance.state = 'DRAINING'
                await self.store.save_instance(instance)
        if closing_full:
            await self._close_instance(instance)
            self._cleaning.discard(training.training_id)
            return
        try:
            spec = self.catalog.get(training.base_model, training.mode)
            await asyncio.wait_for(
                self.driver.release_training(instance, training),
                spec.config.queue_config.absolute_future_ttl(spec.config.num_devices))
        except Exception as exc:
            # A timed-out create/cleanup RPC does not prove that shared workers failed.
            # Keep this slot reserved until serial cleanup can be confirmed.
            if instance.mode == 'lora' and any(
                    self.trainings[identity].state in ('ACTIVE',
                                                       'PENDING') or self.trainings[identity].pending_operation_ids
                    for identity in instance.training_ids if identity != training.training_id):
                logger.warning('Training cleanup remains pending for %s: %s', training.training_id, exc)
            else:
                await self.fail_instance(instance.instance_id, failure_for(exc))
            self._cleaning.discard(training.training_id)
            return
        async with self._lock:
            training.state = {
                'expired': 'EXPIRED',
                'released': 'RELEASED',
                'cancelled': 'CANCELLED'
            }.get(training.terminal_reason, 'FAILED')
            training.terminal_at = time.time()
            await self.store.retire(training)
            self._cleaning.discard(training.training_id)
            if training.training_id in instance.training_ids:
                instance.training_ids.remove(training.training_id)
            close = not instance.training_ids and (not instance.resident or instance.dirty)
            if close:
                instance.state = 'DRAINING'
            elif resetting:
                instance.state = 'READY'
            await self.store.save_instance(instance)
        if close:
            await self._close_instance(instance)

    async def _close_instance(self, instance: InstanceRecord) -> None:
        if instance.instance_id in self._closing:
            return
        self._closing.add(instance.instance_id)
        try:
            try:
                await asyncio.wait_for(self.driver.shutdown(instance), self.config.shutdown_timeout_s)
            except Exception as exc:
                async with self._lock:
                    instance.state = 'QUARANTINED'
                    instance.failure = str(exc)
                    await self.store.save_instance(instance)
                return
            async with self._lock:
                instance.state = 'STOPPED'
                instance.stopped_at = time.time()
                await self.store.save_instance(instance)
                for training_id in list(instance.training_ids):
                    training = self.trainings[training_id]
                    training.pending_operation_ids.clear()
                    training.state = {
                        'expired': 'EXPIRED',
                        'released': 'RELEASED',
                        'cancelled': 'CANCELLED'
                    }.get(training.terminal_reason, 'FAILED')
                    training.terminal_at = time.time()
                    await self.store.retire(training)
                instance.training_ids.clear()
                await self.store.save_instance(instance)
        finally:
            self._closing.discard(instance.instance_id)

    async def fail_instance(self, instance_id: str, failure: FutureFailureRecord) -> None:
        async with self._lock:
            instance = self.instances[instance_id]
            if instance.state in ('DRAINING', 'STOPPED', 'QUARANTINED'):
                return
            instance.state = 'DRAINING'
            instance.failure = failure.message
            await self.store.save_instance(instance)
            for training_id in instance.training_ids:
                training = self.trainings[training_id]
                training.state = 'CLOSING'
                training.terminal_reason = failure.reason_code
                for request_id in (training.create_request_id, *training.pending_operation_ids):
                    await self.store.future(request_id, 'failed', training_id, failure=failure)
                await self.store.save_training(training)
        startup = self._startup.get(instance_id)
        if startup is not None and startup is not asyncio.current_task() and not startup.done():
            startup.cancel()
            await asyncio.gather(startup, return_exceptions=True)
        await self._close_instance(instance)

    async def sweep(self) -> None:
        async with self._lock:
            await self._assert_epoch()
            for training in self.trainings.values():
                if training.state != 'ACTIVE' or training.pending_operation_ids:
                    continue
                if await self._expired(training):
                    training.state = 'CLOSING'
                    training.terminal_reason = 'expired'
                    await self.store.save_training(training)
            uncertain = [(training, request_id) for training in self.trainings.values()
                         for request_id, ticket in training.pending_operation_ids.items()
                         if request_id not in self._reconciling and self.clock()
                         - ticket['accepted_at'] >= self.config.sweep_interval_s]
            self._reconciling.update(request_id for _, request_id in uncertain)
            closing = [
                training for training in self.trainings.values() if training.state == 'CLOSING'
                and not training.pending_operation_ids and training.training_id not in self._cleaning
            ]
        for training in closing:
            self._spawn(self._cleanup_user(training))
        for training, request_id in uncertain:
            self._spawn(self._reconcile_operation(training, request_id))
        for instance in list(self.instances.values()):
            if instance.state == 'QUARANTINED':
                self._spawn(self._close_instance(instance))
        # Rebuilding one specification must not stop expiry or other models.
        for spec in self.catalog.specs():
            residents = sum(
                instance.resident and (instance.base_model,
                                       instance.mode) == spec.key and instance.state in LIVE_INSTANCES
                for instance in self.instances.values())
            for _ in range(spec.config.min_replicas - residents):
                async with self._lock:
                    instance = await self._new_instance(spec, resident=True)
        await self.store.cleanup_receipts()
        # Terminal diagnostics expire like their backend tombstones. Repeated
        # model creation must not grow controller memory for the server lifetime.
        async with self._lock:
            cutoff = time.time() - self.store.state.expiration_timeout
            for identity, training in list(self.trainings.items()):
                if training.state in TERMINAL_TRAINING and training.terminal_at is not None:
                    if training.terminal_at <= cutoff:
                        self.trainings.pop(identity)
            for identity, instance in list(self.instances.items()):
                if (instance.state == 'STOPPED' and instance.stopped_at is not None and instance.stopped_at <= cutoff
                        and not instance.training_ids):
                    self.instances.pop(identity)
                    self._startup.pop(identity, None)

    async def _reconcile_operation(self, training: TrainingRecord, request_id: str) -> None:
        try:
            instance = self.instances[training.instance_id]
            if instance.state != 'READY':
                return
            result = await asyncio.wait_for(
                self.driver.verify_operation(instance, request_id), self.config.shutdown_timeout_s)
            if result['state'] == 'pending':
                await self.commit_operation(training.training_id, request_id, self.epoch)
            elif result['state'] == 'finished':
                await self.finish_operation(
                    training.training_id,
                    request_id,
                    self.epoch,
                    executed=result['executed'],
                    finished_at=result.get('finished_at'))
            elif result['state'] == 'missing':
                await self.store.future(
                    request_id,
                    'failed',
                    training.training_id,
                    failure=FutureFailureRecord(
                        reason_code='internal_error',
                        attribution='server',
                        message='Operation was not committed to the runtime queue'))
                await self.finish_operation(training.training_id, request_id, self.epoch, executed=False)
            # An unsafe/unreachable runtime is handled by the worker health monitor.
        except Exception:
            logger.exception('Operation reconciliation deferred for %s', request_id)
        finally:
            self._reconciling.discard(request_id)

    async def wait_for_jobs(self) -> None:
        while self._jobs:
            jobs = tuple(self._jobs)
            await asyncio.gather(*jobs, return_exceptions=True)
            self._jobs.difference_update(task for task in jobs if task.done())

    async def shutdown(self) -> None:
        self._ready = False
        for instance in list(self.instances.values()):
            await self.fail_instance(
                instance.instance_id,
                FutureFailureRecord(
                    reason_code='orphaned_replica', message='Training service is shutting down', attribution='server'))
