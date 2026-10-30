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

from twinkle.server.config.training import TrainingConfig, TrainingModelConfig
from twinkle.server.exceptions import (
    RequestRejectedError,
    ResourceExhaustedError,
    ResourceNotFoundError,
    ResourceQuotaExceededError,
    TrainingExpiredError,
)
from twinkle.server.resources.allocation import CapacityUnavailableError
from twinkle.server.resources.instances import InstanceLifecycle
from twinkle.server.state.models import FutureFailureRecord
from twinkle.server.state.training_records import (
    LIVE_INSTANCES,
    TERMINAL_TRAINING,
    CreateTraining,
    InstanceRecord,
    RuntimeOwner,
    TrainingRecord,
    TrainingRoute,
)
from twinkle.server.state.training_store import TrainingStore
from twinkle.server.telemetry.metrics import TrainingInstanceMetrics, TrainingInstanceSnapshot, get_training_metrics

logger = logging.getLogger(__name__)


class InstanceDriver(Protocol):
    async def allocate(self, instance: InstanceRecord, spec: TrainingModelConfig) -> dict[str, Any]: ...

    async def deploy(self, instance: InstanceRecord, spec: TrainingModelConfig) -> None: ...

    async def create_training(
        self, instance: InstanceRecord, training: TrainingRecord, request: CreateTraining
    ) -> None: ...

    async def release_training(self, instance: InstanceRecord, training: TrainingRecord) -> None: ...

    async def shutdown(self, instance: InstanceRecord) -> None: ...

    async def verify_operation(self, instance: InstanceRecord, request_id: str) -> dict[str, Any]: ...

    async def recover_orphans(self) -> None: ...


def failure_for(exc: Exception) -> FutureFailureRecord:
    if isinstance(exc, CapacityUnavailableError):
        reason, attribution = "capacity_unavailable", "user"
    elif isinstance(exc, ResourceQuotaExceededError):
        reason, attribution = "resource_quota_exceeded", "user"
    elif isinstance(exc, TimeoutError):
        reason, attribution = "startup_timeout", "server"
    elif isinstance(exc, RequestRejectedError):
        reason, attribution = "invalid_request", "user"
    else:
        reason, attribution = "startup_failed", "server"
    return FutureFailureRecord(
        reason_code=reason,
        message=str(exc)[:1024],
        attribution=attribution,
        diagnostic=traceback.format_exc() if attribution == "server" else None,
    )


class TrainingController:
    def __init__(
        self,
        config: TrainingConfig,
        store: TrainingStore,
        driver: InstanceDriver,
        *,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config = config
        self.store = store
        self.driver = driver
        self.clock = clock
        self.epoch = "controller_" + uuid.uuid4().hex
        self.instances: dict[str, InstanceRecord] = {}
        self.trainings: dict[str, TrainingRecord] = {}
        self.lifecycle = InstanceLifecycle("Training")
        self._cleaning: set[str] = set()
        self._reconciling: set[str] = set()
        self._failed_instances: dict[str, FutureFailureRecord] = {}
        self._lock = asyncio.Lock()
        self._ready = False
        self._metrics = get_training_metrics()
        self._instance_metrics = TrainingInstanceMetrics(self._metrics)

    def _instance_metric_tags(self, instance: InstanceRecord) -> dict[str, str]:
        spec = self.config.get(instance.base_model, instance.mode)
        return {
            "backend": spec.backend,
            "mode": instance.mode,
            "state": instance.state,
            "resident": str(instance.resident).lower(),
        }

    def _report_instance_metrics(self) -> None:
        snapshots = []
        for instance in self.instances.values():
            if instance.state == "STOPPED":
                continue
            model = self.config.get(instance.base_model, instance.mode)
            snapshots.append(
                TrainingInstanceSnapshot(
                    **self._instance_metric_tags(instance),
                    device_type=model.device_type,
                    device_count=len(instance.devices),
                    placement_group_count=int(bool(instance.application_name)),
                )
            )
        self._instance_metrics.report(snapshots)

    async def initialize(self) -> None:
        """Discard old state only after owned compute has been confirmed dead."""
        old_instances = await self.store.instances()
        old_trainings = await self.store.trainings()
        old_controller = await self.store.backend.get(self.store.key("controller", "active"))
        await self.store.begin_epoch(self.epoch)
        failure = FutureFailureRecord(
            reason_code="orphaned_replica",
            message="Training controller restarted; create a new training handle.",
            attribution="server",
        )
        for training in old_trainings:
            for request_id in (training.create_request_id, *training.pending_operation_ids):
                await self.store.future(request_id, "failed", training.training_id, failure=failure)
        await self.driver.recover_orphans()
        for instance in old_instances:
            if instance.state != "STOPPED":
                await asyncio.wait_for(self.driver.shutdown(instance), self.config.shutdown_timeout_s)
                instance.state = "STOPPED"
                instance.stopped_at = time.time()
                await self.store.save_instance(instance)
        for training in old_trainings:
            training.state = "FAILED"
            training.terminal_reason = "orphaned_replica"
            training.terminal_at = time.time()
            await self.store.retire(training)
        await self.store.models.rebuild_indexes()
        if old_controller:
            await self.store.models.unregister_replica(old_controller["epoch"])
        await self.store.models.register_replica(self.epoch, 0)
        await self.store.models.touch_replica_last_seen(self.epoch)
        try:
            for spec in self.config.models:
                for _ in range(spec.config.min_replicas):
                    async with self._lock:
                        instance = await self._new_instance(spec, resident=True)
                    await asyncio.shield(self.lifecycle.startups[instance.instance_id])
        except Exception:
            await self.shutdown()
            raise
        self._ready = True

    @property
    def residents_ready(self) -> bool:
        return all(self.residents_ready_for(spec) for spec in self.config.models)

    def residents_ready_for(self, spec: TrainingModelConfig) -> bool:
        return (
            sum(
                instance.resident
                and instance.instance_id not in self._failed_instances
                and (instance.base_model, instance.mode) == spec.key
                and instance.state in ("READY", "RESETTING")
                for instance in self.instances.values()
            )
            >= spec.config.min_replicas
        )

    async def register_runtime(self, instance_id: str, epoch: str, incarnation: str) -> None:
        async with self._lock:
            await self._assert_epoch()
            instance = self.instances.get(instance_id)
            if (
                epoch != self.epoch
                or instance is None
                or instance.incarnation is not None
                or instance.state not in ("ALLOCATING", "STARTING")
            ):
                raise ResourceNotFoundError("Runtime incarnation is no longer allowed")
            instance.incarnation = incarnation
            instance.state = "STARTING"
            await self.store.save_instance(instance)
            self._report_instance_metrics()

    async def mark_dirty(self, instance_id: str, epoch: str) -> None:
        async with self._lock:
            await self._assert_epoch()
            instance = self.instances.get(instance_id)
            if epoch != self.epoch or instance is None:
                raise ResourceNotFoundError("Unknown runtime")
            instance.dirty = True
            await self.store.save_instance(instance)

    async def _assert_epoch(self) -> None:
        active = await self.store.backend.get(self.store.key("controller", "active"))
        if not active or active["epoch"] != self.epoch:
            raise ResourceNotFoundError("Training controller epoch is no longer active")

    async def _new_instance(self, spec: TrainingModelConfig, *, resident: bool = False) -> InstanceRecord:
        matching = [item for item in self.instances.values() if (item.base_model, item.mode) == spec.key]
        if not self.lifecycle.headroom(matching, spec.config.max_replicas):
            raise CapacityUnavailableError("Training replica limit reached")
        instance = InstanceRecord(
            instance_id="instance_" + uuid.uuid4().hex,
            controller_epoch=self.epoch,
            base_model=spec.name,
            mode=spec.mode,
            resident=resident,
            startup_deadline=self.clock() + self.config.startup_timeout_s,
        )
        await self.store.save_instance(instance)
        self.instances[instance.instance_id] = instance
        self._report_instance_metrics()
        self.lifecycle.start(instance.instance_id, lambda: self._start_instance(instance))
        return instance

    async def _start_instance(self, instance: InstanceRecord) -> None:
        spec = self.config.get(instance.base_model, instance.mode)
        started, status = time.monotonic(), "error"
        tags = {"backend": spec.backend, "mode": spec.mode}
        try:

            async def startup():
                details = await self.driver.allocate(instance, spec)
                async with self._lock:
                    await self._assert_epoch()
                    if instance.state != "ALLOCATING":
                        raise RuntimeError("Instance became unavailable during allocation")
                    for name in details:
                        if name not in ("application_name", "placement_group_id", "node_id", "devices"):
                            raise ValueError(f"Unknown runtime allocation field: {name}")
                    allocated = instance.model_copy(deep=True, update=details)
                    await self.store.save_instance(allocated)
                    for name, value in details.items():
                        setattr(instance, name, value)
                    self._report_instance_metrics()
                # Ownership is persisted before any worker is constructed; the driver does not write records.
                await self.driver.deploy(instance.model_copy(deep=True), spec)

            await asyncio.wait_for(startup(), self.config.startup_timeout_s)
            async with self._lock:
                await self._assert_epoch()
                if self.clock() >= instance.startup_deadline:
                    raise TimeoutError("Training runtime became ready after its startup deadline")
                if instance.state not in ("ALLOCATING", "STARTING"):
                    raise RuntimeError("Instance became unavailable during startup")
                ready = instance.model_copy(update={"state": "READY"})
                await self.store.save_instance(ready)
                instance.state = "READY"
                self._report_instance_metrics()
            status = "success"
        except Exception as exc:
            await self.fail_instance(instance.instance_id, failure_for(exc))
            raise
        finally:
            self._metrics.startup_seconds.observe(time.monotonic() - started, tags={**tags, "status": status})

    def _select(self, spec: TrainingModelConfig) -> InstanceRecord | None:
        candidates = [
            instance
            for instance in self.instances.values()
            if instance.instance_id not in self._failed_instances
            and (instance.base_model, instance.mode) == spec.key
            and instance.state in ("READY", "ALLOCATING", "STARTING")
            and len(instance.training_ids) < spec.capacity
        ]
        return min(
            candidates,
            key=lambda instance: (instance.state != "READY", -len(instance.training_ids), instance.instance_id),
            default=None,
        )

    async def create(self, request: CreateTraining) -> dict[str, str]:
        # Catalog/mode errors are rejected before a future or any resource exists.
        spec = self.config.get(request.base_model, request.mode)
        identity, fingerprint = self.store.create_identity(request)
        async with self._lock:
            await self._assert_epoch()
            if not self._ready:
                raise ResourceExhaustedError("Training service is not ready")
            previous = await self.store.get_create(identity)
            if previous is not None:
                if previous["fingerprint"] != fingerprint:
                    raise RequestRejectedError("Create identity was reused with different parameters")
                return {"request_id": previous["request_id"]}
            request_id = "req_" + uuid.uuid4().hex
            training_id = "training_" + uuid.uuid4().hex
            await self.store.future(
                request_id,
                "pending",
                training_id,
                epoch=self.epoch,
                deadline=time.time() + 2 * self.config.startup_timeout_s,
                owner_token=request.token,
            )
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
                    instance_id="",
                    adapter_label=request.adapter_label,
                    create_request_id=request_id,
                    creation_deadline=self.clock() + self.config.startup_timeout_s,
                    last_activity_at=self.clock(),
                    lora_config=request.payload.get("lora_config"),
                    user_metadata=request.payload.get("user_metadata") or {},
                )
                await self.store.reserve(training)
                instance = self._select(spec)
                if instance is None:
                    if not self.residents_ready_for(spec):
                        raise CapacityUnavailableError(
                            "Resident training capacity is recovering; retry a new create later."
                        )
                    matching = [item for item in self.instances.values() if (item.base_model, item.mode) == spec.key]
                    if not self.lifecycle.headroom(matching, spec.config.max_replicas):
                        raise CapacityUnavailableError("Training replica limit reached; retry a new create later.")
                    instance = await self._new_instance(spec)
                training.instance_id = instance.instance_id
                training.replica_id = instance.instance_id
                instance.training_ids.append(training_id)
                self.trainings[training_id] = training
                await self.store.save_training(training)
                await self.store.save_instance(instance)
                self.lifecycle.spawn(self._create_user(training, request))
            except Exception as exc:
                if training is not None:
                    self.trainings.pop(training_id, None)
                    if training.instance_id in self.instances:
                        instance = self.instances[training.instance_id]
                        if training_id in instance.training_ids:
                            instance.training_ids.remove(training_id)
                        await self.store.save_instance(instance)
                        if not instance.training_ids and not instance.resident:
                            self.lifecycle.spawn(self.fail_instance(instance.instance_id, failure_for(exc)))
                    training.state = "FAILED"
                    training.terminal_reason = failure_for(exc).reason_code
                    await self.store.retire(training)
                await self.store.future(request_id, "failed", training_id, failure=failure_for(exc))
            return {"request_id": request_id}

    async def _create_user(self, training: TrainingRecord, request: CreateTraining) -> None:
        instance = self.instances[training.instance_id]
        try:
            await asyncio.shield(self.lifecycle.startups[instance.instance_id])
            async with self._lock:
                future = await self.store.futures.get(training.create_request_id)
                if future and future.status == "cancelled":
                    training.state = "CLOSING"
                    training.terminal_reason = "cancelled"
                if training.state != "PENDING" or instance.state != "READY":
                    raise RequestRejectedError("Training creation is no longer pending")
                training.creation_deadline = self.clock() + self.config.startup_timeout_s
                await self.store.save_training(training)
                if not await self.store.futures.try_start(training.create_request_id):
                    future = await self.store.futures.get(training.create_request_id)
                    if future and future.status == "cancelled":
                        training.terminal_reason = "cancelled"
                    raise RequestRejectedError("Training creation did not acquire execution")
            remaining = training.creation_deadline - self.clock()
            await asyncio.wait_for(self.driver.create_training(instance, training, request), max(0, remaining))
            # Driver includes training-run metadata commit, before public success.
            async with self._lock:
                await self._assert_epoch()
                if self.clock() >= training.creation_deadline:
                    raise TimeoutError("Training initialization completed after its creation deadline")
                if training.state != "PENDING" or instance.state != "READY":
                    raise RuntimeError("Training became unavailable during initialization")
                training.state = "ACTIVE"
                training.last_activity_at = self.clock()
                await self.store.save_training(training)
                result = (
                    {"model_id": training.training_id}
                    if request.client_type == "tinker"
                    else {"training_id": training.training_id, "adapter_name": training.adapter_label}
                )
                await self.store.future(training.create_request_id, "completed", training.training_id, result=result)
        except Exception as exc:
            if training.state not in TERMINAL_TRAINING:
                await self.store.future(
                    training.create_request_id, "failed", training.training_id, failure=failure_for(exc)
                )
                async with self._lock:
                    training.state = "CLOSING"
                    training.terminal_reason = (
                        "cancelled" if training.terminal_reason == "cancelled" else failure_for(exc).reason_code
                    )
                    await self.store.save_training(training)
                await self._cleanup_user(training)

    async def _expired(self, training: TrainingRecord) -> bool:
        heartbeat = await self.store.sessions.get_last_heartbeat(training.session_id)
        return self._expired_with_heartbeat(training, heartbeat)

    def _session_expired(self, training: TrainingRecord, heartbeat: float | None) -> bool:
        timeout = self.config.get(training.base_model, training.mode).config.adapter_config.adapter_timeout
        return heartbeat is None or (timeout is not None and time.time() - heartbeat >= timeout)

    def _lifetime_expired(self, training: TrainingRecord) -> bool:
        lifetime = self.config.get(training.base_model, training.mode).config.adapter_config.adapter_max_lifetime
        if lifetime is None:
            return False
        created_at = datetime.fromisoformat(training.created_at).timestamp()
        return time.time() - created_at >= lifetime

    def _expired_with_heartbeat(self, training: TrainingRecord, heartbeat: float | None) -> bool:
        spec = self.config.get(training.base_model, training.mode)
        if self._session_expired(training, heartbeat) or self._lifetime_expired(training):
            return True
        return self.clock() - training.last_activity_at >= spec.config.idle_timeout_s

    async def _authorized(self, training_id: str, token: str, session_id: str | None) -> TrainingRecord:
        await self._assert_epoch()
        training = self.trainings.get(training_id)
        if (
            training is None
            or training.token != token
            or (session_id is not None and session_id != training.session_id)
        ):
            raise ResourceNotFoundError("Unknown training handle")
        if training.state == "EXPIRED":
            raise TrainingExpiredError("Training handle expired; create a new handle and explicitly load a checkpoint.")
        if training.state != "ACTIVE" or training.instance_id in self._failed_instances:
            raise ResourceNotFoundError("Training handle is no longer active")
        heartbeat = await self.store.sessions.get_last_heartbeat(training.session_id)
        if self._session_expired(training, heartbeat):
            raise TrainingExpiredError("Training session expired; create a new training handle")
        return training

    async def admit(
        self, training_id: str, token: str, session_id: str | None, request_id: str, *, renew: bool = True
    ) -> RuntimeOwner:
        async with self._lock:
            training = await self._authorized(training_id, token, session_id)
            expired = self._lifetime_expired(training) or (
                not training.pending_operation_ids and await self._expired(training)
            )
            if request_id not in training.pending_operation_ids and expired:
                training.state = "CLOSING"
                training.terminal_reason = "expired"
                await self.store.save_training(training)
                self.lifecycle.spawn(self._cleanup_user(training))
                raise TrainingExpiredError(
                    "Training handle expired; create a new handle and explicitly load a checkpoint."
                )
            instance = self.instances[training.instance_id]
            if instance.state != "READY":
                raise ResourceNotFoundError("Training runtime is unavailable")
            if request_id not in training.pending_operation_ids:
                training.pending_operation_ids[request_id] = {"accepted_at": self.clock(), "renew": renew}
                await self.store.save_training(training)
            return {"instance_id": instance.instance_id, "epoch": self.epoch}

    async def finish_operation(
        self, training_id: str, request_id: str, epoch: str, *, executed: bool, finished_at: float | None = None
    ) -> None:
        async with self._lock:
            try:
                await self._assert_epoch()
            except ResourceNotFoundError:
                return
            training = self.trainings.get(training_id)
            if epoch != self.epoch or training is None:
                return
            ticket = training.pending_operation_ids.get(request_id)
            if ticket is None:
                return
            updated = training.model_copy(deep=True)
            updated.pending_operation_ids.pop(request_id)
            if executed and ticket["renew"]:
                age = max(0, time.time() - finished_at) if finished_at is not None else 0
                updated.last_activity_at = max(training.last_activity_at, self.clock() - age)
            # A failed completion write must leave the original ticket retryable.
            await self.store.save_training(updated)
            training.pending_operation_ids = updated.pending_operation_ids
            training.last_activity_at = updated.last_activity_at
            cleanup = training.state == "CLOSING" and not training.pending_operation_ids
        if cleanup:
            self.lifecycle.spawn(self._cleanup_user(training))

    async def route(self, training_id: str, token: str, session_id: str | None = None) -> TrainingRoute:
        async with self._lock:
            training = await self._authorized(training_id, token, session_id)
            return {"instance_id": training.instance_id, "base_model": training.base_model, "epoch": self.epoch}

    async def release(self, training_id: str, token: str, session_id: str | None = None) -> None:
        async with self._lock:
            training = await self._authorized(training_id, token, session_id)
            training.state = "CLOSING"
            training.terminal_reason = "released"
            await self.store.save_training(training)
            cleanup = not training.pending_operation_ids
        if cleanup:
            await self._cleanup_user(training)
        config = self.config.get(training.base_model, training.mode).config.queue_config
        deadline = asyncio.get_running_loop().time() + config.absolute_future_ttl(1) + self.config.shutdown_timeout_s
        while training.state == "CLOSING" or self.instances[training.instance_id].state in ("DRAINING", "QUARANTINED"):
            if self.instances[training.instance_id].state == "QUARANTINED":
                raise RuntimeError("Runtime cleanup is quarantined; device release is not confirmed")
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Training cleanup is still pending; its slot remains reserved")
            await asyncio.sleep(config.inline_fast_path_timeout)

    async def _cleanup_user(self, training: TrainingRecord) -> None:
        instance = self.instances[training.instance_id]
        claimed = False
        resetting = False
        closing_full = False
        try:
            # Each training must have exactly one cleanup, including concurrent finish callbacks.
            async with self._lock:
                if (
                    instance.state not in ("READY", "RESETTING")
                    or training.state != "CLOSING"
                    or training.pending_operation_ids
                    or training.training_id in self._cleaning
                ):
                    return
                self._cleaning.add(training.training_id)
                claimed = True
                resetting = instance.resident and instance.mode == "full" and not instance.dirty
                closing_full = instance.mode == "full" and (not instance.resident or instance.dirty)
                if resetting:
                    updated_instance = instance.model_copy(update={"state": "RESETTING"})
                    await self.store.save_instance(updated_instance)
                    instance.state = "RESETTING"
                if closing_full:
                    updated_instance = instance.model_copy(update={"state": "DRAINING"})
                    await self.store.save_instance(updated_instance)
                    instance.state = "DRAINING"
                if resetting or closing_full:
                    self._report_instance_metrics()
            if closing_full:
                await self._close_instance(instance)
                return
            try:
                spec = self.config.get(training.base_model, training.mode)
                await asyncio.wait_for(
                    self.driver.release_training(instance, training),
                    spec.config.queue_config.absolute_future_ttl(spec.config.num_devices),
                )
            except Exception as exc:
                # A timed-out create/cleanup RPC does not prove that shared workers failed.
                # Keep this slot reserved until serial cleanup can be confirmed.
                if instance.mode == "lora" and any(
                    self.trainings[identity].state in ("ACTIVE", "PENDING")
                    or self.trainings[identity].pending_operation_ids
                    for identity in instance.training_ids
                    if identity != training.training_id
                ):
                    logger.warning("Training cleanup remains pending for %s: %s", training.training_id, exc)
                else:
                    await self.fail_instance(
                        instance.instance_id,
                        FutureFailureRecord(
                            reason_code="internal_error",
                            attribution="server",
                            message=f"Training cleanup failed: {exc}"[:1024],
                            diagnostic=traceback.format_exc(),
                        ),
                    )
                return
            async with self._lock:
                # A health failure may have taken ownership while release/reset
                # was in flight. Its late ACK cannot retire quota or revive READY.
                if (
                    instance.instance_id in self._failed_instances
                    or instance.state not in ("READY", "RESETTING")
                    or training.state != "CLOSING"
                ):
                    return
                terminal = training.model_copy(
                    deep=True,
                    update={
                        "state": {"expired": "EXPIRED", "released": "RELEASED", "cancelled": "CANCELLED"}.get(
                            training.terminal_reason, "FAILED"
                        ),
                        "terminal_at": time.time(),
                    },
                )
                await self.store.retire(terminal)
                remaining = [identity for identity in instance.training_ids if identity != training.training_id]
                close = not remaining and (not instance.resident or instance.dirty)
                updated_instance = instance.model_copy(
                    deep=True,
                    update={
                        "training_ids": remaining,
                        "state": "DRAINING" if close else ("READY" if resetting else instance.state),
                    },
                )
                await self.store.save_instance(updated_instance)
                training.state = terminal.state
                training.terminal_at = terminal.terminal_at
                instance.training_ids = remaining
                instance.state = updated_instance.state
                self._report_instance_metrics()
            if close:
                await self._close_instance(instance)
        finally:
            if claimed:
                self._cleaning.discard(training.training_id)

    async def _close_instance(self, instance: InstanceRecord) -> None:
        started = time.monotonic()
        tags = self._instance_metric_tags(instance)
        timing_tags = {"backend": tags["backend"], "mode": tags["mode"]}

        async def quarantined(exc):
            async with self._lock:
                entered_quarantine = instance.state != "QUARANTINED"
                quarantined = instance.model_copy(update={"state": "QUARANTINED", "failure": str(exc)})
                await self.store.save_instance(quarantined)
                instance.state = "QUARANTINED"
                instance.failure = str(exc)
                self._report_instance_metrics()
                tags = self._instance_metric_tags(instance)
                if entered_quarantine:
                    self._metrics.quarantined_total.inc(
                        tags={
                            "backend": tags["backend"],
                            "mode": tags["mode"],
                            "reason": TrainingInstanceMetrics.quarantine_reason(exc),
                        }
                    )
            self._metrics.shutdown_seconds.observe(time.monotonic() - started, tags={**timing_tags, "status": "error"})

        async def stopped():
            self._metrics.shutdown_seconds.observe(
                time.monotonic() - started, tags={**timing_tags, "status": "success"}
            )
            await self._retire_instance(instance)

        await self.lifecycle.close(
            instance.instance_id,
            lambda: self.driver.shutdown(instance),
            self.config.shutdown_timeout_s,
            stopped,
            quarantined,
        )

    async def _retire_instance(self, instance):
        async with self._lock:
            terminal_records = []
            terminal_at = time.time()
            failure = self._failed_instances.get(instance.instance_id)
            for training_id in list(instance.training_ids):
                training = self.trainings[training_id]
                if failure is not None:
                    for request_id in (training.create_request_id, *training.pending_operation_ids):
                        await self.store.future(request_id, "failed", training_id, failure=failure)
                terminal = training.model_copy(
                    deep=True,
                    update={
                        "pending_operation_ids": {},
                        "state": {"expired": "EXPIRED", "released": "RELEASED", "cancelled": "CANCELLED"}.get(
                            training.terminal_reason, "FAILED"
                        ),
                        "terminal_at": terminal_at,
                    },
                )
                await self.store.retire(terminal)
                terminal_records.append((training, terminal))
            stopped = instance.model_copy(
                deep=True, update={"state": "STOPPED", "stopped_at": terminal_at, "training_ids": []}
            )
            await self.store.save_instance(stopped)
            for training, terminal in terminal_records:
                training.pending_operation_ids.clear()
                training.state = terminal.state
                training.terminal_at = terminal.terminal_at
            instance.state = "STOPPED"
            instance.stopped_at = terminal_at
            instance.training_ids.clear()
            self._failed_instances.pop(instance.instance_id, None)
            self._report_instance_metrics()

    async def fail_instance(self, instance_id: str, failure: FutureFailureRecord) -> None:
        async with self._lock:
            instance = self.instances[instance_id]
            if instance.state == "STOPPED":
                self._failed_instances.pop(instance_id, None)
                return
            # Record the retry intent without promoting unpersisted state; block new routing.
            self._failed_instances[instance_id] = failure
            if instance.state not in ("DRAINING", "QUARANTINED"):
                draining = instance.model_copy(update={"state": "DRAINING", "failure": failure.message})
                await self.store.save_instance(draining)
                instance.state = "DRAINING"
                instance.failure = failure.message
                self._report_instance_metrics()
        try:
            async with self._lock:
                for training_id in instance.training_ids:
                    training = self.trainings[training_id]
                    closing = training.model_copy(
                        deep=True, update={"state": "CLOSING", "terminal_reason": failure.reason_code}
                    )
                    await self.store.save_training(closing)
                    training.state = "CLOSING"
                    training.terminal_reason = failure.reason_code
                    for request_id in (training.create_request_id, *training.pending_operation_ids):
                        await self.store.future(request_id, "failed", training_id, failure=failure)
            await self.lifecycle.cancel_start(instance_id)
        finally:
            # Once DRAINING is durable, shutdown must be attempted even if a
            # later training/future write fails. Sweep will retry if this fails.
            await self._close_instance(instance)

    async def sweep(self) -> None:
        # Batch I/O outside the decision lock; re-verify state after re-acquiring, ignore handles added since snapshot.
        session_ids = list(
            {
                training.session_id
                for training in self.trainings.values()
                if training.state == "ACTIVE" and not training.pending_operation_ids
            }
        )
        heartbeats = await self.store.sessions.get_last_heartbeats(session_ids)
        async with self._lock:
            await self._assert_epoch()
            for training in self.trainings.values():
                if (
                    training.state != "ACTIVE"
                    or training.pending_operation_ids
                    or training.session_id not in heartbeats
                ):
                    continue
                if self._expired_with_heartbeat(training, heartbeats[training.session_id]):
                    closing_record = training.model_copy(update={"state": "CLOSING", "terminal_reason": "expired"})
                    await self.store.save_training(closing_record)
                    training.state = "CLOSING"
                    training.terminal_reason = "expired"
            uncertain = [
                (training, request_id)
                for training in self.trainings.values()
                for request_id, ticket in training.pending_operation_ids.items()
                if request_id not in self._reconciling
                and self.clock() - ticket["accepted_at"] >= self.config.sweep_interval_s
            ]
            self._reconciling.update(request_id for _, request_id in uncertain)
            closing = [
                training
                for training in self.trainings.values()
                if training.state == "CLOSING"
                and not training.pending_operation_ids
                and training.training_id not in self._cleaning
            ]
        for training in closing:
            self.lifecycle.spawn(self._cleanup_user(training))
        for training, request_id in uncertain:
            self.lifecycle.spawn(self._reconcile_operation(training, request_id))
        for instance in list(self.instances.values()):
            failure = self._failed_instances.get(instance.instance_id)
            if failure is not None:
                self.lifecycle.spawn(self.fail_instance(instance.instance_id, failure))
            elif instance.state in ("DRAINING", "QUARANTINED"):
                self.lifecycle.spawn(self._close_instance(instance))
        # Rebuilding one specification must not stop expiry or other models.
        for spec in self.config.models:
            async with self._lock:
                matching = [item for item in self.instances.values() if (item.base_model, item.mode) == spec.key]
                missing = self.lifecycle.missing(
                    matching,
                    spec.config.min_replicas,
                    spec.config.max_replicas,
                    lambda instance: instance.resident
                    and instance.instance_id not in self._failed_instances
                    and instance.state in LIVE_INSTANCES,
                )
                for _ in range(missing):
                    await self._new_instance(spec, resident=True)
        await self.store.cleanup_receipts()
        # Terminal diagnostics expire like their backend tombstones. Repeated
        # model creation must not grow controller memory for the server lifetime.
        async with self._lock:
            cutoff = time.time() - self.store.expiration_timeout
            for identity, training in list(self.trainings.items()):
                if training.state in TERMINAL_TRAINING and training.terminal_at is not None:
                    if training.terminal_at <= cutoff:
                        self.trainings.pop(identity)
            for identity, instance in list(self.instances.items()):
                if (
                    instance.state == "STOPPED"
                    and instance.stopped_at is not None
                    and instance.stopped_at <= cutoff
                    and not instance.training_ids
                ):
                    self.instances.pop(identity)
                    self.lifecycle.startups.pop(identity, None)
            self._report_instance_metrics()

    async def _reconcile_operation(self, training: TrainingRecord, request_id: str) -> None:
        try:
            instance = self.instances[training.instance_id]
            if instance.state != "READY":
                return
            result = await asyncio.wait_for(
                self.driver.verify_operation(instance, request_id), self.config.shutdown_timeout_s
            )
            if result["state"] == "finished":
                await self.finish_operation(
                    training.training_id,
                    request_id,
                    self.epoch,
                    executed=result["executed"],
                    finished_at=result.get("finished_at"),
                )
            elif result["state"] == "missing":
                await self.store.future(
                    request_id,
                    "failed",
                    training.training_id,
                    failure=FutureFailureRecord(
                        reason_code="internal_error",
                        attribution="server",
                        message="Operation was not committed to the runtime queue",
                    ),
                )
                await self.finish_operation(training.training_id, request_id, self.epoch, executed=False)
            # An unsafe/unreachable runtime is handled by the worker health monitor.
        except Exception:
            logger.exception("Operation reconciliation deferred for %s", request_id)
        finally:
            self._reconciling.discard(request_id)

    async def wait_for_jobs(self) -> None:
        await self.lifecycle.join()

    async def shutdown(self) -> None:
        self._ready = False
        outcomes = await asyncio.gather(
            *(
                self.fail_instance(
                    instance.instance_id,
                    FutureFailureRecord(
                        reason_code="orphaned_replica",
                        message="Training service is shutting down",
                        attribution="server",
                    ),
                )
                for instance in list(self.instances.values())
            ),
            return_exceptions=True,
        )
        await self.wait_for_jobs()
        errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        for error in errors:
            logger.error("Training instance shutdown failed", exc_info=(type(error), error, error.__traceback__))
        if errors:
            raise errors[0]
        self.lifecycle.require_released(self.instances.values())
