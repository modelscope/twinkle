# Copyright (c) ModelScope Contributors. All rights reserved.
"""Sampling pool decisions; generation ownership outlives HTTP and future timeouts."""

from __future__ import annotations
import asyncio
import hashlib
import logging
import math
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Literal

import json

if TYPE_CHECKING:
    from twinkle.server.lifecycle.proxy import ForwardedReply, ForwardedRequest

from twinkle.server.config.application_spec import SamplerArgs
from twinkle.server.config.resources import ResourceOptions
from twinkle.server.exceptions import ResourceExhaustedError
from twinkle.server.resources.instances import InstanceLifecycle
from twinkle.server.sampler.concurrency import SamplerConcurrency

logger = logging.getLogger(__name__)
_CONTROL_ENDPOINTS = frozenset(
    {"twinkle/set_template", "twinkle/add_adapter_to_sampler", "twinkle/apply_patch", "twinkle/unload_adapter_paths"}
)


def sampler_controller_name(namespace, pool_id):
    return f"sampler-controller-{namespace}-{pool_id}"


def sampling_pool_id(route):
    return hashlib.sha256(route.encode()).hexdigest()[:16]


@dataclass
class SamplingInstance:
    identity: str
    epoch: str
    state: Literal["STARTING", "READY", "DRAINING", "QUARANTINED", "STOPPED"] = "STARTING"
    application: str | None = None
    devices: list[str] = field(default_factory=list)
    requests: set[str] = field(default_factory=set)
    last_finished: float = 0
    failure: str | None = None


class SamplerController:
    def __init__(self, args: SamplerArgs, driver, budgets: dict, *, clock=time.monotonic):
        self.args, self.driver, self.clock = args, driver, clock
        self.budgets = ResourceOptions.model_validate(budgets)
        self.epoch = uuid.uuid4().hex
        self.instances = {}
        self.lifecycle = InstanceLifecycle("Sampling")
        self._actions = []
        self._base_path = None
        self._lock = asyncio.Lock()
        self._config_lock = asyncio.Lock()
        self._guard = SamplerConcurrency()
        self._ready = False
        self._stopping = False
        self._initialization = None
        self._monitor_task = None

    def _owned_count(self):
        return self.lifecycle.owned_count(self.instances.values())

    def _ready_instances(self):
        return [instance for instance in self.instances.values() if instance.state == "READY"]

    def _new_instance(self):
        if not self.lifecycle.headroom(self.instances.values(), self.args.max_replicas):
            raise ResourceExhaustedError("Sampling replica limit reached")
        instance = SamplingInstance("sampler_" + uuid.uuid4().hex, self.epoch, last_finished=self.clock())
        self.instances[instance.identity] = instance
        self.lifecycle.start(instance.identity, lambda: self._start(instance))
        return instance

    async def initialize(self):
        if self._initialization is None:
            self._initialization = asyncio.create_task(self._initialize())
        await asyncio.shield(self._initialization)

    async def _initialize(self):
        await self.driver.recover_orphans()
        try:
            for _ in range(self.args.min_replicas):
                instance = self._new_instance()
                await asyncio.shield(self.lifecycle.startups[instance.identity])
            self._ready = True
        except Exception:
            self._stopping = True
            await asyncio.gather(*(self._close(instance) for instance in list(self.instances.values())))
            raise

    async def _start(self, instance):
        async def startup():
            await self.driver.start(instance)
            async with self._config_lock:
                if self._stopping or instance.state != "STARTING":
                    raise ResourceExhaustedError("Sampling startup no longer owns this instance")
                for action in self._actions:
                    if "base_path" in action:
                        await self.driver.load_weights(instance, action["base_path"])
                    else:
                        reply = await self._invoke_finished(instance, action)
                        if reply["status"] >= 400:
                            raise RuntimeError("Sampler control replay failed")
                instance.state = "READY"
                instance.last_finished = self.clock()

        try:
            # Replaying control state is part of readiness, inside the same budget.
            await asyncio.wait_for(startup(), self.budgets.startup_timeout_s)
        except BaseException:
            await self._close(instance)
            raise

    async def _select(self):
        while True:
            async with self._lock:
                if self._stopping or not self._ready:
                    raise ResourceExhaustedError("Sampling pool is not ready")
                ready = self._ready_instances()
                active = sum(len(instance.requests) for instance in self.instances.values())
                desired = min(
                    self.args.max_replicas,
                    max(self.args.min_replicas, math.ceil((active + 1) / self.args.target_ongoing_requests)),
                )
                if self._owned_count() < desired:
                    self._new_instance()
                if ready:
                    instance = min(ready, key=lambda item: (len(item.requests), item.identity))
                    ticket = uuid.uuid4().hex
                    instance.requests.add(ticket)
                    return instance, ticket
                pending = [task for task in self.lifecycle.startups.values() if not task.done()]
                if not pending:
                    raise ResourceExhaustedError("No available sampler instance; device allocation was not confirmed")
            await asyncio.wait_for(asyncio.shield(pending[0]), self.budgets.startup_timeout_s)

    async def _invoke_finished(self, instance, payload):
        ticket = uuid.uuid4().hex
        reply = await self.driver.invoke(instance, ticket, payload)
        await self.driver.wait_operation(instance, ticket)
        return reply

    @staticmethod
    def _action_key(action):
        if "base_path" in action:
            return ("base_path",)
        endpoint = action["endpoint"]
        if endpoint == "twinkle/set_template":
            return (endpoint,)
        if endpoint == "twinkle/add_adapter_to_sampler":
            # Adapter names are scoped by the caller's token in the handler.
            headers = {key.lower(): value for key, value in action["headers"]}
            owner = headers.get(b"twinkle-authorization") or headers.get(b"authorization")
            return endpoint, owner, json.loads(action["body"])["adapter_name"]
        return None

    def _remember(self, action):
        """Compact overwritten state only within the segment after the last opaque patch."""
        key = self._action_key(action)
        if key is not None:
            for index in range(len(self._actions) - 1, -1, -1):
                previous = self._actions[index]
                if previous.get("endpoint") == "twinkle/apply_patch":
                    break
                if self._action_key(previous) == key:
                    del self._actions[index]
        self._actions.append(deepcopy(action))

    async def _release(self, instance, ticket):
        async with self._lock:
            # Unknown completion retains ownership until shutdown is confirmed.
            if instance.state not in ("DRAINING", "QUARANTINED"):
                instance.requests.discard(ticket)
                instance.last_finished = self.clock()

    async def _fence(self, replicas):
        async with self._lock:
            for replica in replicas:
                if replica.state != "STOPPED":
                    replica.state = "DRAINING"
        await asyncio.gather(*(self._close(replica) for replica in replicas))

    async def _mutate(self, payload=None, full_path=None):
        # Lock order: config -> state. Never wait for startup while holding config,
        # because startup also replays the committed configuration under that lock.
        seed, seed_ticket = await self._select()
        replicas = []
        ticket = uuid.uuid4().hex
        try:
            async with self._config_lock:
                async with self._lock:
                    if self._stopping:
                        raise ResourceExhaustedError("Sampling pool is stopping")
                    replicas = self._ready_instances()
                    if not replicas:
                        raise ResourceExhaustedError("No ready sampler for control update")
                    for replica in replicas:
                        replica.requests.add(ticket)

                async def broadcast():
                    response = None
                    for replica in replicas:
                        if full_path is not None:
                            await self.driver.load_weights(replica, full_path)
                        else:
                            response = await self._invoke_finished(replica, payload)
                            if response["status"] >= 400:
                                return response
                    return response

                try:
                    response = await asyncio.wait_for(broadcast(), self.args.queue_config.effective_execution_timeout)
                    if response is not None and response["status"] >= 400:
                        await self._fence(replicas)
                        return response
                    # A health probe or shutdown can invalidate a participant while
                    # its RPC is pending. Such an ACK must not commit pool state.
                    if any(replica.state != "READY" for replica in replicas):
                        raise ResourceExhaustedError("Sampler changed state during control update")
                    if full_path is not None:
                        self._base_path = full_path
                        self._remember({"base_path": full_path})
                    elif payload["endpoint"] != "twinkle/unload_adapter_paths":
                        self._remember(payload)
                    return response
                except BaseException:
                    await self._fence(replicas)
                    raise
        finally:
            for replica in replicas:
                await self._release(replica, ticket)
            await self._release(seed, seed_ticket)

    async def request(self, payload: ForwardedRequest, stream_queue=None, full_path=None) -> ForwardedReply:
        await self.initialize()
        endpoint = payload["endpoint"]
        if endpoint == "twinkle/create":
            return {"status": 200, "headers": [(b"content-type", b"application/json")], "body": b'{"status":"ok"}'}
        control = endpoint in _CONTROL_ENDPOINTS
        reply_ready = asyncio.get_running_loop().create_future()

        async def operation():
            instance = None
            ticket = None
            try:
                if control:
                    reply_ready.set_result(await self._mutate(payload=payload))
                    return
                if full_path is not None and full_path != self._base_path:
                    await self._mutate(full_path=full_path)
                instance, ticket = await self._select()
                response = await self.driver.invoke(instance, ticket, payload, stream_queue)
                if not reply_ready.done():
                    reply_ready.set_result(response)
                # A pending future reply is not evidence that the generation stopped.
                await asyncio.wait_for(
                    self.driver.wait_operation(instance, ticket), self.args.queue_config.absolute_future_ttl(1)
                )
            except BaseException as error:
                if not reply_ready.done():
                    reply_ready.set_exception(error)
                if instance is not None:
                    await self._close(instance)
                raise
            finally:
                if instance is not None and ticket is not None:
                    await self._release(instance, ticket)

        async def guarded():
            try:
                await self._guard.run(
                    operation,
                    exclusive=control or full_path is not None,
                    queue_timeout=self.args.queue_config.queue_timeout,
                )
            except BaseException as error:
                if not reply_ready.done():
                    reply_ready.set_exception(error)
                raise

        self.lifecycle.spawn(guarded())
        return await asyncio.shield(reply_ready)

    async def _close(self, instance):
        if instance.state == "STOPPED":
            return
        instance.state = "DRAINING"

        async def quarantined(error):
            instance.state = "QUARANTINED"
            instance.failure = str(error)

        async def stopped():
            instance.state = "STOPPED"
            instance.requests.clear()

        await self.lifecycle.close(
            instance.identity,
            lambda: self.driver.shutdown(instance),
            self.budgets.shutdown_timeout_s,
            stopped,
            quarantined,
        )

    async def sweep(self):
        async with self._lock:
            ready = sorted(self._ready_instances(), key=lambda instance: instance.last_finished)
            removable = max(0, len(ready) - self.args.min_replicas)
            victims = [
                instance
                for instance in ready
                if not instance.requests and self.clock() - instance.last_finished >= self.args.idle_timeout_s
            ][:removable]
            for instance in victims:
                instance.state = "DRAINING"
            missing = self.lifecycle.missing(
                self.instances.values(),
                self.args.min_replicas,
                self.args.max_replicas,
                lambda instance: instance.state in ("READY", "STARTING"),
            )
            if not self._stopping:
                for _ in range(missing):
                    self._new_instance()
        for instance in list(self.instances.values()):
            if instance.state in ("DRAINING", "QUARANTINED"):
                self.lifecycle.spawn(self._close(instance))
            elif instance.state == "STOPPED" and not instance.requests:
                self.instances.pop(instance.identity, None)
                self.lifecycle.startups.pop(instance.identity, None)

    async def _monitor(self):
        async def health(instance):
            healthy = await self.lifecycle.health(instance.identity, lambda: self.driver.health(instance))
            if healthy is False:
                await self._close(instance)

        while not self._stopping:
            try:
                for instance in self._ready_instances():
                    self.lifecycle.spawn_once(("health", instance.identity), lambda item=instance: health(item))
                await self.sweep()
            except Exception:
                logger.exception("Sampler monitor failed; ownership remains reserved")
            await asyncio.sleep(self.budgets.sweep_interval_s)

    async def diagnostics(self):
        await self.initialize()
        return {
            "ready": self._ready and not self._stopping and len(self._ready_instances()) >= self.args.min_replicas,
            "epoch": self.epoch,
            "instances": [
                {**asdict(instance), "requests": len(instance.requests)} for instance in self.instances.values()
            ],
        }

    async def shutdown(self):
        self._stopping = True
        self._ready = False
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)
        for task in list(self.lifecycle.startups.values()):
            if not task.done():
                task.cancel()
        await asyncio.gather(*tuple(self.lifecycle.startups.values()), return_exceptions=True)
        await asyncio.gather(*(self._close(instance) for instance in list(self.instances.values())))
        self._guard.close()
        await self.lifecycle.join()
        self.lifecycle.require_released(self.instances.values())
