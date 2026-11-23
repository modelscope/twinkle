# Copyright (c) ModelScope Contributors. All rights reserved.
"""Stable CPU endpoint for one catalog model; requests route by training ID."""

from __future__ import annotations
import asyncio
import uuid
from types import SimpleNamespace

import json
import ray
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from tinker import types as tinker

from twinkle.protocol.types import model as native
from twinkle.server.checkpoint import _resolve_client_save_dir, validate_user_path
from twinkle.server.deployment import LazyCleanupMixin, bind_deployment, build_deployment_app
from twinkle.server.exceptions import ProtocolMismatchError, RequestRejectedError, ResourceNotFoundError
from twinkle.server.lifecycle.envelope import envelope_from_record
from twinkle.server.lifecycle.proxy import ServiceProxy
from twinkle.server.middleware.auth import get_session_id_from_request, get_token_from_request
from twinkle.server.state import get_server_state
from twinkle.server.state.training_records import CreateTraining
from twinkle.server.task_queue import TaskQueueMixin
from twinkle.server.validation import assert_request_supported
from .actor import controller_name
from .hub_upload import submit_upload

NATIVE_REQUESTS = {
    "forward": native.ForwardRequest,
    "forward_only": native.ForwardOnlyRequest,
    "forward_backward": native.ForwardBackwardTaskRequest,
    "calculate_loss": native.AdapterRequest,
    "backward": native.AdapterRequest,
    "forward_from_data_plane": native.DataPlaneForwardRequest,
    "forward_backward_from_data_plane": native.DataPlaneForwardRequest,
    "forward_only_from_data_plane": native.DataPlaneForwardOnlyRequest,
    "step": native.StepRequest,
    "zero_grad": native.AdapterRequest,
    "lr_step": native.LrStepRequest,
    "clip_grad_norm": native.ClipGradNormRequest,
    "clip_grad_and_step": native.ClipGradAndStepRequest,
    "get_train_configs": native.AdapterRequest,
    "set_loss": native.SetLossRequest,
    "set_optimizer": native.SetOptimizerRequest,
    "set_lr_scheduler": native.SetLrSchedulerRequest,
    "set_template": native.SetTemplateRequest,
    "set_processor": native.SetProcessorRequest,
    "add_metric": native.AddMetricRequest,
    "apply_patch": native.ApplyPatchRequest,
    "calculate_metric": native.CalculateMetricRequest,
    "save": native.SaveRequest,
    "load": native.LoadRequest,
    "resume_from_checkpoint": native.ResumeFromCheckpointRequest,
    "remove_adapter": native.AdapterRequest,
    "stop_training": native.AdapterRequest,
    "add_adapter_to_model": native.AddAdapterRequest,
    "upload_to_hub": native.UploadToHubRequest,
    "create": native.CreateRequest,
}
TINKER_REQUESTS = {
    "create_model": tinker.CreateModelRequest,
    "get_info": tinker.GetInfoRequest,
    "unload_model": tinker.UnloadModelRequest,
    "forward": tinker.ForwardRequest,
    "forward_backward": tinker.ForwardBackwardRequest,
    "optim_step": tinker.OptimStepRequest,
    "save_weights": tinker.SaveWeightsRequest,
    "load_weights": tinker.LoadWeightsRequest,
    "save_weights_for_sampler": tinker.SaveWeightsForSamplerRequest,
}


class TrainingRouter(TaskQueueMixin, LazyCleanupMixin):
    # __init__ must stay synchronous: Ray Serve's FastAPI ingress wrapper calls the user
    # constructor synchronously (ASGIIngressWrapper in ray/serve/api.py) -- an async
    # __init__'s coroutine is never awaited, so none of its body runs and forwarding dies
    # on a missing attribute. Replica registration and the heartbeat task (needs a
    # running loop) move to _complete_startup, completed once from check_health -- the
    # first instance method Serve awaits; a failure there still fails the deployment
    # loudly, matching the old async-init semantics.
    def __init__(self, config, model, namespace, http_options):
        self.config = config
        self.model = model
        self.backend = model.backend
        self.state = get_server_state()
        self.store = self.state.create_training_store(namespace)
        self.controller = ray.get_actor(controller_name(namespace), namespace=namespace)
        self.proxy = ServiceProxy(http_options)
        self.replica_id = "training-router_" + uuid.uuid4().hex
        self._init_task_queue(None, deployment_name="TrainingRouter")
        self._startup_async_done = False

    async def _complete_startup(self) -> None:
        await self.state.register_replica(self.replica_id, 0)
        self._heartbeat = asyncio.create_task(self._keep_alive())

    async def check_health(self) -> None:
        if not self._startup_async_done:
            self._startup_async_done = True
            await self._complete_startup()

    async def _keep_alive(self):
        while True:
            await self.state.touch_replica_last_seen(self.replica_id)
            await asyncio.sleep(5)

    async def shutdown(self):
        # Startup may have failed before registration/heartbeat existed; only
        # unregister what was registered, and do not mask the startup error.
        heartbeat = getattr(self, '_heartbeat', None)
        if heartbeat is not None:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            await self.state.unregister_replica(self.replica_id)
        await self.shutdown_task_queue()
        await self.proxy.close()

    async def _check_session(self, session):
        if not session or await self.state.get_session_last_heartbeat(session) is None:
            raise ResourceNotFoundError("Unknown or expired session")

    async def native_request(self, request, endpoint):
        schema = NATIVE_REQUESTS.get(endpoint)
        if schema is None:
            raise ResourceNotFoundError("Unknown training endpoint")
        try:
            body = schema.model_validate(await request.json())
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        token = get_token_from_request(request)
        if endpoint == "create":
            request.state.model_id = self.model.name
            if not body.training_handles:
                raise ProtocolMismatchError("Upgrade Twinkle SDK: training_handles handshake is required")
            return native.CreateResponse(training_handles=True)
        if endpoint == "add_adapter_to_model":
            request.state.model_id = self.model.name
            if not body.client_create_id:
                raise ProtocolMismatchError("Upgrade Twinkle SDK: client_create_id is required")
            if not body.adapter_name or not validate_user_path(token, body.adapter_name):
                raise RequestRejectedError("Invalid adapter_name")
            try:
                config = json.loads(body.config) if body.config is not None else None
            except (ValueError, TypeError) as exc:
                raise RequestRejectedError("Expected a serialized LoRA config") from exc
            mode = "full" if config is None else "lora"
            spec = self.config.get(self.model.name, mode)
            if config is not None:
                rank = config.get("r") if isinstance(config, dict) else None
                if (
                    not isinstance(config, dict)
                    or config.get("_TWINKLE_TYPE_") != "LoraConfig"
                    or type(rank) is not int
                    or not 0 < rank <= spec.config.max_r
                ):
                    raise RequestRejectedError("LoRA config/rank is not supported by this training specification")
            payload = body.model_dump(mode="json")
            if body.save_dir:
                try:
                    payload["save_dir"] = _resolve_client_save_dir(body.save_dir).as_posix()
                except ValueError as exc:
                    raise RequestRejectedError(str(exc)) from exc
            session = get_session_id_from_request(request)
            await self._check_session(session)
            ref = await self.controller.create.remote(
                CreateTraining(
                    base_model=self.model.name,
                    mode=mode,
                    token=token,
                    session_id=session,
                    create_key=body.client_create_id,
                    client_type="twinkle",
                    adapter_label=body.adapter_name,
                    payload=payload,
                ).model_dump()
            )
            return envelope_from_record(ref["request_id"], await self.state.get_future(ref["request_id"]))
        if endpoint == "upload_to_hub":
            return await submit_upload(self, token, body)
        if endpoint == "stop_training":
            # Owner-side explicit stop by training id: the webui kills the local driver, so
            # the driver's session dies with it and cannot release the training itself.
            # The token that owns the run releases it WITHOUT a session -- the same
            # authority level as the gateway's checkpoint deletion (delete-by-id with the
            # owner token). A training whose session heartbeat already expired reports
            # "unknown/expired handle" here, which is idempotent: the reaper released it.
            return await self._release(body.training_id, token, None, "twinkle")
        session = get_session_id_from_request(request)
        await self._check_session(session)
        if endpoint == "remove_adapter":
            return await self._release(body.training_id, token, session, "twinkle")
        capability = endpoint if endpoint in ("forward", "backward", "calculate_loss") else None
        assert_request_supported(SimpleNamespace(backend=self.backend), body, capability=capability)
        route = await self.controller.route.remote(body.training_id, token, session)
        if route["base_model"] != self.model.name:
            raise ResourceNotFoundError("Training handle belongs to another model endpoint")
        request.state.model_id = body.training_id
        return await self.proxy.proxy_request(request, f"twinkle/{endpoint}", route["instance_id"], "training-runtime")

    async def tinker_request(self, request, endpoint):
        schema = TINKER_REQUESTS.get(endpoint)
        if schema is None:
            raise ResourceNotFoundError("Unknown training endpoint")
        try:
            body = schema.model_validate(await request.json())
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc
        token = get_token_from_request(request)
        if endpoint == "create_model":
            request.state.model_id = self.model.name
            if body.base_model != self.model.name:
                raise RequestRejectedError("Model does not match the catalog endpoint")
            mode = "lora" if body.lora_config is not None else "full"
            spec = self.config.get(self.model.name, mode)
            if body.lora_config is not None and not 0 < body.lora_config.rank <= spec.config.max_r:
                raise RequestRejectedError("LoRA rank exceeds the training specification")
            await self._check_session(body.session_id)
            return await self.controller.create.remote(
                CreateTraining(
                    base_model=self.model.name,
                    mode=mode,
                    token=token,
                    session_id=body.session_id,
                    create_key=str(body.model_seq_id),
                    client_type="tinker",
                    payload=body.model_dump(mode="json"),
                ).model_dump()
            )
        if endpoint == "unload_model":
            return await self._release(body.model_id, token, None, "tinker")
        route = await self.controller.route.remote(body.model_id, token)
        if route["base_model"] != self.model.name:
            raise ResourceNotFoundError("Training handle belongs to another model endpoint")
        request.state.model_id = body.model_id
        return await self.proxy.proxy_request(request, f"tinker/{endpoint}", route["instance_id"], "training-runtime")

    async def _release(self, training_id, token, session, client_type):
        route = await self.controller.route.remote(training_id, token, session, for_release=True)
        if route["base_model"] != self.model.name:
            raise ResourceNotFoundError("Training handle belongs to another model endpoint")

        async def release():
            await self.controller.release.remote(training_id, token, session)
            return {"model_id": training_id} if client_type == "tinker" else None

        ref = await self.schedule_background_task(release, task_type="release_training", token=token)
        if client_type == "tinker":
            return {"request_id": ref["request_id"]}
        return envelope_from_record(ref["request_id"], await self.state.get_future(ref["request_id"]))


def build_training_router(config, model, namespace, http_options, deploy_options):
    def routes(app: FastAPI, get_self):
        @app.get("/healthz")
        async def health(self=Depends(get_self)):
            diagnostic = await self.controller.diagnostics.remote(self.model.name)
            active = [item for item in diagnostic["instances"] if item["state"] != "STOPPED"]
            return {
                "healthy": diagnostic["ready"] and not any(item["state"] == "QUARANTINED" for item in active),
                "status": "unloaded" if not active else "loaded",
                "instances": active,
            }

        @app.post("/twinkle/{endpoint}")
        async def native_route(request: Request, endpoint: str, self=Depends(get_self)):
            return await self.native_request(request, endpoint)

        @app.post("/tinker/{endpoint}")
        async def tinker_route(request: Request, endpoint: str, self=Depends(get_self)):
            return await self.tinker_request(request, endpoint)

    async def shutdown(servable):
        await servable.shutdown()

    app = build_deployment_app("TrainingRouter", routes, on_shutdown=shutdown)
    return bind_deployment(
        app,
        TrainingRouter,
        deploy_options,
        deployment_name="TrainingRouter",
        bind_args=(config, model, namespace, http_options),
    )
