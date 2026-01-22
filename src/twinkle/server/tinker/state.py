# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import re
import time
import uuid
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional

import ray
from tinker import types


class ServerState:
    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.models: Dict[str, Dict[str, Any]] = {}
        self.sampling_sessions: Dict[str, Dict[str, Any]] = {}
        self.futures: Dict[str, Dict[str, Any]] = {}

    def create_session(self, payload: Dict[str, Any]) -> str:
        session_id = payload.get("session_id") or f"session_{uuid.uuid4().hex}"
        self.sessions[session_id] = {
            "tags": list(payload.get("tags") or []),
            "user_metadata": payload.get("user_metadata") or {},
            "sdk_version": payload.get("sdk_version"),
            "created_at": datetime.now().isoformat(),
        }
        return session_id

    def touch_session(self, session_id: str) -> bool:
        if session_id not in self.sessions:
            return False
        self.sessions[session_id]["last_heartbeat"] = time.time()
        return True

    def register_model(self, payload: Dict[str, Any], model_id: Optional[str] = None) -> str:
        _session_id = payload.get("session_id")
        _model_id: str = model_id or payload.get("model_id") or f"{payload.get('base_model', 'model')}-{_session_id or uuid.uuid4().hex[:8]}"
        _model_id = re.sub(r'[^\w\-]', '_', _model_id)

        self.models[_model_id] = {
            "session_id": payload.get("session_id"),
            "model_seq_id": payload.get("model_seq_id"),
            "base_model": payload.get("base_model"),
            "user_metadata": payload.get("user_metadata") or {},
            "lora_config": payload.get("lora_config"),
            "created_at": datetime.now().isoformat(),
        }
        return _model_id

    def unload_model(self, model_id: str) -> bool:
        return self.models.pop(model_id, None) is not None

    def create_sampling_session(self, payload: Dict[str, Any], sampling_session_id: Optional[str] = None) -> str:
        _sampling_session_id: str = sampling_session_id or payload.get("sampling_session_id") or f"sampling_{uuid.uuid4().hex}"
        self.sampling_sessions[_sampling_session_id] = {
            "session_id": payload.get("session_id"),
            "seq_id": payload.get("sampling_session_seq_id"),
            "base_model": payload.get("base_model"),
            "model_path": payload.get("model_path"),
            "created_at": datetime.now().isoformat(),
        }
        return _sampling_session_id

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        return self.models.get(model_id)

    def store_future(self, request_id: str, result: Any, model_id: Optional[str]):
        if hasattr(result, "model_dump"):
            result = result.model_dump()
        self.futures[request_id] = {
            "status": "completed",
            "result": result,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(),
        }

    def get_future(self, request_id: str) -> Optional[Dict[str, Any]]:
        return self.futures.get(request_id)


class ServerStateProxy:
    def __init__(self, actor_handle):
        self._actor = actor_handle

    def create_session(self, payload: Dict[str, Any]) -> str:
        return ray.get(self._actor.create_session.remote(payload))

    def touch_session(self, session_id: str) -> bool:
        return ray.get(self._actor.touch_session.remote(session_id))

    def register_model(self, payload: Dict[str, Any], model_id: Optional[str] = None) -> str:
        return ray.get(self._actor.register_model.remote(payload, model_id))

    def unload_model(self, model_id: str) -> bool:
        return ray.get(self._actor.unload_model.remote(model_id))

    def create_sampling_session(self, payload: Dict[str, Any], sampling_session_id: Optional[str] = None) -> str:
        return ray.get(self._actor.create_sampling_session.remote(payload, sampling_session_id))

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        return ray.get(self._actor.get_model_metadata.remote(model_id))

    async def store_future(self, request_id: str, result: Any, model_id: Optional[str]):
        # Make the Ray call asynchronously
        await self._actor.store_future.remote(request_id, result, model_id)

    def get_future(self, request_id: str) -> Optional[Dict[str, Any]]:
        return ray.get(self._actor.get_future.remote(request_id))


def get_server_state(actor_name: str = "tinker_server_state") -> ServerStateProxy:
    _ServerState = ray.remote(ServerState)
    try:
        actor = ray.get_actor(actor_name)
    except ValueError:
        try:
            actor = _ServerState.options(name=actor_name, lifetime="detached").remote()
        except ValueError:
            actor = ray.get_actor(actor_name)
    assert actor is not None
    return ServerStateProxy(actor)


async def schedule_task(
    state: ServerStateProxy,
    coro: Any,
    model_id: Optional[str] = None,
) -> types.UntypedAPIFuture:
    request_id = f"req_{uuid.uuid4().hex}"

    async def _runner():
        try:
            val = await coro
            await state.store_future(request_id, val, model_id)
        except Exception as e:
            # Structure the error so the client SDK can interpret it
            err_payload = {"error": str(e), "category": "Internal"}
            await state.store_future(request_id, err_payload, model_id)

    # Schedule execution in the background
    asyncio.create_task(_runner())
    return types.UntypedAPIFuture(request_id=request_id, model_id=model_id)
