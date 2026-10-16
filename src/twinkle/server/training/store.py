# Copyright (c) ModelScope Contributors. All rights reserved.
"""Controller-owned records on the existing shared state backend."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from twinkle.server.state import ServerState
from twinkle.server.state.models import FutureFailureRecord
from .records import CreateTraining, InstanceRecord, TrainingRecord


class TrainingStore:

    def __init__(self, state: ServerState, namespace: str):
        self.state = state
        self.backend = state.backend
        self.prefix = f'training::{namespace}::'

    def key(self, kind: str, identity: str) -> str:
        return f'{self.prefix}{kind}::{identity}'

    @staticmethod
    def create_identity(request: CreateTraining) -> tuple[str, str]:
        identity = json.dumps([request.client_type, request.token, request.session_id, request.create_key])
        fingerprint = json.dumps(request.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(identity.encode()).hexdigest(), hashlib.sha256(fingerprint.encode()).hexdigest()

    async def reserve(self, record: TrainingRecord) -> None:
        # ModelManager applies the existing global quota atomically, including PENDING.
        await self.state.register_model_record(record.training_id, record)

    async def save_training(self, record: TrainingRecord) -> None:
        await self.backend.set(f'model::{record.training_id}', record.model_dump())

    async def retire(self, record: TrainingRecord) -> None:
        # Keep a bounded diagnostic tombstone after freeing the live quota row.
        await self.backend.set(
            self.key('terminal', record.training_id),
            record.model_dump(),
            ttl=max(1, int(self.state.expiration_timeout)))
        await self.state.unload_model(record.training_id)
        for pattern in (f'seq::*::{record.training_id}::*', f'tinker_seq::{record.training_id}::*',
                        f'seq_ledger::{record.training_id}::*'):
            for key in await self.backend.keys(pattern):
                await self.backend.delete(key)

    async def save_instance(self, record: InstanceRecord) -> None:
        ttl = max(1, int(self.state.expiration_timeout)) if record.state == 'STOPPED' else None
        await self.backend.set(self.key('instance', record.instance_id), record.model_dump(), ttl=ttl)

    async def instances(self) -> list[InstanceRecord]:
        keys = await self.backend.keys(self.key('instance', '*'))
        return [InstanceRecord.model_validate(value) for value in await self.backend.mget(keys) if value is not None]

    async def trainings(self) -> list[TrainingRecord]:
        keys = await self.backend.keys('model::*')
        records = await self.backend.mget(keys)
        return [
            TrainingRecord.model_validate(value) for value in records if isinstance(value, dict)
            and value.get('controller_epoch') and value.get('training_namespace') == self.prefix
        ]

    async def get_create(self, identity: str) -> dict[str, str] | None:
        return await self.backend.get(self.key('create', identity))

    async def save_create(self, identity: str, fingerprint: str, request_id: str, session_id: str) -> None:
        # Heartbeats can keep a session alive longer than expiration_timeout.
        # Keep its creation receipts until session cleanup, never recreate on retry.
        await self.backend.set(
            self.key('create', identity), {
                'fingerprint': fingerprint,
                'request_id': request_id,
                'session_id': session_id
            })

    async def cleanup_receipts(self) -> None:
        for key in await self.backend.keys(self.key('create', '*')):
            receipt = await self.backend.get(key)
            if receipt and await self.state.get_session_last_heartbeat(receipt['session_id']) is None:
                await self.backend.delete(key)

    async def future(self,
                     request_id: str,
                     status: str,
                     model_id: str | None,
                     *,
                     epoch: str | None = None,
                     result: Any = None,
                     failure: FutureFailureRecord | None = None,
                     deadline: float | None = None) -> None:
        await self.state.store_future_status(
            request_id, status, model_id, result=result, failure=failure, replica_id=epoch, absolute_deadline=deadline)

    async def own_future(self, request_id: str, token: str) -> None:
        await self.backend.set(f'future_owner::{request_id}', token, ttl=max(1, int(self.state.expiration_timeout)))

    async def begin_epoch(self, epoch: str) -> None:
        await self.backend.set(self.key('controller', 'active'), {'epoch': epoch, 'started_at': time.time()})
