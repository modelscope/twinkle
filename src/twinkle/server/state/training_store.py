# Copyright (c) ModelScope Contributors. All rights reserved.
"""Training persistence and retry receipts, composed from shared state managers."""
from __future__ import annotations

import functools
import hashlib
import json
import time
from typing import Any

from twinkle.server.exceptions import RequestRejectedError
from .backend.base import StateBackend
from .future_manager import FutureManager
from .model_manager import ModelManager
from .models import FutureFailureRecord
from .session_manager import SessionManager
from .training_records import CreateTraining, InstanceRecord, TrainingRecord


class TrainingStore:

    def __init__(self, namespace: str, backend: StateBackend, *, models: ModelManager, sessions: SessionManager,
                 futures: FutureManager, sequences: TrainingSequences, expiration_timeout: float):
        self.backend = backend
        self.models, self.sessions, self.futures, self.sequences = models, sessions, futures, sequences
        self.expiration_timeout = expiration_timeout
        self.prefix = f'training::{namespace}::'

    def key(self, kind: str, identity: str) -> str:
        return f'{self.prefix}{kind}::{identity}'

    @staticmethod
    def create_identity(request: CreateTraining) -> tuple[str, str]:
        identity = json.dumps([request.client_type, request.token, request.session_id, request.create_key])
        fingerprint = json.dumps(request.model_dump(mode='json'), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(identity.encode()).hexdigest(), hashlib.sha256(fingerprint.encode()).hexdigest()

    async def reserve(self, record: TrainingRecord) -> None:
        await self.models.add(record.training_id, record)

    async def save_training(self, record: TrainingRecord) -> None:
        await self.models.save_record(record.training_id, record)

    async def retire(self, record: TrainingRecord) -> None:
        await self.backend.set(
            self.key('terminal', record.training_id), record.model_dump(), ttl=max(1, int(self.expiration_timeout)))
        await self.sequences.delete(record.training_id)
        # Release the quota row last so a failed earlier step can be idempotently retried.
        await self.models.remove(record.training_id)

    async def save_instance(self, record: InstanceRecord) -> None:
        ttl = max(1, int(self.expiration_timeout)) if record.state == 'STOPPED' else None
        await self.backend.set(self.key('instance', record.instance_id), record.model_dump(), ttl=ttl)

    async def instances(self) -> list[InstanceRecord]:
        keys = await self.backend.keys(self.key('instance', '*'))
        return [InstanceRecord.model_validate(value) for value in await self.backend.mget(keys) if value is not None]

    async def trainings(self) -> list[TrainingRecord]:
        return await self.models.get_training_records(self.prefix)

    async def get_create(self, identity: str) -> dict[str, str] | None:
        return await self.backend.get(self.key('create', identity))

    async def save_create(self, identity: str, fingerprint: str, request_id: str, session_id: str) -> None:
        await self.backend.set(
            self.key('create', identity), {
                'fingerprint': fingerprint,
                'request_id': request_id,
                'session_id': session_id
            })

    async def cleanup_receipts(self) -> None:
        for key in await self.backend.keys(self.key('create', '*')):
            receipt = await self.backend.get(key)
            if receipt and await self.sessions.get_last_heartbeat(receipt['session_id']) is None:
                await self.backend.delete(key)

    async def future(self,
                     request_id: str,
                     status: str,
                     model_id: str | None,
                     *,
                     epoch: str | None = None,
                     result: Any = None,
                     failure: FutureFailureRecord | None = None,
                     deadline: float | None = None,
                     owner_token: str | None = None) -> None:
        await self.futures.store_status(
            request_id,
            status,
            model_id,
            result=result,
            failure=failure,
            replica_id=epoch,
            absolute_deadline=deadline,
            owner_token=owner_token)

    async def begin_epoch(self, epoch: str) -> None:
        await self.backend.set(self.key('controller', 'active'), {'epoch': epoch, 'started_at': time.time()})


class TrainingSequences:
    """Atomic retry receipts and an expired-sequence fence, scoped to a training handle.

    Expiration compacts receipts but retains the sequence floor until that handle is
    retired. An old update therefore cannot be applied twice after its receipt ages out.
    """

    def __init__(self, backend: StateBackend):
        self.backend = backend

    async def claim(self, training_id: str, stream: str, sequence: int, request_id: str,
                    retention: float) -> str | None:
        ledger = await self.backend.update_atomic(
            f'seq_ledger::{training_id}::{stream}',
            functools.partial(
                self.claim_transform, sequence=sequence, request_id=request_id, now=time.time(), retention=retention))
        receipt = ledger['receipts'].get(str(sequence))
        if receipt is None:
            raise RequestRejectedError('Sequence retry window expired; the operation will not be executed again')
        return None if receipt['request_id'] == request_id else receipt['request_id']

    async def release(self, training_id: str, stream: str, sequence: int, request_id: str) -> None:
        await self.backend.update_atomic(
            f'seq_ledger::{training_id}::{stream}',
            functools.partial(self.release_transform, sequence=sequence, request_id=request_id))

    async def delete(self, training_id: str) -> None:
        for key in await self.backend.keys(f'seq_ledger::{training_id}::*'):
            await self.backend.delete(key)

    @staticmethod
    def claim_transform(current, *, sequence, request_id, now, retention):
        floor = current.get('floor') if current else None
        receipts = {}
        for key, receipt in (current or {}).get('receipts', {}).items():
            if receipt['expires_at'] <= now:
                floor = int(key) if floor is None else max(floor, int(key))
            else:
                receipts[key] = receipt
        key = str(sequence)
        # Unexpired receipts remain retryable even if another sequence expired first.
        if key not in receipts and (floor is None or sequence > floor):
            receipts[key] = {'request_id': request_id, 'expires_at': now + retention}
        return {'floor': floor, 'receipts': receipts}

    @staticmethod
    def release_transform(current, *, sequence, request_id):
        if current is None:
            return None
        receipts = dict(current['receipts'])
        key = str(sequence)
        if receipts.get(key, {}).get('request_id') == request_id:
            receipts.pop(key)
        return {'floor': current['floor'], 'receipts': receipts}
