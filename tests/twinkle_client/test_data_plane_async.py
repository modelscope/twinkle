# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import threading

import pytest

from twinkle_client.data_plane import DataPlaneClient
from twinkle_client.types import DataRef, DataRowsResponse


def test_async_convenience_methods_delegate_to_sync_operations(monkeypatch) -> None:
    client = DataPlaneClient('http://server/data-plane')
    original_ref = DataRef(ref_id='data-1', size=1, fields=['value'])
    appended_ref = DataRef(ref_id='data-1', size=2, fields=['value'])
    calls = []
    caller_thread = threading.get_ident()

    def put(rows, *, kind='data'):
        calls.append(('put', rows, kind, threading.get_ident()))
        return original_ref

    def get(ref, *, fields=None):
        calls.append(('get', ref, fields, threading.get_ident()))
        return [{'value': 1}]

    def append(ref, rows):
        calls.append(('append', ref, rows, threading.get_ident()))
        return appended_ref

    def release(ref):
        calls.append(('release', ref, threading.get_ident()))

    monkeypatch.setattr(client, 'put', put)
    monkeypatch.setattr(client, 'get', get)
    monkeypatch.setattr(client, 'append', append)
    monkeypatch.setattr(client, 'release', release)

    async def run():
        assert await client.aput([{'value': 1}], kind='rollout') == original_ref
        assert await client.aget(original_ref, fields=['value']) == [{'value': 1}]
        assert await client.aappend(original_ref, [{'value': 2}]) == appended_ref
        assert await client.arelease(appended_ref) is None

    asyncio.run(run())

    assert [call[:-1] for call in calls] == [
        ('put', [{'value': 1}], 'rollout'),
        ('get', original_ref, ['value']),
        ('append', original_ref, [{'value': 2}]),
        ('release', appended_ref),
    ]
    assert all(call[-1] != caller_thread for call in calls)


def test_async_convenience_method_propagates_sync_error(monkeypatch) -> None:
    client = DataPlaneClient('http://server/data-plane')

    def fail(_rows, *, kind='data'):
        raise RuntimeError(f'put failed for {kind}')

    monkeypatch.setattr(client, 'put', fail)

    with pytest.raises(RuntimeError, match='put failed for rollout'):
        asyncio.run(client.aput([], kind='rollout'))


def test_async_tagged_methods_and_batch_read_delegate_to_sync_operations(monkeypatch) -> None:
    client = DataPlaneClient('http://server/data-plane')
    ref = DataRef(ref_id='data-1', size=1, fields=['value'])
    tags = [{'group_id': 'group-1'}]
    calls = []

    def put(rows, *, kind='data', tags=None):
        calls.append(('put', rows, kind, tags))
        return ref

    def get_batch(value, *, fields=None):
        calls.append(('get_batch', value, fields))
        return DataRowsResponse(rows=[{'value': 1}], tags=tags)

    def append(value, rows, *, tags=None):
        calls.append(('append', value, rows, tags))
        return value

    monkeypatch.setattr(client, 'put', put)
    monkeypatch.setattr(client, 'get_batch', get_batch)
    monkeypatch.setattr(client, 'append', append)

    async def run():
        assert await client.aput([{'value': 1}], tags=tags) == ref
        assert await client.aget_batch(ref) == DataRowsResponse(rows=[{'value': 1}], tags=tags)
        assert await client.aappend(ref, [{'reward': 1.0}], tags=tags) == ref

    asyncio.run(run())

    assert calls == [
        ('put', [{'value': 1}], 'data', tags),
        ('get_batch', ref, None),
        ('append', ref, [{'reward': 1.0}], tags),
    ]
