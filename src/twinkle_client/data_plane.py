# Copyright (c) ModelScope Contributors. All rights reserved.
"""Client for the server-side TransferQueue DataRef service."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, TypeVar

from twinkle_client.common.json_utils import json_safe
from twinkle_client.http import get_base_url, http_post
from twinkle_client.types.component import DataRef, DataRowsResponse


_T = TypeVar('_T')


async def _call_in_thread(func: Callable[..., _T], /, *args: Any, **kwargs: Any) -> _T:
    """Run one synchronous DataPlane operation without blocking the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


class DataPlaneClient:

    def __init__(self, server_url: str | None = None):
        self.server_url = (server_url or f'{get_base_url()}/data-plane').rstrip('/')

    def put(
        self,
        rows: list[dict[str, Any]],
        *,
        kind: str = 'data',
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        response = http_post(
            f'{self.server_url}/twinkle/put',
            json_data={'rows': json_safe(rows), 'kind': kind, 'tags': json_safe(tags)},
        )
        response.raise_for_status()
        return DataRef(**response.json())

    async def aput(
        self,
        rows: list[dict[str, Any]],
        *,
        kind: str = 'data',
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        """Asynchronously store rows while preserving :meth:`put` semantics."""
        if tags is None:
            return await _call_in_thread(self.put, rows, kind=kind)
        return await _call_in_thread(self.put, rows, kind=kind, tags=tags)

    def get(self, ref: DataRef, *, fields: list[str] | None = None) -> list[dict[str, Any]]:
        response = http_post(
            f'{self.server_url}/twinkle/get',
            json_data={'ref': ref.model_dump(), 'fields': fields},
        )
        response.raise_for_status()
        return DataRowsResponse(**response.json()).rows

    def get_batch(
        self,
        ref: DataRef,
        *,
        fields: list[str] | None = None,
    ) -> DataRowsResponse:
        response = http_post(
            f'{self.server_url}/twinkle/get',
            json_data={'ref': ref.model_dump(), 'fields': fields, 'include_tags': True},
        )
        response.raise_for_status()
        return DataRowsResponse(**response.json())

    async def aget(self, ref: DataRef, *, fields: list[str] | None = None) -> list[dict[str, Any]]:
        """Asynchronously fetch rows while preserving :meth:`get` semantics."""
        if fields is None:
            return await _call_in_thread(self.get, ref)
        return await _call_in_thread(self.get, ref, fields=fields)

    async def aget_batch(
        self,
        ref: DataRef,
        *,
        fields: list[str] | None = None,
    ) -> DataRowsResponse:
        if fields is None:
            return await _call_in_thread(self.get_batch, ref)
        return await _call_in_thread(self.get_batch, ref, fields=fields)

    def append(
        self,
        ref: DataRef,
        rows: list[dict[str, Any]],
        *,
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        response = http_post(
            f'{self.server_url}/twinkle/append',
            json_data={
                'ref': ref.model_dump(),
                'rows': json_safe(rows),
                'tags': json_safe(tags),
            },
        )
        response.raise_for_status()
        return DataRef(**response.json())

    async def aappend(
        self,
        ref: DataRef,
        rows: list[dict[str, Any]],
        *,
        tags: list[dict[str, Any]] | None = None,
    ) -> DataRef:
        """Asynchronously append rows while preserving :meth:`append` semantics."""
        if tags is None:
            return await _call_in_thread(self.append, ref, rows)
        return await _call_in_thread(self.append, ref, rows, tags=tags)

    def release(self, ref: DataRef) -> None:
        response = http_post(
            f'{self.server_url}/twinkle/release',
            json_data={'ref': ref.model_dump()},
        )
        response.raise_for_status()

    async def arelease(self, ref: DataRef) -> None:
        """Asynchronously release a reference while preserving :meth:`release` semantics."""
        await _call_in_thread(self.release, ref)
