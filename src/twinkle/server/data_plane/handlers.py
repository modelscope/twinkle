# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from collections.abc import Callable
from fastapi import Depends, FastAPI
from typing import TYPE_CHECKING

import twinkle_client.types as types

if TYPE_CHECKING:
    from .app import DataPlaneManagement


def register_data_plane_routes(app: FastAPI, self_fn: Callable[[], DataPlaneManagement]) -> None:

    @app.post('/twinkle/put', response_model=types.DataRef)
    async def put(body: types.DataPutRequest, self: DataPlaneManagement = Depends(self_fn)) -> types.DataRef:
        return await self.store.put(
            body.rows,
            kind=body.kind,
            tags=body.tags,
        )

    @app.post('/twinkle/get', response_model=types.DataRowsResponse)
    async def get(body: types.DataGetRequest, self: DataPlaneManagement = Depends(self_fn)) -> types.DataRowsResponse:
        rows = await self.store.get(
            body.ref,
            fields=body.fields,
        )
        tags = (await self.store.get_tags(body.ref) if body.include_tags else [])
        return types.DataRowsResponse(rows=rows, tags=tags)

    @app.post('/twinkle/append', response_model=types.DataRef)
    async def append(body: types.DataAppendRequest, self: DataPlaneManagement = Depends(self_fn)) -> types.DataRef:
        return await self.store.append(
            body.ref,
            body.rows,
            tags=body.tags,
        )

    @app.post('/twinkle/release')
    async def release(body: types.DataReleaseRequest, self: DataPlaneManagement = Depends(self_fn)) -> dict[str, str]:
        await self.store.release(body.ref)
        return {'status': 'ok'}
