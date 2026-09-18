# Copyright (c) ModelScope Contributors. All rights reserved.
"""Protocol types for directly orchestrating asynchronous server components."""
from __future__ import annotations

from pydantic import BaseModel, Field, JsonValue, model_validator
from typing import Any, Optional

from .base import ResponseModel, StrictRequest
from .data import WireInputBatch


class DataRef(BaseModel):
    """Opaque reference to rows stored in the server-side TransferQueue.

    A value carried inside other bodies rather than a body of its own, and it is
    round-tripped by the client, so it keeps the plain base. No wire schema is applied
    to what it points at: the rows never travel in the request body, so the data-plane
    constraints would be a category error here.
    """

    ref_id: str
    size: int
    fields: list[str] = Field(default_factory=list)
    kind: str = 'data'
    num_tokens: int = 0


class DataPutRequest(StrictRequest):
    rows: list[dict[str, Any]]
    kind: str = 'data'
    tags: Optional[list[dict[str, Any]]] = None


class DataGetRequest(StrictRequest):
    ref: DataRef
    fields: Optional[list[str]] = None
    include_tags: bool = False


class DataAppendRequest(StrictRequest):
    ref: DataRef
    rows: list[dict[str, Any]]
    tags: Optional[list[dict[str, Any]]] = None


class DataReleaseRequest(StrictRequest):
    ref: DataRef


class DataRowsResponse(ResponseModel):
    rows: list[dict[str, Any]]
    tags: list[dict[str, Any]] = Field(default_factory=list)


class DataPlaneSampleRequest(StrictRequest):
    """Body of ``POST /twinkle/sample_to_data_plane``.

    Exactly one input source: inline entries (wire-validated) or a ``DataRef``.
    """

    inputs: Optional[WireInputBatch] = None
    input_ref: Optional[DataRef] = None
    sampling_params: Optional[dict[str, JsonValue]] = None
    adapter_name: str = ''
    adapter_uri: Optional[str] = None
    policy_version: Optional[int] = None
    group_ids: Optional[list[str]] = None
    num_samples: int = Field(default=1, ge=1)

    @model_validator(mode='after')
    def validate_input(self) -> 'DataPlaneSampleRequest':
        if (self.inputs is None) == (self.input_ref is None):
            raise ValueError('exactly one of inputs and input_ref must be provided')
        if self.group_ids is not None and self.inputs is not None:
            if len(self.group_ids) != len(self.inputs):
                raise ValueError('group_ids must contain one value per sampler input')
        return self


class UnloadAdapterPathsRequest(StrictRequest):
    adapter_paths: list[str]
