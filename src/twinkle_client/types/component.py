# Copyright (c) ModelScope Contributors. All rights reserved.
"""Protocol types for directly orchestrating asynchronous server components."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class DataRef(BaseModel):
    """Opaque reference to rows stored in the server-side TransferQueue."""

    ref_id: str
    size: int
    fields: list[str] = Field(default_factory=list)
    kind: str = 'data'
    num_tokens: int = 0


class DataPutRequest(BaseModel):
    rows: list[dict[str, Any]]
    kind: str = 'data'
    tags: list[dict[str, Any]] | None = None


class DataGetRequest(BaseModel):
    ref: DataRef
    fields: list[str] | None = None
    include_tags: bool = False


class DataAppendRequest(BaseModel):
    ref: DataRef
    rows: list[dict[str, Any]]
    tags: list[dict[str, Any]] | None = None


class DataReleaseRequest(BaseModel):
    ref: DataRef


class DataRowsResponse(BaseModel):
    rows: list[dict[str, Any]]
    tags: list[dict[str, Any]] = Field(default_factory=list)


class DataPlaneSampleRequest(BaseModel):
    inputs: Any = None
    input_ref: DataRef | None = None
    sampling_params: dict[str, Any] | None = None
    adapter_name: str = ''
    adapter_uri: str | None = None
    policy_version: int | None = None
    group_ids: list[str] | None = None
    num_samples: int = 1

    @model_validator(mode='after')
    def validate_input(self) -> 'DataPlaneSampleRequest':
        if (self.inputs is None) == (self.input_ref is None):
            raise ValueError('exactly one of inputs and input_ref must be provided')
        if self.group_ids is not None and self.inputs is not None:
            size = len(self.inputs) if isinstance(self.inputs, list) else 1
            if len(self.group_ids) != size:
                raise ValueError('group_ids must contain one value per sampler input')
        return self


class UnloadAdapterPathsRequest(BaseModel):
    adapter_paths: list[str]
