# Copyright (c) ModelScope Contributors. All rights reserved.
"""Serializable identities and states. Ray handles never enter these records."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Any, Literal

from twinkle.server.state.models import ModelRecord

TrainingState = Literal['PENDING', 'ACTIVE', 'CLOSING', 'RELEASED', 'EXPIRED', 'FAILED', 'CANCELLED']
InstanceState = Literal['ALLOCATING', 'STARTING', 'READY', 'RESETTING', 'DRAINING', 'STOPPED', 'QUARANTINED']
TERMINAL_TRAINING = frozenset({'RELEASED', 'EXPIRED', 'FAILED', 'CANCELLED'})
LIVE_INSTANCES = frozenset({'ALLOCATING', 'STARTING', 'READY', 'RESETTING'})


class TrainingRecord(ModelRecord):
    training_id: str
    controller_epoch: str
    instance_id: str
    mode: Literal['lora', 'full']
    adapter_label: str | None = None
    state: TrainingState = 'PENDING'
    create_request_id: str
    creation_deadline: float
    last_activity_at: float
    pending_operation_ids: dict[str, dict[str, Any]] = Field(default_factory=dict)
    terminal_reason: str | None = None
    terminal_at: float | None = None


class InstanceRecord(BaseModel):
    instance_id: str
    controller_epoch: str
    base_model: str
    mode: Literal['lora', 'full']
    resident: bool = False
    state: InstanceState = 'ALLOCATING'
    training_ids: list[str] = Field(default_factory=list)
    startup_deadline: float
    application_name: str | None = None
    placement_group_id: str | None = None
    node_id: str | None = None
    devices: list[str] = Field(default_factory=list)
    incarnation: str | None = None
    dirty: bool = False
    failure: str | None = None
    stopped_at: float | None = None


class CreateTraining(BaseModel):
    base_model: str
    mode: Literal['lora', 'full']
    token: str
    session_id: str
    create_key: str
    client_type: Literal['twinkle', 'tinker']
    payload: dict[str, Any]
    adapter_label: str | None = None
