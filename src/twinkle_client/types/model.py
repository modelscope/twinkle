# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Pydantic request/response models for twinkle model management endpoints.

These models are used by both the server-side handler and the twinkle client.
"""
from pydantic import BaseModel
from typing import Any, Optional


class CreateRequest(BaseModel):

    class Config:
        extra = 'allow'


class ForwardRequest(BaseModel):
    inputs: Any
    adapter_name: str

    class Config:
        extra = 'allow'


class ForwardOnlyRequest(BaseModel):
    inputs: Any
    adapter_name: Optional[str] = None

    class Config:
        extra = 'allow'


class AdapterRequest(BaseModel):
    adapter_name: str

    class Config:
        extra = 'allow'


class SetLossRequest(BaseModel):
    loss_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class SetOptimizerRequest(BaseModel):
    optimizer_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class SetLrSchedulerRequest(BaseModel):
    scheduler_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class SaveRequest(BaseModel):
    adapter_name: str
    save_optimizer: bool = False
    name: Optional[str] = None

    class Config:
        extra = 'allow'


class UploadToHubRequest(BaseModel):
    checkpoint_dir: str
    hub_model_id: str
    hub_token: Optional[str] = None
    async_upload: bool = True

    class Config:
        extra = 'allow'


class LoadRequest(BaseModel):
    adapter_name: str
    load_optimizer: bool = False
    name: str

    class Config:
        extra = 'allow'


class AddAdapterRequest(BaseModel):
    adapter_name: str
    config: str

    class Config:
        extra = 'allow'


class SetTemplateRequest(BaseModel):
    template_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class SetProcessorRequest(BaseModel):
    processor_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class HeartbeatRequest(BaseModel):
    adapter_name: str


class CalculateMetricRequest(BaseModel):
    adapter_name: str
    is_training: bool = True

    class Config:
        extra = 'allow'


class GetStateDictRequest(BaseModel):
    adapter_name: str

    class Config:
        extra = 'allow'
