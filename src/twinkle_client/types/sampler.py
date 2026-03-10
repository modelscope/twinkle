# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Pydantic request/response models for twinkle sampler endpoints.

These models are used by both the server-side handler and the twinkle client.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


class SampleRequest(BaseModel):
    """Request body for the /sample endpoint."""
    inputs: Any = Field(..., description='List of Trajectory or InputFeature dicts')
    sampling_params: Optional[Dict[str, Any]] = Field(
        None, description='Sampling parameters (max_tokens, temperature, etc.)')
    adapter_name: str = Field('', description='Adapter name for LoRA inference')
    adapter_uri: Optional[str] = Field(
        None, description='Adapter URI (twinkle:// path or local path) for LoRA inference')
    num_samples: int = Field(1, description='Number of completions to generate per prompt')


class SampleResponseModel(BaseModel):
    """Response body for the /sample endpoint."""
    sequences: List[Dict[str, Any]] = Field(
        ..., description='List of sampled sequences, each with tokens, logprobs, stop_reason')
    prompt_logprobs: Optional[List[Optional[float]]] = None
    topk_prompt_logprobs: Optional[List[Optional[List]]] = None


class SetTemplateRequest(BaseModel):
    """Request body for the /set_template endpoint."""
    template_cls: str = Field(..., description="Template class name (e.g. 'Template')")
    adapter_name: str = Field('', description='Adapter name to associate the template with')

    class Config:
        extra = 'allow'


class SetTemplateResponse(BaseModel):
    """Response body for the /set_template endpoint."""
    status: str = 'ok'


class AddAdapterRequest(BaseModel):
    """Request body for the /add_adapter_to_sampler endpoint."""
    adapter_name: str = Field(..., description='Name of the adapter to add')
    config: Any = Field(..., description='LoRA configuration dict')


class AddAdapterResponse(BaseModel):
    """Response body for the /add_adapter_to_sampler endpoint."""
    status: str = 'ok'
    adapter_name: str


class HeartbeatRequest(BaseModel):
    """Request body for the /heartbeat endpoint."""
    adapter_name: str = Field(..., description='Adapter name to keep alive')


class HeartbeatResponse(BaseModel):
    """Response body for the /heartbeat endpoint."""
    status: str = 'ok'


class CreateResponse(BaseModel):
    """Response body for the /create endpoint."""
    status: str = 'ok'
