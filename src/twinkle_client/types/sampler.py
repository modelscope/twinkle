# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Pydantic request/response models for twinkle sampler endpoints.

These models are used by both the server-side handler and the twinkle client.
"""
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal, Optional, Tuple

StopReason = Literal['length', 'stop']


class SampleRequest(BaseModel):
    """Request body for the /sample endpoint."""
    inputs: Any = Field(..., description='List of Trajectory or InputFeature dicts')
    sampling_params: Optional[Dict[str, Any]] = Field(
        None, description='Sampling parameters (max_tokens, temperature, num_samples, etc.)')
    adapter_name: str = Field('', description='Adapter name for LoRA inference')
    adapter_uri: Optional[str] = Field(
        None, description='Adapter URI (twinkle:// path or local path) for LoRA inference')


class SampledSequenceModel(BaseModel):
    """A single sampled sequence, mirroring twinkle.data_format.SampledSequence."""
    stop_reason: StopReason = Field(..., description="Stop reason: 'length' or 'stop'")
    tokens: List[int] = Field(..., description='Token IDs of the sampled sequence')
    logprobs: Optional[List[Optional[List[Tuple[int, float]]]]] = Field(None, description='Per-token log-probabilities')
    decoded: Optional[str] = Field(None, description='Decoded text of the sampled sequence')
    new_input_feature: Optional[Dict[str, Any]] = Field(
        None, description='Updated InputFeature after sampling (input_ids, labels, etc.)')


class SampleResponseModel(BaseModel):
    """Mirroring twinkle.data_format.SampleResponse."""
    sequences: List[SampledSequenceModel] = Field(
        ..., description='List of sampled sequences')
    prompt_logprobs: Optional[List[Optional[float]]] = None
    topk_prompt_logprobs: Optional[List[Optional[List[Tuple[int, float]]]]] = None


class SampleResponseModelList(BaseModel):
    """Response body for the /sample endpoint"""
    samples: List[SampleResponseModel] = Field(..., description='List of sample responses')


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


class CreateResponse(BaseModel):
    """Response body for the /create endpoint."""
    status: str = 'ok'
