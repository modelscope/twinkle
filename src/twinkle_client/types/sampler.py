# Copyright (c) ModelScope Contributors. All rights reserved.
"""Request / response models for the twinkle-native sampler endpoints.

Shared by the server handler and the twinkle client. Field roles follow
:mod:`twinkle_client.types.base`; the sampler handlers pass everything they need
explicitly, so these requests carry control fields and -- for the template setter --
one passthrough region, and no free-floating backend kwargs.

Class names carry a ``Sampler`` prefix wherever ``model.py`` already owns the bare
name (``AddAdapterRequest``, ``SetTemplateRequest``, ``CreateResponse`` and their
responses). The two modules describe *different* endpoints with different field sets;
a shared bare name is distinguished only by an import alias and, when a handler does
``import twinkle_client.types as types``, silently resolves to whichever module the
package ``__init__`` re-exported first -- which is how the sampler endpoints once
bound ``model.py``'s schema. Prefixing at the definition site removes the ambiguity,
matching :mod:`twinkle_client.types.processor`.
"""
from __future__ import annotations

from pydantic import Field, JsonValue
from typing import Any, Dict, List, Literal, Optional, Tuple

from .base import ResponseModel, StrictRequest, passthrough
from .data import WireInputBatch

StopReason = Literal['length', 'stop', 'abort', 'error']


class SampleRequest(StrictRequest):
    """Request body for the ``/sample`` and ``/sample_stream`` endpoints.

    ``num_samples`` is not a top-level field: it is a sampling parameter and
    ``SamplingParams.from_dict(sampling_params)`` is the one place sampling
    parameters are built. A second, top-level spelling would be a second source of
    truth for the same value.
    """

    inputs: WireInputBatch = Field(..., description='Trajectory or InputFeature entries to sample from')
    sampling_params: dict[str, JsonValue] | None = Field(
        None, description='Sampling parameters (max_tokens, temperature, num_samples, etc.)')
    adapter_name: str = Field('', description='Adapter name for LoRA inference')
    adapter_uri: str | None = Field(None, description='Adapter URI (twinkle:// path or local path) for LoRA inference')


class SampledSequenceModel(ResponseModel):
    """A single sampled sequence, mirroring twinkle.data_format.SampledSequence."""
    stop_reason: StopReason = Field(..., description="Stop reason: 'length' or 'stop'")
    tokens: list[int] = Field(..., description='Token IDs of the sampled sequence')
    logprobs: list[list[tuple[int, float]] | None] | None = Field(None, description='Per-token log-probabilities')
    decoded: str | None = Field(None, description='Decoded text of the sampled sequence')
    new_input_feature: dict[str, Any] | None = Field(
        None, description='Updated InputFeature after sampling (input_ids, labels, etc.)')


class SampleResponseModel(ResponseModel):
    """Mirroring twinkle.data_format.SampleResponse."""
    sequences: list[SampledSequenceModel] = Field(..., description='List of sampled sequences')
    prompt_token_ids: list[int] | None = Field(None, description='Token IDs of the prompt the sequences continue')
    prompt_logprobs: list[float | None] | None = None
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None


class SampleResponseModelList(ResponseModel):
    """Response body for the /sample endpoint"""
    samples: list[SampleResponseModel] = Field(..., description='List of sample responses')


class SamplerSetTemplateRequest(StrictRequest):
    """Request body for the sampler ``/set_template`` endpoint."""
    template_cls: str = Field(..., description="Template class name (e.g. 'Template')")
    adapter_name: str = Field('', description='Adapter name to associate the template with')
    init_kwargs: dict[str, JsonValue] = passthrough()


class SamplerSetTemplateResponse(ResponseModel):
    """Response body for the sampler /set_template endpoint."""
    status: str = 'ok'


class SamplerAddAdapterRequest(StrictRequest):
    """Request body for the ``/add_adapter_to_sampler`` endpoint."""
    adapter_name: str = Field(..., description='Name of the adapter to add')
    config: Any = Field(..., description='LoRA configuration dict')


class SamplerAddAdapterResponse(ResponseModel):
    """Response body for the /add_adapter_to_sampler endpoint."""
    status: str = 'ok'
    adapter_name: str


class SamplerCreateResponse(ResponseModel):
    """Response body for the sampler /create endpoint."""
    status: str = 'ok'
