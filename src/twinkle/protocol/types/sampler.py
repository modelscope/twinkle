# Copyright (c) ModelScope Contributors. All rights reserved.
"""Request / response models for the twinkle-native sampler endpoints.

Shared by the server handler and the twinkle client. Field roles follow
:mod:`twinkle.protocol.types.base`; the sampler handlers pass everything they need
explicitly. Template and patch requests each declare a passthrough region for
plugin arguments.

Class names carry a ``Sampler`` prefix wherever ``model.py`` already owns the bare
name (``AddAdapterRequest``, ``SetTemplateRequest``, ``CreateResponse`` and their
responses). The two modules describe *different* endpoints with different field sets;
a shared bare name is distinguished only by an import alias and, when a handler does
``import twinkle.protocol.types as types``, silently resolves to whichever module the
package ``__init__`` re-exported first -- which is how the sampler endpoints once
bound ``model.py``'s schema. Prefixing at the definition site removes the ambiguity,
matching :mod:`twinkle.protocol.types.processor`.
"""
from __future__ import annotations

from pydantic import Field, JsonValue, StrictInt, field_validator, model_validator
from typing import Any, Dict, List, Literal, Optional, Tuple

from .base import ResponseModel, StrictRequest, passthrough
from .data import Ints3, WireInputBatch

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
    token_only: bool = Field(
        False, description='Return sampled tokens/logprobs only; encoded inputs are engine ids, media alongside')


class SamplingMaskModel(ResponseModel):
    """CSR sampling support sets, mirroring the two fields of SamplingMask."""

    token_ids: list[StrictInt]
    offsets: list[StrictInt] = Field(min_length=1)

    @model_validator(mode='after')
    def validate_offsets(self) -> SamplingMaskModel:
        if self.offsets[0] != 0 or self.offsets[-1] != len(self.token_ids):
            raise ValueError('sampling_masks offsets must start at zero and end at len(token_ids)')
        if any(left > right for left, right in zip(self.offsets, self.offsets[1:])):
            raise ValueError('sampling_masks offsets must be nondecreasing')
        return self


class SampledSequenceModel(ResponseModel):
    """A single sampled sequence, mirroring twinkle.data_format.SampledSequence."""
    stop_reason: StopReason = Field(..., description="Stop reason: 'length' or 'stop'")
    tokens: list[int] = Field(..., description='Token IDs of the sampled sequence')
    logprobs: list[list[tuple[int, float]] | None] | None = Field(None, description='Per-token log-probabilities')
    decoded: str | None = Field(None, description='Decoded text of the sampled sequence')
    text: str | None = Field(None, description='Assistant content parsed by the serving template')
    new_input_feature: dict[str, Any] | None = Field(
        None, description='Complete trajectory after sampling; request media and user_data may be omitted on the wire')
    routed_experts: Ints3 | None = Field(None, description='Expert routing exported by token-only sampling')
    sampling_masks: SamplingMaskModel | None = Field(None, description='Support sets for sampled tokens')


class SampleResponseModel(ResponseModel):
    """Mirroring twinkle.data_format.SampleResponse."""
    sequences: list[SampledSequenceModel] = Field(..., description='List of sampled sequences')
    prompt_token_ids: list[int] | None = Field(None, description='Token IDs of the prompt the sequences continue')
    prompt_logprobs: list[float | None] | None = None
    topk_prompt_logprobs: list[list[tuple[int, float]] | None] | None = None


class SampleResponseModelList(ResponseModel):
    """Response body for the /sample endpoint"""
    samples: list[SampleResponseModel] = Field(..., description='List of sample responses')


class EncodeRequest(StrictRequest):
    """Request body for the ``/encode`` pooling endpoint.

    The pooling counterpart of :class:`SampleRequest`: the same wire-validated ``inputs``
    surface (Trajectory or InputFeature entries), with a ``pooling_params`` block in place
    of sampling parameters.
    """

    inputs: WireInputBatch = Field(..., description='Trajectory or InputFeature entries to run the pooling forward on')
    pooling_params: dict[str, JsonValue] | None = Field(
        None, description='Pooling parameters (task, dimensions, etc.)')
    adapter_name: str = Field('', description='Adapter name for LoRA inference')


class SamplerSetTemplateRequest(StrictRequest):
    """Request body for the sampler ``/set_template`` endpoint."""
    template_cls: str = Field(..., description="Template class name (e.g. 'Template')")
    adapter_name: str = Field('', description='Adapter name to associate the template with')
    init_kwargs: dict[str, JsonValue] = passthrough()


    @field_validator('template_cls')
    @classmethod
    def registered_template_name(cls, value):
        if not value.isidentifier():
            raise ValueError('HTTP template setup requires a registered class name, not a source path')
        return value


class SamplerSetTemplateResponse(ResponseModel):
    """Response body for the sampler /set_template endpoint."""
    status: str = 'ok'


class SamplerApplyPatchRequest(StrictRequest):
    """Apply a patch to the sampler instance."""
    patch_cls: str
    init_kwargs: dict[str, JsonValue] = passthrough()


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
