# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native sampler handler mixin.

Provides /twinkle/* sampler endpoints that call the sampler directly (no queue needed).
"""
from __future__ import annotations

import traceback
from fastapi import Depends, FastAPI, HTTPException, Request
from typing import TYPE_CHECKING, Callable

from twinkle.server.common.serialize import deserialize_object

if TYPE_CHECKING:
    from .app import SamplerManagement

import numpy as np

import twinkle_client.types as types
from twinkle.data_format import InputFeature, SamplingParams, Trajectory
from twinkle.utils.logger import get_logger

logger = get_logger()


def _serialize_input_feature(feature: dict) -> dict:
    """Convert numpy arrays / torch tensors in an InputFeature to plain Python lists."""
    result = {}
    for k, v in feature.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    result[k] = v.tolist()
                    continue
            except ImportError:
                pass
            result[k] = v
    return result


def _get_twinkle_sampler_adapter_name(request: Request, adapter_name: str | None) -> str | None:
    """Prefix the adapter name with the request ID for per-request isolation."""
    if adapter_name is None or adapter_name == '':
        return None
    return request.state.request_id + '-' + adapter_name


def _register_twinkle_sampler_routes(app: FastAPI, self_fn: Callable[[], SamplerManagement]) -> None:
    """Register all /twinkle/* sampler routes on the given FastAPI app.

    self_fn is a zero-argument callable returning the current SamplerManagement replica instance.
    It is wired in via Depends so it is resolved lazily at request time.
    """

    @app.post('/twinkle/create', response_model=types.CreateResponse)
    def create(request: Request, self: SamplerManagement = Depends(self_fn)) -> types.CreateResponse:
        """Health check / session creation endpoint."""
        return types.CreateResponse()

    @app.post('/twinkle/sample', response_model=types.SampleResponseModelList)
    def sample(request: Request, body: types.SampleRequest,
               self: SamplerManagement = Depends(self_fn)) -> types.SampleResponseModelList:
        """Sample completions from the model.

        Supports Trajectory or InputFeature inputs, with optional LoRA adapter.
        """
        try:
            # Resolve adapter
            adapter_path = None
            adapter_name = body.adapter_name or ''
            full_adapter_name = _get_twinkle_sampler_adapter_name(request, adapter_name) or ''

            if body.adapter_uri:
                from twinkle.server.common.checkpoint_factory import create_checkpoint_manager
                from twinkle.server.utils.validation import get_token_from_request
                token = get_token_from_request(request)
                checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
                _, adapter_path = checkpoint_manager.parse_adapter_uri(body.adapter_uri)

            # Parse inputs
            inputs = body.inputs
            if isinstance(inputs, list) and inputs:
                first = inputs[0]
                if isinstance(first, dict) and 'input_ids' in first:
                    inputs = [InputFeature(**item) for item in inputs]
                else:
                    inputs = [Trajectory(**item) for item in inputs]
            elif isinstance(inputs, dict):
                if 'input_ids' in inputs:
                    inputs = [InputFeature(**inputs)]
                else:
                    inputs = [Trajectory(**inputs)]

            # Build sampling params
            params = None
            if body.sampling_params:
                params = SamplingParams.from_dict(body.sampling_params)

            # Call sampler
            responses = self.sampler.sample(
                inputs,
                params,
                adapter_name=full_adapter_name,
                adapter_path=adapter_path,
            )
            if callable(responses):
                responses = responses()

            sample_models = []
            for response in responses:
                sequences = [
                    types.SampledSequenceModel(
                        stop_reason=seq.stop_reason,
                        tokens=list(seq.tokens),
                        logprobs=list(seq.logprobs) if seq.logprobs is not None else None,
                        decoded=seq.decoded,
                        new_input_feature=_serialize_input_feature(seq.new_input_feature)
                        if seq.new_input_feature is not None else None,
                    ) for seq in response.sequences
                ]

                sample_models.append(
                    types.SampleResponseModel(
                        sequences=sequences,
                        prompt_logprobs=response.prompt_logprobs,
                        topk_prompt_logprobs=response.topk_prompt_logprobs,
                    ))
            return types.SampleResponseModelList(samples=sample_models)
        except Exception:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post('/twinkle/set_template', response_model=types.SetTemplateResponse)
    def set_template(
            request: Request,
            body: types.SetTemplateRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> types.SetTemplateResponse:
        """Set the chat template for encoding Trajectory inputs."""
        extra_kwargs = body.model_extra or {}
        self.sampler.set_template(body.template_cls, **extra_kwargs)
        return types.SetTemplateResponse()

    @app.post('/twinkle/add_adapter_to_sampler', response_model=types.AddAdapterResponse)
    def add_adapter_to_sampler(
            request: Request,
            body: types.AddAdapterRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> types.AddAdapterResponse:
        """Add a LoRA adapter to the sampler."""
        assert body.adapter_name, 'You need to specify a valid `adapter_name`'
        full_adapter_name = _get_twinkle_sampler_adapter_name(request, body.adapter_name)
        from twinkle.server.utils.validation import get_token_from_request
        token = get_token_from_request(request)

        from peft import LoraConfig
        config = LoraConfig(**body.config) if isinstance(body.config, dict) else body.config

        self.register_adapter(full_adapter_name, token)
        self.sampler.add_adapter_to_sampler(full_adapter_name, config)

        return types.AddAdapterResponse(adapter_name=full_adapter_name)

    @app.post('/twinkle/apply_patch')
    async def apply_patch(
            request: Request,
            body: types.ApplyPatchRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> None:
        extra_kwargs = body.model_extra or {}
        patch_cls = deserialize_object(body.patch_cls)
        self.sampler.apply_patch(patch_cls, **extra_kwargs)
