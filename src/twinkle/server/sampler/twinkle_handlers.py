# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native sampler handler mixin.

Provides /twinkle/* sampler endpoints that call the sampler directly (no queue needed).
"""
import traceback
from fastapi import FastAPI, Request
from typing import Optional

from twinkle.data_format import InputFeature, SamplingParams, Trajectory
from twinkle.utils.logger import get_logger
from twinkle_client.types.sampler import (AddAdapterRequest, AddAdapterResponse, CreateResponse, HeartbeatRequest,
                                          HeartbeatResponse, SampleRequest, SampleResponseModel, SetTemplateRequest,
                                          SetTemplateResponse)

logger = get_logger()


class TwinkleSamplerHandlers:
    """
    Mixin providing Twinkle-native sampler endpoints.

    Expects the combined class to also have:
      self.sampler, self.state
    The class should also inherit AdapterManagerMixin for adapter lifecycle.
    """

    @staticmethod
    def _register_twinkle_sampler_routes(app: FastAPI):
        """Register all twinkle sampler routes on the given FastAPI app."""

        @staticmethod
        def _get_twinkle_sampler_adapter_name(request: Request, adapter_name: Optional[str]) -> Optional[str]:
            if adapter_name is None or adapter_name == '':
                return None
            return request.state.request_id + '-' + adapter_name

        @app.post('/twinkle/create', response_model=CreateResponse)
        def create(self, request: Request) -> CreateResponse:
            """Health check / session creation endpoint."""
            return CreateResponse()

        @app.post('/twinkle/sample', response_model=SampleResponseModel)
        def sample(self, request: Request, body: SampleRequest) -> SampleResponseModel:
            """Sample completions from the model.

            Supports Trajectory or InputFeature inputs, with optional LoRA adapter.
            """
            try:
                # Resolve adapter
                adapter_path = None
                adapter_name = body.adapter_name or ''
                full_adapter_name = _get_twinkle_sampler_adapter_name(request, adapter_name) or ''

                if body.adapter_uri:
                    from twinkle.server.common.io_utils import create_checkpoint_manager
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
                response = self.sampler.sample(
                    inputs,
                    params,
                    adapter_name=full_adapter_name,
                    adapter_path=adapter_path,
                    num_samples=body.num_samples,
                )
                if callable(response):
                    response = response()

                sequences = []
                for seq in response.sequences:
                    sequences.append({
                        'stop_reason': seq.stop_reason,
                        'tokens': list(seq.tokens),
                        'logprobs': list(seq.logprobs) if seq.logprobs is not None else None,
                    })

                return SampleResponseModel(
                    sequences=sequences,
                    prompt_logprobs=response.prompt_logprobs,
                    topk_prompt_logprobs=response.topk_prompt_logprobs,
                )
            except Exception:
                logger.error(traceback.format_exc())
                raise

        @app.post('/twinkle/set_template', response_model=SetTemplateResponse)
        def set_template(self, request: Request, body: SetTemplateRequest) -> SetTemplateResponse:
            """Set the chat template for encoding Trajectory inputs."""
            extra_kwargs = body.model_extra or {}
            self.sampler.set_template(body.template_cls, **extra_kwargs)
            return SetTemplateResponse()

        @app.post('/twinkle/add_adapter_to_sampler', response_model=AddAdapterResponse)
        def add_adapter_to_sampler(self, request: Request, body: AddAdapterRequest) -> AddAdapterResponse:
            """Add a LoRA adapter to the sampler."""
            assert body.adapter_name, 'You need to specify a valid `adapter_name`'
            full_adapter_name = _get_twinkle_sampler_adapter_name(request, body.adapter_name)
            from twinkle.server.utils.validation import get_token_from_request
            token = get_token_from_request(request)

            from peft import LoraConfig
            config = LoraConfig(**body.config) if isinstance(body.config, dict) else body.config

            self.register_adapter(full_adapter_name, token)
            self.sampler.add_adapter_to_sampler(full_adapter_name, config)

            return AddAdapterResponse(adapter_name=full_adapter_name)

        @app.post('/twinkle/heartbeat', response_model=HeartbeatResponse)
        def heartbeat(self, request: Request, body: HeartbeatRequest) -> HeartbeatResponse:
            """Keep an adapter alive by resetting its inactivity timer."""
            full_adapter_name = _get_twinkle_sampler_adapter_name(request, body.adapter_name)
            self.assert_adapter_exists(adapter_name=full_adapter_name)
            self.touch_adapter(full_adapter_name)
            return HeartbeatResponse()
