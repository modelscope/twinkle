# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Tinker-compatible sampler handler mixin.

Provides POST /tinker/asample using schedule_task() returning UntypedAPIFuture.
"""
from __future__ import annotations

import os
import traceback
from fastapi import Depends, FastAPI, Request
from tinker import types
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .app import SamplerManagement

from twinkle.data_format import SamplingParams
from twinkle.server.common.checkpoint_factory import create_checkpoint_manager
from twinkle.utils.logger import get_logger

logger = get_logger()


def _register_tinker_sampler_routes(app: FastAPI, self_fn: Callable[[], SamplerManagement]) -> None:
    """Register the tinker sampler route on the given FastAPI app.

    self_fn is a zero-argument callable returning the current SamplerManagement replica instance.
    It is wired in via Depends so it is resolved lazily at request time.
    """

    @app.post('/tinker/asample')
    async def asample(request: Request, body: types.SampleRequest,
                      self: SamplerManagement = Depends(self_fn)) -> types.UntypedAPIFuture:
        """Execute text generation (inference) for Tinker clients.

        Args:
            request: FastAPI request with auth token
            body: SampleRequest with prompt, sampling params, and adapter info

        Returns:
            UntypedAPIFuture wrapping SampleResponse with generated sequences
        """
        token = await self._on_request_start(request)

        async def _do_sample():
            try:
                # Extract prompt token IDs from ModelInput
                prompt_inputs = {'input_ids': body.prompt.to_ints()}

                # Get model_path from body or sampling session
                model_path = body.model_path
                if not model_path and body.sampling_session_id:
                    session = self.state.get_sampling_session(body.sampling_session_id)
                    if session:
                        model_path = session.get('model_path')

                # Parse and resolve adapter URI from model_path
                adapter_uri = None
                if model_path:
                    checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
                    adapter_name, adapter_uri = checkpoint_manager.parse_adapter_uri(model_path)

                # Validate adapter URI
                if not adapter_uri or not os.path.exists(adapter_uri):
                    return types.RequestFailedResponse(
                        error=f'Adapter URI {model_path} does not exist. Please check the model_path.',
                        category=types.RequestErrorCategory.User,
                    )

                # Convert tinker SamplingParams to twinkle SamplingParams if needed
                sampling_params = None
                if body.sampling_params:
                    sampling_params = SamplingParams(
                        max_tokens=body.sampling_params.max_tokens or 256,
                        temperature=body.sampling_params.temperature or 1.0,
                        top_p=body.sampling_params.top_p,
                        top_k=body.sampling_params.top_k,
                        stop=body.sampling_params.stop,
                    )

                response = self.sampler.sample(
                    inputs=[prompt_inputs] * body.num_samples,
                    sampling_params=sampling_params,
                    adapter_path=adapter_uri,
                )

                # Convert twinkle SampleResponse to tinker types
                tinker_sequences = []
                for seq in response.sequences:
                    logprobs = None
                    if seq.logprobs is not None:
                        if any(lp is None for lp in seq.logprobs):
                            logprobs = None
                        else:
                            logprobs = list(seq.logprobs)
                    tinker_sequences.append(
                        types.SampledSequence(
                            stop_reason=seq.stop_reason,
                            tokens=list(seq.tokens),
                            logprobs=logprobs,
                        ))
                return types.SampleResponse(
                    sequences=tinker_sequences,
                    prompt_logprobs=response.prompt_logprobs,
                    topk_prompt_logprobs=response.topk_prompt_logprobs,
                )
            except Exception:
                logger.error(traceback.format_exc())
                return types.RequestFailedResponse(
                    error=traceback.format_exc(),
                    category=types.RequestErrorCategory.Server,
                )

        input_tokens = len(body.prompt.to_ints())
        return await self.schedule_task(
            _do_sample,
            token=token,
            input_tokens=input_tokens,
            task_type='sample',
        )
