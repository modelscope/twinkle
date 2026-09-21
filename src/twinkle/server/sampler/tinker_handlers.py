# Copyright (c) ModelScope Contributors. All rights reserved.
"""Tinker-compatible routes for the Sampler deployment.

Registered by ``_register_tinker_sampler_routes(app, self_fn)`` -- module-level route
registration closing over ``self_fn`` via ``Depends``, not a mixin: there is no
inheritance relationship with the deployment class. Provides POST /tinker/asample using
schedule_task() returning UntypedAPIFuture.
"""
from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from fastapi import Depends, FastAPI, Request
from tinker import types
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import SamplerManagement

from twinkle.data_format import SamplingParams
from twinkle.server.checkpoint import create_checkpoint_manager
from twinkle.server.sampler.weights import resolve_sampler_weights
from twinkle.server.task_queue.types import UserTaskError
from twinkle.server.utils import get_template_for_model
from twinkle.utils.logger import get_logger

logger = get_logger()


def _sampled_sequence(*, stop_reason, tokens, logprobs):
    return types.SampledSequence(
        stop_reason=stop_reason,
        tokens=tokens,
        logprobs=logprobs,
    )


def _sample_response(*, sequences, prompt_logprobs, topk_prompt_logprobs):
    return types.SampleResponse(
        sequences=sequences,
        prompt_logprobs=prompt_logprobs,
        topk_prompt_logprobs=topk_prompt_logprobs,
    )


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

                # Set template for sampler based on model type
                template = get_template_for_model(self.model_id)
                await self.call_backend(self.sampler.set_template, template, model_id=self.model_id)
                # Reset prefix cache unconditionally on every tinker request (by
                # design): the tinker dialect does not signal whether weights
                # changed, so it always invalidates. This differs from the twinkle
                # endpoints, which reset only when an adapter_uri is supplied.
                await self.call_backend(self.sampler.reset_prefix_cache)

                # Get model_path from body or sampling session
                model_path = body.model_path
                if not model_path and body.sampling_session_id:
                    session = await self.state.get_sampling_session(body.sampling_session_id)
                    if session:
                        model_path = session.get('model_path')

                # Parse and resolve adapter URI from model_path
                adapter_uri = None
                if model_path:
                    checkpoint_manager = create_checkpoint_manager(token, client_type='tinker')
                    adapter_name, adapter_uri = checkpoint_manager.parse_adapter_uri(model_path)

                # Base-model sampling is valid when no model_path was provided.
                if adapter_uri and not os.path.exists(adapter_uri):
                    raise UserTaskError(f'Adapter URI {model_path} does not exist. Please check the model_path.')

                # Convert tinker SamplingParams to twinkle SamplingParams if needed
                sampling_params = None
                if body.sampling_params:
                    sampling_params = SamplingParams(
                        max_tokens=body.sampling_params.max_tokens or 256,
                        temperature=body.sampling_params.temperature or 1.0,
                        top_p=body.sampling_params.top_p,
                        top_k=body.sampling_params.top_k,
                        stop=body.sampling_params.stop,
                        # tinker 0.16.1 has no SamplingParams.logprobs field, but its
                        # SampledSequence contract and GRPO training require one
                        # chosen-token logprob per generated token.
                        logprobs=1,
                    )

                # LoRA adapter dir vs full-parameter checkpoint (shared helper);
                # a full checkpoint is loaded into the base model and yields no
                # LoRA path.
                lora_path = await resolve_sampler_weights(self, adapter_uri)

                responses = await self.call_backend(
                    self.sampler.sample,
                    inputs=[prompt_inputs] * body.num_samples,
                    sampling_params=sampling_params,
                    adapter_path=lora_path,
                )

                tinker_sequences = []
                for response in responses:
                    # twinkle logprobs are ``List[List[Tuple[int, float]]]``
                    # (top-k per position); tinker wants ``List[float]``
                    # (chosen-token logprob per position).
                    for seq in response.sequences:
                        logprobs = None
                        if seq.logprobs is not None:
                            try:
                                flattened = [float(lp_list[0][1]) for lp_list in seq.logprobs if lp_list]
                            except (IndexError, TypeError):
                                flattened = []
                            if len(flattened) == len(seq.tokens):
                                logprobs = flattened
                            else:
                                raise RuntimeError(
                                    f'Sampler returned {len(flattened)} logprobs for {len(seq.tokens)} generated '
                                    'tokens; refusing to return a misaligned Tinker SampledSequence.')
                        tinker_sequences.append(
                            _sampled_sequence(
                                stop_reason=seq.stop_reason,
                                tokens=list(seq.tokens),
                                logprobs=logprobs,
                            ))
                return _sample_response(
                    sequences=tinker_sequences,
                    prompt_logprobs=responses[0].prompt_logprobs,
                    topk_prompt_logprobs=responses[0].topk_prompt_logprobs,
                )
            except Exception:
                logger.error(traceback.format_exc())
                raise

        input_tokens = len(body.prompt.to_ints())
        return await self.schedule_task(
            _do_sample,
            token=token,
            input_tokens=input_tokens,
            task_type='sample',
        )
