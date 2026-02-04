# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Tinker-compatible sampler (inference) server.

This module provides a Ray Serve deployment for distributed text generation/inference.
It supports:
1. VLLM and Torch sampler backends
2. LoRA adapter loading via adapter URIs
3. Multi-user inference with rate limiting
4. Flexible sampling parameters
"""
import os
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from ray import serve
from tinker import types

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.server.utils.validation import verify_request_token
from twinkle.server.utils.state import get_server_state, ServerStateProxy
from twinkle.sampler.types import SamplingParams as TwinkleSamplingParams
from twinkle.utils.logger import get_logger

from .common.task_queue import TaskQueueMixin, TaskQueueConfig

logger = get_logger()


def build_sampler_app(model_id: str,
                      nproc_per_node: int,
                      device_group: Dict[str, Any],
                      device_mesh: Dict[str, Any],
                      deploy_options: Dict[str, Any],
                      sampler_type: str = 'vllm',
                      engine_args: Optional[Dict[str, Any]] = None,
                      queue_config: Optional[Dict[str, Any]] = None,
                      **kwargs):
    """Build a sampler application for tinker-compatible inference.
    
    This factory function creates a Ray Serve deployment that manages a sampler
    (inference engine) with support for LoRA adapters and rate limiting.
    
    Args:
        model_id: Model identifier (e.g., "ms://Qwen/Qwen2.5-0.5B-Instruct")
        nproc_per_node: Number of processes per node
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for parallelism
        deploy_options: Ray Serve deployment options
        sampler_type: Type of sampler to use ('vllm' or 'torch')
        engine_args: Additional engine arguments for the sampler
        queue_config: Task queue configuration dict (rps_limit, tps_limit, etc.)
        **kwargs: Additional arguments passed to the sampler
        
    Returns:
        Ray Serve deployment bound with configuration
    """
    app = FastAPI()

    @app.middleware('http')
    async def verify_token(request: Request, call_next):
        """Middleware to verify authentication token for all requests."""
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name='SamplerManagement')
    @serve.ingress(app)
    class SamplerManagement(TaskQueueMixin):
        """Sampler management service for text generation inference.
        
        This class manages:
        - VLLM or Torch sampler initialization and lifecycle
        - Inference requests with LoRA adapter support
        - Rate limiting via task queue
        - Sampling parameter conversion between Tinker and Twinkle formats
        """

        def __init__(self, nproc_per_node: int, device_group: Dict[str, Any],
                     device_mesh: Dict[str, Any], sampler_type: str = 'vllm',
                     engine_args: Optional[Dict[str, Any]] = None,
                     queue_config: Optional[Dict[str, Any]] = None, **kwargs):
            """Initialize the sampler management service.
            
            Args:
                nproc_per_node: Number of processes per node
                device_group: Device group configuration
                device_mesh: Device mesh configuration for parallelism
                sampler_type: Type of sampler ('vllm' or 'torch')
                engine_args: Additional engine arguments for sampler
                queue_config: Task queue configuration dict
                **kwargs: Additional sampler initialization arguments
            """
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(mode='ray',
                               nproc_per_node=nproc_per_node,
                               groups=[self.device_group],
                               lazy_collect=False)
            self.device_mesh = DeviceMesh(**device_mesh)
            self.sampler_type = sampler_type
            
            # Initialize sampler based on type
            if sampler_type == 'vllm':
                from twinkle.sampler import VLLMSampler
                sampler_kwargs = engine_args or {}
                self.sampler = VLLMSampler(
                    model_id=model_id,
                    engine_args=sampler_kwargs,
                    device_mesh=self.device_mesh,
                    **{k: v for k, v in kwargs.items() if k not in ['engine_args']}
                )
            else:  # torch sampler
                from twinkle.sampler import TorchSampler
                self.sampler = TorchSampler(
                    model_id=model_id,
                    device_mesh=self.device_mesh,
                    **kwargs
                )
            
            self.state: ServerStateProxy = get_server_state()
            self._init_task_queue(TaskQueueConfig.from_dict(queue_config))

        @app.post('/asample')
        async def asample(
                self, request: Request,
                body: types.SampleRequest) -> types.UntypedAPIFuture:
            """Execute text generation (inference).
            
            This endpoint:
            1. Extracts prompt token IDs from the request
            2. Determines adapter URI from model_path if provided
            3. Converts Tinker sampling params to Twinkle format
            4. Calls the sampler engine to generate text
            5. Converts results back to Tinker format
            
            Args:
                request: FastAPI request with auth token
                body: SampleRequest with prompt, sampling params, and adapter info
                
            Returns:
                UntypedAPIFuture wrapping SampleResponse with generated sequences
            """
            async def _do_sample():
                try:
                    # Extract prompt token IDs from ModelInput
                    prompt_token_ids = body.prompt.to_ints()
                    
                    # Determine adapter URI from model_path
                    adapter_uri = body.model_path if body.model_path else None
                    
                    # Convert tinker SamplingParams to twinkle SamplingParams if needed
                    sampling_params = None
                    if body.sampling_params:
                        sampling_params = TwinkleSamplingParams(
                            max_tokens=body.sampling_params.max_tokens or 256,
                            temperature=body.sampling_params.temperature or 1.0,
                            top_p=body.sampling_params.top_p,
                            top_k=body.sampling_params.top_k,
                            stop=body.sampling_params.stop,
                        )
                    
                    response = await self.sampler.engine.sample(
                        prompt_token_ids=prompt_token_ids,
                        sampling_params=sampling_params,
                        num_samples=body.num_samples or 1,
                        logprobs=True,
                        include_prompt_logprobs=body.prompt_logprobs or False,
                        topk_prompt_logprobs=body.topk_prompt_logprobs or 0,
                        adapter_uri=adapter_uri,
                    )
                    
                    # Convert twinkle SampleResponse to tinker types.SampleResponse
                    tinker_sequences = [
                        types.SampledSequence(
                            stop_reason=seq.stop_reason,
                            tokens=list(seq.tokens),
                            logprobs=list(seq.logprobs) if seq.logprobs else None,
                        )
                        for seq in response.sequences
                    ]
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

            # Calculate input tokens for rate limiting
            input_tokens = len(body.prompt.to_ints())
            return await self.schedule_task(
                _do_sample(),
                token=request.state.token,
                input_tokens=input_tokens,
            )

    return SamplerManagement.options(**deploy_options).bind(
        nproc_per_node, device_group, device_mesh, sampler_type, engine_args, queue_config, **kwargs)
