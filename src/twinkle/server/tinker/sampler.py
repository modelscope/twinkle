# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Tinker-compatible sampler (inference) server.

This module provides a Ray Serve deployment for distributed text generation/inference.
It supports:
1. vLLM and Torch sampler backends
2. LoRA adapter loading via adapter URIs
3. Multi-user inference with rate limiting
4. Flexible sampling parameters
"""
import os
import traceback
from fastapi import FastAPI, Request
from ray import serve
from tinker import types
from typing import Any, Dict, Optional

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import SamplingParams
from twinkle.server.utils.state import ServerStateProxy, get_server_state
from twinkle.server.utils.task_queue import TaskQueueConfig, TaskQueueMixin
from twinkle.server.utils.validation import verify_request_token
from twinkle.utils.logger import get_logger
from .common.io_utils import create_checkpoint_manager

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
        - vLLM or Torch sampler initialization and lifecycle
        - Inference requests with LoRA adapter support
        - Rate limiting via task queue
        - Sampling parameter conversion between Tinker and Twinkle formats
        """

        def __init__(self,
                     nproc_per_node: int,
                     device_group: Dict[str, Any],
                     device_mesh: Dict[str, Any],
                     sampler_type: str = 'vllm',
                     engine_args: Optional[Dict[str, Any]] = None,
                     queue_config: Optional[Dict[str, Any]] = None,
                     **kwargs):
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
            twinkle.initialize(
                mode='ray', nproc_per_node=nproc_per_node, groups=[self.device_group], lazy_collect=False)
            if 'mesh_dim_names' in device_mesh:
                self.device_mesh = DeviceMesh(**device_mesh)
            else:
                self.device_mesh = DeviceMesh.from_sizes(**device_mesh)
            self.sampler_type = sampler_type
            replica_context = serve.get_replica_context()
            replica_id = replica_context.replica_id.unique_id

            # Initialize sampler based on type
            if sampler_type == 'vllm':
                from twinkle.sampler import vLLMSampler
                sampler_kwargs = engine_args or {}
                self.sampler = vLLMSampler(
                    model_id=model_id,
                    engine_args=sampler_kwargs,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    instance_id=replica_id,
                    **{
                        k: v
                        for k, v in kwargs.items() if k not in ['engine_args']
                    })
            else:  # torch sampler
                from twinkle.sampler import TorchSampler
                self.sampler = TorchSampler(model_id=model_id, device_mesh=self.device_mesh, **kwargs)
            self.sampler.set_template('Template', model_id=model_id)
            self.state: ServerStateProxy = get_server_state()
            self._init_task_queue(TaskQueueConfig.from_dict(queue_config))

        @app.post('/asample')
        async def asample(self, request: Request, body: types.SampleRequest) -> types.UntypedAPIFuture:
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
                    prompt_inputs = {'input_ids': body.prompt.to_ints()}

                    # Get model_path: use body.model_path or look up from sampling session
                    model_path = body.model_path
                    if not model_path and body.sampling_session_id:
                        session = self.state.get_sampling_session(body.sampling_session_id)
                        if session:
                            model_path = session.get('model_path')

                    # Parse and resolve adapter URI from model_path
                    adapter_uri = None
                    if model_path:
                        token = request.state.token
                        checkpoint_manager = create_checkpoint_manager(token)
                        adapter_name, adapter_uri = checkpoint_manager.parse_adapter_uri(model_path)

                    # Validate adapter URI existence if provided
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

                    # Only request logprobs when the client asks for them. Some backends may
                    # return None entries in logprobs, which breaks pydantic validation.
                    response = self.sampler.sample(
                        inputs=[prompt_inputs] * body.num_samples,  # For speed up
                        sampling_params=sampling_params,
                        adapter_path=adapter_uri,
                        # adapter_name=adapter_name,
                    )

                    # Convert twinkle SampleResponse to tinker types.SampleResponse
                    tinker_sequences = []
                    for seq in response.sequences:
                        logprobs = None
                        if seq.logprobs is not None:
                            if any(lp is None for lp in seq.logprobs):
                                # Fix: backend can emit None logprobs for some tokens, which triggers
                                # pydantic "Input should be a valid number" errors in SampleResponse.
                                # We drop the field to keep the response valid.
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

            # Calculate input tokens for rate limiting
            input_tokens = len(body.prompt.to_ints())
            return await self.schedule_task(
                _do_sample,
                token=request.state.token,
                input_tokens=input_tokens,
                task_type='sample',
            )

    return SamplerManagement.options(**deploy_options).bind(nproc_per_node, device_group, device_mesh, sampler_type,
                                                            engine_args, queue_config, **kwargs)
