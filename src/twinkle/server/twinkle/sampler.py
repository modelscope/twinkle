# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle sampler (inference) server.

This module provides a Ray Serve deployment for distributed text generation/inference.
It supports:
1. vLLM and Torch sampler backends
2. LoRA adapter loading via adapter URIs (twinkle:// paths or local paths)
3. Multi-user inference with adapter lifecycle management
4. Flexible sampling parameters
"""
import traceback
"""
Twinkle sampler (inference) server.

This module provides a Ray Serve deployment for distributed text generation/inference.
It supports:
1. vLLM and Torch sampler backends
2. LoRA adapter loading via adapter URIs (twinkle:// paths or local paths)
3. Multi-user inference with adapter lifecycle management
4. Flexible sampling parameters
"""
import traceback
from typing import Dict, Any, List, Optional, Union

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from ray import serve

import twinkle
from twinkle import DeviceGroup, DeviceMesh
from twinkle.data_format import Trajectory, InputFeature, SamplingParams
from twinkle.server.utils.adapter_manager import AdapterManagerMixin
from twinkle.server.utils.validation import verify_request_token, get_token_from_request
from twinkle.data_format import Trajectory, InputFeature, SamplingParams
from twinkle.server.utils.adapter_manager import AdapterManagerMixin
from twinkle.server.utils.validation import verify_request_token, get_token_from_request
from twinkle.server.utils.state import get_server_state, ServerStateProxy
from twinkle.utils.logger import get_logger

logger = get_logger()


# ----- Request/Response Models -----

class SampleRequest(BaseModel):
    """Request body for the /sample endpoint."""
    inputs: Any = Field(..., description="List of Trajectory or InputFeature dicts")
    sampling_params: Optional[Dict[str, Any]] = Field(
        None, description="Sampling parameters (max_tokens, temperature, etc.)")
    adapter_name: str = Field('', description="Adapter name for LoRA inference")
    adapter_uri: Optional[str] = Field(
        None, description="Adapter URI (twinkle:// path or local path) for LoRA inference")
    num_samples: int = Field(1, description="Number of completions to generate per prompt")


class SampleResponseModel(BaseModel):
    """Response body for the /sample endpoint."""
    sequences: List[Dict[str, Any]] = Field(
        ..., description="List of sampled sequences, each with tokens, logprobs, stop_reason")
    prompt_logprobs: Optional[List[Optional[float]]] = None
    topk_prompt_logprobs: Optional[List[Optional[List]]] = None


class SetTemplateRequest(BaseModel):
    """Request body for the /set_template endpoint."""
    template_cls: str = Field(..., description="Template class name (e.g. 'Template')")
    adapter_name: str = Field('', description="Adapter name to associate the template with")

    class Config:
        extra = "allow"


class SetTemplateResponse(BaseModel):
    """Response body for the /set_template endpoint."""
    status: str = "ok"


class AddAdapterRequest(BaseModel):
    """Request body for the /add_adapter_to_sampler endpoint."""
    adapter_name: str = Field(..., description="Name of the adapter to add")
    config: Any = Field(..., description="LoRA configuration dict")


class AddAdapterResponse(BaseModel):
    """Response body for the /add_adapter_to_sampler endpoint."""
    status: str = "ok"
    adapter_name: str


class SyncWeightsRequest(BaseModel):
    """Request body for the /sync_weights endpoint."""
    state_dict: Dict[str, Any] = Field(..., description="Model state dict to sync")
    adapter_name: str = Field('', description="Adapter name for LoRA weight sync")


class SyncWeightsResponse(BaseModel):
    """Response body for the /sync_weights endpoint."""
    status: str = "ok"


class HeartbeatRequest(BaseModel):
    """Request body for the /heartbeat endpoint."""
    adapter_name: str = Field(..., description="Adapter name to keep alive")


class HeartbeatResponse(BaseModel):
    """Response body for the /heartbeat endpoint."""
    status: str = "ok"


class CreateResponse(BaseModel):
    """Response body for the /create endpoint."""
    status: str = "ok"


# ----- Application Builder -----
from twinkle.utils.logger import get_logger

logger = get_logger()


# ----- Request/Response Models -----

class SampleRequest(BaseModel):
    """Request body for the /sample endpoint."""
    inputs: Any = Field(..., description="List of Trajectory or InputFeature dicts")
    sampling_params: Optional[Dict[str, Any]] = Field(
        None, description="Sampling parameters (max_tokens, temperature, etc.)")
    adapter_name: str = Field('', description="Adapter name for LoRA inference")
    adapter_uri: Optional[str] = Field(
        None, description="Adapter URI (twinkle:// path or local path) for LoRA inference")
    num_samples: int = Field(1, description="Number of completions to generate per prompt")


class SampleResponseModel(BaseModel):
    """Response body for the /sample endpoint."""
    sequences: List[Dict[str, Any]] = Field(
        ..., description="List of sampled sequences, each with tokens, logprobs, stop_reason")
    prompt_logprobs: Optional[List[Optional[float]]] = None
    topk_prompt_logprobs: Optional[List[Optional[List]]] = None


class SetTemplateRequest(BaseModel):
    """Request body for the /set_template endpoint."""
    template_cls: str = Field(..., description="Template class name (e.g. 'Template')")
    adapter_name: str = Field('', description="Adapter name to associate the template with")

    class Config:
        extra = "allow"


class SetTemplateResponse(BaseModel):
    """Response body for the /set_template endpoint."""
    status: str = "ok"


class AddAdapterRequest(BaseModel):
    """Request body for the /add_adapter_to_sampler endpoint."""
    adapter_name: str = Field(..., description="Name of the adapter to add")
    config: Any = Field(..., description="LoRA configuration dict")


class AddAdapterResponse(BaseModel):
    """Response body for the /add_adapter_to_sampler endpoint."""
    status: str = "ok"
    adapter_name: str


class SyncWeightsRequest(BaseModel):
    """Request body for the /sync_weights endpoint."""
    state_dict: Dict[str, Any] = Field(..., description="Model state dict to sync")
    adapter_name: str = Field('', description="Adapter name for LoRA weight sync")


class SyncWeightsResponse(BaseModel):
    """Response body for the /sync_weights endpoint."""
    status: str = "ok"


class HeartbeatRequest(BaseModel):
    """Request body for the /heartbeat endpoint."""
    adapter_name: str = Field(..., description="Adapter name to keep alive")


class HeartbeatResponse(BaseModel):
    """Response body for the /heartbeat endpoint."""
    status: str = "ok"


class CreateResponse(BaseModel):
    """Response body for the /create endpoint."""
    status: str = "ok"


# ----- Application Builder -----

def build_sampler_app(model_id: str,
                      nproc_per_node: int = 1,
                      device_group: Dict[str, Any] = None,
                      device_mesh: Dict[str, Any] = None,
                      deploy_options: Dict[str, Any] = None,
                      sampler_type: str = 'vllm',
                      engine_args: Optional[Dict[str, Any]] = None,
                      adapter_config: Optional[Dict[str, Any]] = None,
                      **kwargs):
    """Build a sampler application for text generation inference.

    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        nproc_per_node: Number of GPU processes per node
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for parallelism
        deploy_options: Ray Serve deployment options
        sampler_type: Type of sampler to use ('vllm' or 'torch')
        engine_args: Additional engine arguments for the sampler
        adapter_config: Adapter lifecycle config (adapter_timeout, per_token_adapter_limit)
        **kwargs: Additional arguments passed to the sampler

    Returns:
        Ray Serve deployment bound with configuration
    """
    app = FastAPI(
        title="Twinkle Sampler",
        description="REST API for distributed text generation inference",
        version="1.0.0"
    )
    """Build a sampler application for text generation inference.

    Args:
        model_id: Model identifier (e.g., "Qwen/Qwen2.5-7B-Instruct")
        nproc_per_node: Number of GPU processes per node
        device_group: Device group configuration dict
        device_mesh: Device mesh configuration dict for parallelism
        deploy_options: Ray Serve deployment options
        sampler_type: Type of sampler to use ('vllm' or 'torch')
        engine_args: Additional engine arguments for the sampler
        adapter_config: Adapter lifecycle config (adapter_timeout, per_token_adapter_limit)
        **kwargs: Additional arguments passed to the sampler

    Returns:
        Ray Serve deployment bound with configuration
    """
    app = FastAPI(
        title="Twinkle Sampler",
        description="REST API for distributed text generation inference",
        version="1.0.0"
    )

    @app.middleware("http")
    async def verify_token(request: Request, call_next):
        return await verify_request_token(request=request, call_next=call_next)

    @serve.deployment(name="SamplerManagement")
    @serve.ingress(app)
    class SamplerManagement(AdapterManagerMixin):
        """Sampler management service for text generation inference.

        Manages:
        - vLLM or Torch sampler initialization and lifecycle
        - Adapter lifecycle via AdapterManagerMixin
        - Inference requests with LoRA adapter support
        - Template configuration for trajectory encoding
        """

        def __init__(self, nproc_per_node: int, device_group: Dict[str, Any],
                     device_mesh: Dict[str, Any], sampler_type: str = 'vllm',
                     engine_args: Optional[Dict[str, Any]] = None,
                     adapter_config: Optional[Dict[str, Any]] = None, **kwargs):
            self.device_group = DeviceGroup(**device_group)
            twinkle.initialize(mode='ray',
                               nproc_per_node=nproc_per_node,
                               groups=[self.device_group],
                               lazy_collect=False)
            self.device_mesh = DeviceMesh(**device_mesh)
            self.sampler_type = sampler_type

            # Initialize sampler based on type
            if sampler_type == 'vllm':
                from twinkle.sampler import vLLMSampler
                sampler_kwargs = engine_args or {}
                self.sampler = vLLMSampler(
                    model_id=model_id,
                    engine_args=sampler_kwargs,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    **{k: v for k, v in kwargs.items() if k not in ['engine_args']}
                )
            else:
                from twinkle.sampler import TorchSampler
                self.sampler = TorchSampler(
                    model_id=model_id,
                    device_mesh=self.device_mesh,
                    remote_group=self.device_group.name,
                    **kwargs
                )

            # Initialize state and adapter manager
            self.state: ServerStateProxy = get_server_state()
            _adapter_config = adapter_config or {}
            self._init_adapter_manager(**_adapter_config)
            self.start_adapter_countdown()

        def _on_adapter_expired(self, adapter_name: str, token: str) -> None:
            """Handle expired adapters by removing them from the sampler."""
            try:
                self.sampler.remove_adapter(adapter_name)
                logger.info(f"Removed expired adapter {adapter_name}")
                self.check_adapter_limit(token, False)
            except Exception as e:
                logger.warning(f"Failed to remove expired adapter {adapter_name}: {e}")
            _adapter_config = adapter_config or {}
            self._init_adapter_manager(**_adapter_config)
            self.start_adapter_countdown()

        def _on_adapter_expired(self, adapter_name: str, token: str) -> None:
            """Handle expired adapters by removing them from the sampler."""
            try:
                self.sampler.remove_adapter(adapter_name)
                logger.info(f"Removed expired adapter {adapter_name}")
                self.check_adapter_limit(token, False)
            except Exception as e:
                logger.warning(f"Failed to remove expired adapter {adapter_name}: {e}")

        @staticmethod
        def _get_adapter_name(request: Request, adapter_name: Optional[str]) -> Optional[str]:
            if adapter_name is None or adapter_name == '':
                return None
        def _get_adapter_name(request: Request, adapter_name: Optional[str]) -> Optional[str]:
            if adapter_name is None or adapter_name == '':
                return None
            return request.state.request_id + '-' + adapter_name

        @app.post("/create", response_model=CreateResponse)
        def create(self, request: Request) -> CreateResponse:
            """Health check / session creation endpoint."""
            return CreateResponse()

        @app.post("/sample", response_model=SampleResponseModel)
        def sample(self, request: Request, body: SampleRequest) -> SampleResponseModel:
            """Sample completions from the model.

            Supports:
            - Trajectory inputs (messages-based, requires template to be set)
            - InputFeature inputs (pre-tokenized input_ids)
            - LoRA adapter via adapter_name or adapter_uri (twinkle:// path)
            - Multiple completions per prompt via num_samples
            """
            try:
                # Resolve adapter
                adapter_path = None
                adapter_name = body.adapter_name or ''
                full_adapter_name = self._get_adapter_name(request, adapter_name) or ''

                if body.adapter_uri:
                    from .common.io_utils import create_checkpoint_manager
                    token = get_token_from_request(request)
                    checkpoint_manager = create_checkpoint_manager(token)
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

                # Convert to response model
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

        @app.post("/set_template", response_model=SetTemplateResponse)
        def set_template(self, request: Request, body: SetTemplateRequest) -> SetTemplateResponse:
            """Set the chat template for encoding Trajectory inputs."""
            full_adapter_name = self._get_adapter_name(request, body.adapter_name) or ''
            extra_kwargs = body.model_extra or {}
            self.sampler.set_template(body.template_cls, adapter_name=full_adapter_name, **extra_kwargs)
            return SetTemplateResponse()

        @app.post("/add_adapter_to_sampler", response_model=AddAdapterResponse)
        def add_adapter_to_sampler(self, request: Request, body: AddAdapterRequest) -> AddAdapterResponse:
            """Add a LoRA adapter to the sampler."""
            assert body.adapter_name, 'You need to specify a valid `adapter_name`'
            full_adapter_name = self._get_adapter_name(request, body.adapter_name)
            token = get_token_from_request(request)

            from peft import LoraConfig
            config = LoraConfig(**body.config) if isinstance(body.config, dict) else body.config

            with self._adapter_lock:
                self.sampler.add_adapter_to_sampler(full_adapter_name, config)

            self.register_adapter(full_adapter_name, token)
            allowed, reason = self.check_adapter_limit(token, True)
            if not allowed:
                raise RuntimeError(reason)

            return AddAdapterResponse(adapter_name=full_adapter_name)

        @app.post("/sync_weights", response_model=SyncWeightsResponse)
        def sync_weights(self, request: Request, body: SyncWeightsRequest) -> SyncWeightsResponse:
            """Synchronize model weights to the sampler."""
            full_adapter_name = self._get_adapter_name(request, body.adapter_name) or ''
            self.sampler.sync_weights(body.state_dict, full_adapter_name)
            return SyncWeightsResponse()
        @app.post("/create", response_model=CreateResponse)
        def create(self, request: Request) -> CreateResponse:
            """Health check / session creation endpoint."""
            return CreateResponse()

        @app.post("/sample", response_model=SampleResponseModel)
        def sample(self, request: Request, body: SampleRequest) -> SampleResponseModel:
            """Sample completions from the model.

            Supports:
            - Trajectory inputs (messages-based, requires template to be set)
            - InputFeature inputs (pre-tokenized input_ids)
            - LoRA adapter via adapter_name or adapter_uri (twinkle:// path)
            - Multiple completions per prompt via num_samples
            """
            try:
                # Resolve adapter
                adapter_path = None
                adapter_name = body.adapter_name or ''
                full_adapter_name = self._get_adapter_name(request, adapter_name) or ''

                if body.adapter_uri:
                    from .common.io_utils import create_checkpoint_manager
                    token = get_token_from_request(request)
                    checkpoint_manager = create_checkpoint_manager(token)
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

                # Convert to response model
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

        @app.post("/set_template", response_model=SetTemplateResponse)
        def set_template(self, request: Request, body: SetTemplateRequest) -> SetTemplateResponse:
            """Set the chat template for encoding Trajectory inputs."""
            full_adapter_name = self._get_adapter_name(request, body.adapter_name) or ''
            extra_kwargs = body.model_extra or {}
            self.sampler.set_template(body.template_cls, adapter_name=full_adapter_name, **extra_kwargs)
            return SetTemplateResponse()

        @app.post("/add_adapter_to_sampler", response_model=AddAdapterResponse)
        def add_adapter_to_sampler(self, request: Request, body: AddAdapterRequest) -> AddAdapterResponse:
            """Add a LoRA adapter to the sampler."""
            assert body.adapter_name, 'You need to specify a valid `adapter_name`'
            full_adapter_name = self._get_adapter_name(request, body.adapter_name)
            token = get_token_from_request(request)

            from peft import LoraConfig
            config = LoraConfig(**body.config) if isinstance(body.config, dict) else body.config

            with self._adapter_lock:
                self.sampler.add_adapter_to_sampler(full_adapter_name, config)

            self.register_adapter(full_adapter_name, token)
            allowed, reason = self.check_adapter_limit(token, True)
            if not allowed:
                raise RuntimeError(reason)

            return AddAdapterResponse(adapter_name=full_adapter_name)

        @app.post("/sync_weights", response_model=SyncWeightsResponse)
        def sync_weights(self, request: Request, body: SyncWeightsRequest) -> SyncWeightsResponse:
            """Synchronize model weights to the sampler."""
            full_adapter_name = self._get_adapter_name(request, body.adapter_name) or ''
            self.sampler.sync_weights(body.state_dict, full_adapter_name)
            return SyncWeightsResponse()

        @app.post("/heartbeat", response_model=HeartbeatResponse)
        def heartbeat(self, request: Request, body: HeartbeatRequest) -> HeartbeatResponse:
            """Keep an adapter alive by resetting its inactivity timer."""
            full_adapter_name = self._get_adapter_name(request, body.adapter_name)
            self.assert_adapter_exists(adapter_name=full_adapter_name)
            self.touch_adapter(full_adapter_name)
            return HeartbeatResponse()
        @app.post("/heartbeat", response_model=HeartbeatResponse)
        def heartbeat(self, request: Request, body: HeartbeatRequest) -> HeartbeatResponse:
            """Keep an adapter alive by resetting its inactivity timer."""
            full_adapter_name = self._get_adapter_name(request, body.adapter_name)
            self.assert_adapter_exists(adapter_name=full_adapter_name)
            self.touch_adapter(full_adapter_name)
            return HeartbeatResponse()

    return SamplerManagement.options(**deploy_options).bind(
        nproc_per_node, device_group, device_mesh, sampler_type, engine_args, adapter_config, **kwargs)
    return SamplerManagement.options(**deploy_options).bind(
        nproc_per_node, device_group, device_mesh, sampler_type, engine_args, adapter_config, **kwargs)
