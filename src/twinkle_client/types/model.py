# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Pydantic request/response models for twinkle model management endpoints.

These models are used by both the server-side handler and the twinkle client.
"""
from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class CreateRequest(BaseModel):

    class Config:
        extra = 'allow'


class ForwardRequest(BaseModel):
    inputs: Any
    adapter_name: str

    class Config:
        extra = 'allow'


class ForwardOnlyRequest(BaseModel):
    inputs: Any
    adapter_name: Optional[str] = None

    class Config:
        extra = 'allow'


class AdapterRequest(BaseModel):
    adapter_name: str

    class Config:
        extra = 'allow'


class SetLossRequest(BaseModel):
    loss_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class SetOptimizerRequest(BaseModel):
    optimizer_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class SetLrSchedulerRequest(BaseModel):
    scheduler_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class SaveRequest(BaseModel):
    adapter_name: str
    save_optimizer: bool = False
    name: Optional[str] = None

    class Config:
        extra = 'allow'


class UploadToHubRequest(BaseModel):
    checkpoint_dir: str
    hub_model_id: str
    hub_token: Optional[str] = None
    async_upload: bool = False

    class Config:
        extra = 'allow'


class LoadRequest(BaseModel):
    adapter_name: str
    load_optimizer: bool = False
    name: str

    class Config:
        extra = 'allow'


class AddAdapterRequest(BaseModel):
    adapter_name: str
    config: str

    class Config:
        extra = 'allow'


class SetTemplateRequest(BaseModel):
    template_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class SetProcessorRequest(BaseModel):
    processor_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class CalculateMetricRequest(BaseModel):
    adapter_name: str
    is_training: bool = True

    class Config:
        extra = 'allow'


class GetStateDictRequest(BaseModel):
    adapter_name: str

    class Config:
        extra = 'allow'


class ClipGradAndStepRequest(BaseModel):
    adapter_name: str
    max_grad_norm: float = 1.0
    norm_type: int = 2

    class Config:
        extra = 'allow'


class ApplyPatchRequest(BaseModel):
    patch_cls: str
    adapter_name: str

    class Config:
        extra = 'allow'


class AddMetricRequest(BaseModel):
    metric_cls: str
    adapter_name: str
    is_training: Optional[bool] = None

    class Config:
        extra = 'allow'


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class OkResponse(BaseModel):
    """Response for endpoints whose underlying method returns None."""
    status: str = 'ok'


class ModelResult(BaseModel):
    """Generic single-value result wrapper returned by result-bearing endpoints."""
    result: Any


# --- Result-bearing responses ---

class ForwardResponse(BaseModel):
    """Response for /forward and /forward_only endpoints (returns ModelOutput)."""
    result: Any


class ForwardBackwardResponse(BaseModel):
    """Response for /forward_backward endpoint (returns ModelOutput)."""
    result: Any


class CalculateLossResponse(BaseModel):
    """Response for /calculate_loss endpoint (returns float)."""
    result: float


class ClipGradNormResponse(BaseModel):
    """Response for /clip_grad_norm endpoint (returns float as str)."""
    result: str


class GetTrainConfigsResponse(BaseModel):
    """Response for /get_train_configs endpoint (returns str)."""
    result: str


class GetStateDictResponse(BaseModel):
    """Response for /get_state_dict endpoint (returns Dict)."""
    result: Dict[str, Any]


class CalculateMetricResponse(BaseModel):
    """Response for /calculate_metric endpoint (returns Dict)."""
    result: Dict[str, Any]


class SaveResponse(BaseModel):
    """Response for /save endpoint (returns twinkle path + checkpoint dir)."""
    twinkle_path: str
    checkpoint_dir: Optional[str] = None


# --- Void responses (return None → OkResponse) ---

class BackwardResponse(OkResponse):
    """Response for /backward endpoint."""
    pass


class StepResponse(OkResponse):
    """Response for /step (optimizer step) endpoint."""
    pass


class ZeroGradResponse(OkResponse):
    """Response for /zero_grad endpoint."""
    pass


class LrStepResponse(OkResponse):
    """Response for /lr_step endpoint."""
    pass


class SetLossResponse(OkResponse):
    """Response for /set_loss endpoint."""
    pass


class SetOptimizerResponse(OkResponse):
    """Response for /set_optimizer endpoint."""
    pass


class SetLrSchedulerResponse(OkResponse):
    """Response for /set_lr_scheduler endpoint."""
    pass


class LoadResponse(OkResponse):
    """Response for /load endpoint."""
    pass


class SetTemplateResponse(OkResponse):
    """Response for /set_template endpoint."""
    pass


class SetProcessorResponse(OkResponse):
    """Response for /set_processor endpoint."""
    pass


class UploadToHubResponse(OkResponse):
    """Response for /upload_to_hub endpoint."""
    pass


class ClipGradAndStepResponse(OkResponse):
    """Response for /clip_grad_and_step endpoint."""
    pass


class ApplyPatchResponse(OkResponse):
    """Response for /apply_patch endpoint."""
    pass


class AddMetricResponse(OkResponse):
    """Response for /add_metric endpoint."""
    pass


# --- Other responses ---

class CreateResponse(BaseModel):
    """Response for /create endpoint."""
    status: str = 'ok'


class AddAdapterResponse(BaseModel):
    """Response for /add_adapter_to_model endpoint."""
    status: str = 'ok'
    adapter_name: str
