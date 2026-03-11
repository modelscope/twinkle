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
    async_upload: bool = True

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


class HeartbeatRequest(BaseModel):
    adapter_name: str


class CalculateMetricRequest(BaseModel):
    adapter_name: str
    is_training: bool = True

    class Config:
        extra = 'allow'


class GetStateDictRequest(BaseModel):
    adapter_name: str

    class Config:
        extra = 'allow'


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class ModelResult(BaseModel):
    """Generic single-value result wrapper returned by most model endpoints."""
    result: Any


class ForwardResponse(ModelResult):
    """Response for /forward and /forward_only endpoints."""
    pass


class ForwardBackwardResponse(ModelResult):
    """Response for /forward_backward endpoint."""
    pass


class BackwardResponse(ModelResult):
    """Response for /backward endpoint."""
    pass


class StepResponse(ModelResult):
    """Response for /step (optimizer step) endpoint."""
    pass


class ZeroGradResponse(ModelResult):
    """Response for /zero_grad endpoint."""
    pass


class LrStepResponse(ModelResult):
    """Response for /lr_step endpoint."""
    pass


class SetLossResponse(ModelResult):
    """Response for /set_loss endpoint."""
    pass


class ClipGradNormResponse(ModelResult):
    """Response for /clip_grad_norm endpoint."""
    pass


class SetOptimizerResponse(ModelResult):
    """Response for /set_optimizer endpoint."""
    pass


class SetLrSchedulerResponse(ModelResult):
    """Response for /set_lr_scheduler endpoint."""
    pass


class SaveResponse(ModelResult):
    """Response for /save endpoint."""
    pass


class LoadResponse(ModelResult):
    """Response for /load endpoint."""
    pass


class SetTemplateResponse(ModelResult):
    """Response for /set_template endpoint."""
    pass


class SetProcessorResponse(ModelResult):
    """Response for /set_processor endpoint."""
    pass


class CalculateLossResponse(ModelResult):
    """Response for /calculate_loss endpoint."""
    pass


class CalculateMetricResponse(ModelResult):
    """Response for /calculate_metric endpoint."""
    pass


class GetTrainConfigsResponse(ModelResult):
    """Response for /get_train_configs endpoint."""
    pass


class GetStateDictResponse(ModelResult):
    """Response for /get_state_dict endpoint."""
    pass


class UploadToHubResponse(BaseModel):
    """Response for /upload_to_hub endpoint."""
    status: Optional[str] = None
    message: Optional[str] = None

    class Config:
        extra = 'allow'


class CreateResponse(BaseModel):
    """Response for /create endpoint."""
    status: str = 'ok'


class AddAdapterResponse(BaseModel):
    """Response for /add_adapter_to_model endpoint."""
    status: str = 'ok'
    adapter_name: str


class HeartbeatResponse(BaseModel):
    """Response for /heartbeat endpoint."""
    status: str = 'ok'
