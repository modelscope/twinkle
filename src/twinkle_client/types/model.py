# Copyright (c) ModelScope Contributors. All rights reserved.
"""Request / response models for the twinkle-native model endpoints.

One declaration per endpoint, shared by Twinkle_Client and the server handler, so
there is a single answer to "what may this endpoint receive". Every field carries a
role (see :mod:`twinkle_client.types.base`):

- plain fields are **control** fields: the handler consumes them or passes them as a
  named argument, and they are never re-forwarded through ``**backend_kwargs``;
- :func:`backend_kwarg` / :func:`backend_only` fields are forwarded to the backend
  when their value is not ``None``;
- :func:`passthrough` fields are declared dicts whose keys are dynamic (plugin
  constructor / loss arguments) and are flattened into the backend kwargs.

Requests are strict: an unknown top-level field is a typo and fails with 422 before
the task is enqueued. That is only safe because dynamic parameters have a declared
home -- the passthrough regions -- so strictness never blocks a legitimate
user-supplied argument.
"""
from __future__ import annotations

from pydantic import Field, JsonValue, field_validator, model_validator
from typing import Any, Dict, List, Optional, Union

from .base import ResponseModel, StrictRequest, backend_kwarg, backend_only, passthrough
from .component import DataRef
from .data import WireInputBatch


class CreateRequest(StrictRequest):
    """Body of ``POST /twinkle/create``: a session-establishing no-op."""


# --------------------------------------------------------------------------- #
# Control-plane requests
# --------------------------------------------------------------------------- #


class AdapterRequest(StrictRequest):
    """The shared shape of an adapter-scoped operation.

    ``seq_id`` is an idempotency key, not a backend parameter: the submit shell
    claims ``(session, adapter, seq_id)`` before enqueueing so a retried
    gradient-mutating call is applied at most once. It must stay a declared field --
    under ``extra='forbid'`` an undeclared ``seq_id`` would be rejected outright,
    which would silently disable that dedup.
    """

    adapter_name: str
    seq_id: Optional[int] = None
    gradient_accumulation_steps: Optional[int] = backend_kwarg(default=None, ge=1)


class StepRequest(AdapterRequest):
    """Body of ``POST /twinkle/step``."""

    optim_params: Optional[Dict[str, JsonValue]] = backend_kwarg(default=None)


class LrStepRequest(AdapterRequest):
    """Body of ``POST /twinkle/lr_step``."""

    # ``OptimizerParamScheduler.step(increment=...)``; the transformers scheduler has
    # no equivalent knob.
    increment: Optional[int] = backend_only('megatron', default=None, ge=0)


class ClipGradNormRequest(AdapterRequest):
    """Body of ``POST /twinkle/clip_grad_norm``.

    Bound to its own model rather than the bare :class:`AdapterRequest`: the endpoint
    has always read these two values, and sharing a model with the parameterless ops
    meant the schema could not say so.
    """

    max_grad_norm: float = Field(default=1.0, gt=0)
    norm_type: int = Field(default=2, gt=0)


class ClipGradAndStepRequest(ClipGradNormRequest):
    """Body of ``POST /twinkle/clip_grad_and_step``."""

    optim_params: Optional[Dict[str, JsonValue]] = backend_kwarg(default=None)


class CalculateMetricRequest(StrictRequest):
    """Body of ``POST /twinkle/calculate_metric``."""

    adapter_name: str
    is_training: bool = True


# --------------------------------------------------------------------------- #
# Inline forward family
#
# Three endpoints, three models. They were one shared model, which meant the schema
# could not express that only the gradient-mutating variants take ``seq_id`` or that
# ``forward_only`` does not need an adapter -- the handler had to carry that
# knowledge instead, as a second source of truth.
#
# ``ForwardBackwardRequest`` is deliberately NOT the name of the fwd-bwd model:
# ``tinker.types.ForwardBackwardRequest`` already exists and two handlers import a
# module named ``types`` from each package, so a same-named model would be
# distinguishable only by import alias.
# --------------------------------------------------------------------------- #


class _InlineForwardBase(StrictRequest):
    """Fields common to the three inline forward endpoints."""

    inputs: WireInputBatch
    task: Optional[str] = backend_kwarg(default=None)
    temperature: Optional[float] = backend_kwarg(default=None, gt=0)
    return_logits: Optional[bool] = backend_kwarg(default=None)
    micro_batch_size: Optional[int] = backend_kwarg(default=None, ge=1)
    gradient_accumulation_steps: Optional[int] = backend_kwarg(default=None, ge=1)
    # Read only by the transformers backend.
    sampling_masks: Optional[JsonValue] = backend_only('transformers', default=None)
    router_replay_action: Optional[str] = backend_only('transformers', default=None)
    # Loss inputs (``advantages`` / ``old_logps`` / ``ref_outputs`` / ...). Their key
    # set is decided by the configured Loss, so they get a declared dict rather than
    # top-level fields; the flattening in ``backend_kwargs`` keeps the backend call
    # shape identical to before.
    loss_kwargs: Dict[str, JsonValue] = passthrough()


class ForwardRequest(_InlineForwardBase):
    """Body of ``POST /twinkle/forward``: keeps the graph, mutates no gradients."""

    adapter_name: str
    disable_lora: Optional[bool] = backend_kwarg(default=None)


class ForwardOnlyRequest(_InlineForwardBase):
    """Body of ``POST /twinkle/forward_only``: no graph, no gradients.

    ``adapter_name`` is optional here -- a reference forward may run against the base
    weights -- and there is no ``seq_id`` because nothing is mutated to be idempotent
    about.
    """

    adapter_name: Optional[str] = None
    disable_lora: Optional[bool] = backend_kwarg(default=None)


class ForwardBackwardTaskRequest(_InlineForwardBase):
    """Body of ``POST /twinkle/forward_backward``: accumulates gradients."""

    adapter_name: str
    seq_id: Optional[int] = None
    sync_gradients: Optional[bool] = backend_kwarg(default=None)
    loss_scale: Optional[float] = backend_kwarg(default=None)


# --------------------------------------------------------------------------- #
# Data-plane forward family
#
# ``input_refs`` / ``input_field`` / ``kwarg_fields`` are control fields: the handler
# resolves them into rows and bound kwargs. No wire schema applies to a ``DataRef`` --
# it is an opaque handle and the rows it points at never travel in this body.
# --------------------------------------------------------------------------- #


class DataPlaneForwardRequest(StrictRequest):
    """Body of the ``*_from_data_plane`` forward endpoints."""

    input_refs: List[DataRef] = Field(min_length=1)
    input_field: Optional[str] = None
    # Values are *field paths*, not parameter values, so this is not a passthrough
    # region: nothing in it is forwarded verbatim.
    kwarg_fields: Dict[str, str] = Field(default_factory=dict)
    adapter_name: str
    seq_id: Optional[int] = None
    task: Optional[str] = backend_kwarg(default=None)
    temperature: Optional[float] = backend_kwarg(default=None, gt=0)
    return_logits: Optional[bool] = backend_kwarg(default=None)
    disable_lora: Optional[bool] = backend_kwarg(default=None)
    micro_batch_size: Optional[int] = backend_kwarg(default=None, ge=1)
    gradient_accumulation_steps: Optional[int] = backend_kwarg(default=None, ge=1)
    loss_kwargs: Dict[str, JsonValue] = passthrough()


class DataPlaneForwardOnlyRequest(DataPlaneForwardRequest):
    """Body of ``POST /twinkle/forward_only_from_data_plane``."""

    output_ref: Optional[DataRef] = None
    output_fields: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_output(self) -> 'DataPlaneForwardOnlyRequest':
        if (self.output_ref is None) != (len(self.output_fields) == 0):
            raise ValueError('output_ref and output_fields must be configured together')
        return self


# --------------------------------------------------------------------------- #
# Plugin setters
#
# Each takes the plugin identifier as a control field (the handler passes it
# positionally) plus one passthrough region for the plugin's constructor arguments.
# The passthrough keys are forwarded to the plugin as given -- there is no spelling
# check against a sibling ``target``: signature reflection cannot see a parameter a
# plugin reads straight out of ``**kwargs`` (``InputProcessor`` does this with
# ``padding_side``), so any such check rejects valid requests. A misspelt argument
# therefore surfaces from the plugin itself.
# --------------------------------------------------------------------------- #


class SetLossRequest(StrictRequest):
    loss_cls: str
    adapter_name: str
    init_kwargs: Dict[str, JsonValue] = passthrough()


class SetOptimizerRequest(StrictRequest):
    optimizer_cls: str
    adapter_name: str
    init_kwargs: Dict[str, JsonValue] = passthrough()


class SetLrSchedulerRequest(StrictRequest):
    scheduler_cls: str
    adapter_name: str
    init_kwargs: Dict[str, JsonValue] = passthrough()


class SetTemplateRequest(StrictRequest):
    """Body of ``POST /twinkle/set_template``.

    No top-level ``model_id``: the backend always overrides it with its own
    ``tokenizer_id``, so a declared field would advertise a parameter that has no
    effect. Callers that pass ``model_id`` reach the template constructor through
    ``init_kwargs`` like any other template argument.
    """

    template_cls: str
    adapter_name: str
    init_kwargs: Dict[str, JsonValue] = passthrough()


class SetProcessorRequest(StrictRequest):
    processor_cls: str
    adapter_name: str
    init_kwargs: Dict[str, JsonValue] = passthrough()


class AddMetricRequest(StrictRequest):
    metric_cls: str
    adapter_name: str
    is_training: Optional[bool] = None
    init_kwargs: Dict[str, JsonValue] = passthrough()


class ApplyPatchRequest(StrictRequest):
    patch_cls: str
    adapter_name: str
    init_kwargs: Dict[str, JsonValue] = passthrough()


# --------------------------------------------------------------------------- #
# Checkpoint I/O and adapter lifecycle
# --------------------------------------------------------------------------- #


class SaveRequest(StrictRequest):
    adapter_name: str
    name: Optional[str] = None
    save_optimizer: bool = False
    is_sampler: bool = False  # If True, delete existing sampler weights before saving
    consumed_train_samples: Optional[int] = backend_kwarg(default=None, ge=0)
    merge_lora: Optional[bool] = backend_only('megatron', default=None)


class LoadRequest(StrictRequest):
    adapter_name: str
    name: str
    load_optimizer: bool = False
    no_load_optim: Optional[bool] = backend_only('megatron', default=None)
    no_load_rng: Optional[bool] = backend_only('megatron', default=None)
    strict: Optional[bool] = backend_only('transformers', default=None)


class ResumeFromCheckpointRequest(StrictRequest):
    """Body of ``POST /twinkle/resume_from_checkpoint``."""

    name: str
    adapter_name: str = ''
    resume_only_model: bool = False


class AddAdapterRequest(StrictRequest):
    adapter_name: str
    # ``config`` is None for full-parameter training (no LoRA adapter) and a
    # serialized LoraConfig string for LoRA training.
    config: Optional[str] = None
    save_dir: Optional[str] = None
    gradient_accumulation_steps: Optional[int] = backend_kwarg(default=None, ge=1)
    init_kwargs: Dict[str, JsonValue] = passthrough()


class UploadToHubRequest(StrictRequest):
    """Body of ``POST /twinkle/upload_to_hub``.

    No ``async_upload``: the server always runs the upload as a background task and
    the client waits through the future layer, so the flag could only ever be ignored.
    """

    checkpoint_dir: Union[str, Dict[str, Any]]
    hub_model_id: str
    hub_token: Optional[str] = None

    @field_validator('checkpoint_dir', mode='before')
    @classmethod
    def extract_checkpoint_dir(cls, v):
        """Accept a ``save`` response dict and take its twinkle path.

        Raises a validation error -- not ``KeyError`` -- when the key is absent, so a
        wrong-shaped dict is a 422 naming the missing key instead of a 500.
        """
        if isinstance(v, dict):
            if 'twinkle_path' not in v:
                raise ValueError("checkpoint_dir dict must contain 'twinkle_path'")
            return v['twinkle_path']
        return v


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #


class OkResponse(ResponseModel):
    """Response for endpoints whose underlying method returns None."""
    status: str = 'ok'


class ModelResult(ResponseModel):
    """Generic single-value result wrapper returned by result-bearing endpoints."""
    result: Any


# --- Result-bearing responses ---


class ForwardResponse(ResponseModel):
    """Response for /forward and /forward_only endpoints (returns ModelOutput)."""
    result: Any


class ForwardBackwardResponse(ResponseModel):
    """Response for /forward_backward endpoint (returns ModelOutput)."""
    result: Any


class CalculateLossResponse(ResponseModel):
    """Response for /calculate_loss endpoint (returns float)."""
    result: float


class ClipGradNormResponse(ResponseModel):
    """Response for /clip_grad_norm endpoint (returns float as str)."""
    result: str


class GetTrainConfigsResponse(ResponseModel):
    """Response for /get_train_configs endpoint (returns str)."""
    result: str


class CalculateMetricResponse(ResponseModel):
    """Response for /calculate_metric endpoint (returns Dict)."""
    result: Dict[str, Any]


class SaveResponse(ResponseModel):
    """Response for /save endpoint (returns twinkle path + checkpoint dir)."""
    twinkle_path: str
    checkpoint_dir: Optional[str] = None


class TrainingProgressResponse(ResponseModel):
    """Response for /resume_from_checkpoint endpoint."""
    result: Dict[str, Any]


# --- Void responses (return None → OkResponse) ---


class BackwardResponse(OkResponse):
    """Response for /backward endpoint."""


class StepResponse(OkResponse):
    """Response for /step (optimizer step) endpoint."""


class ZeroGradResponse(OkResponse):
    """Response for /zero_grad endpoint."""


class LrStepResponse(OkResponse):
    """Response for /lr_step endpoint."""


class SetLossResponse(OkResponse):
    """Response for /set_loss endpoint."""


class SetOptimizerResponse(OkResponse):
    """Response for /set_optimizer endpoint."""


class SetLrSchedulerResponse(OkResponse):
    """Response for /set_lr_scheduler endpoint."""


class LoadResponse(OkResponse):
    """Response for /load endpoint."""


class SetTemplateResponse(OkResponse):
    """Response for /set_template endpoint."""


class SetProcessorResponse(OkResponse):
    """Response for /set_processor endpoint."""


class ClipGradAndStepResponse(OkResponse):
    """Response for /clip_grad_and_step endpoint."""


class ApplyPatchResponse(OkResponse):
    """Response for /apply_patch endpoint."""


class AddMetricResponse(OkResponse):
    """Response for /add_metric endpoint."""


# --- Other responses ---


class CreateResponse(ResponseModel):
    """Response for /create endpoint."""
    status: str = 'ok'
