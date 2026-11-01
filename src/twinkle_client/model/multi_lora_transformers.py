from __future__ import annotations

import itertools
import logging
import threading
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from peft import LoraConfig

from twinkle.protocol.types import model as model_types
from twinkle.protocol.types.component import DataRef
from twinkle_client._request_builder import build_request
from twinkle_client.exceptions import TwinkleHTTPError
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport

logger = logging.getLogger('twinkle_client')


def _data_refs(inputs: DataRef | list[DataRef]) -> list[dict[str, Any]]:
    """Encode one or more opaque references for a DataPlane model endpoint."""
    refs = [inputs] if isinstance(inputs, DataRef) else list(inputs)
    if not refs:
        raise ValueError('at least one DataRef is required')
    if not all(isinstance(item, DataRef) for item in refs):
        raise TypeError('data-plane model inputs must contain only DataRef values')
    return [item.model_dump() for item in refs]


class MultiLoraTransformersModel:
    """Client wrapper for TrainableModel that calls server HTTP endpoints.

    This client manages adapters and sends training/inference requests to the model server.
    The session authenticates requests; executed training work renews the idle deadline.

    Every method builds its endpoint's request model rather than a dict, so a
    misspelled or wrongly-typed argument fails here -- in the caller's own stack trace,
    with no request sent. Arguments that are not declared fields (loss inputs, plugin
    constructor arguments) are routed into that model's passthrough region, so public
    signatures stay ``**kwargs`` and callers are unchanged.
    """

    def __init__(
        self,
        model_id: str,
        *,
        transport: ClientTransport | None = None,
    ):
        """Initialize a model wrapper bound to one immutable request identity."""
        self._transport = capture_transport(transport)

        if '://' in model_id:
            model_id = model_id.split('://')[1]
        self.model_id = model_id
        self.server_url = f'{self._transport.context.base_url}/model/{model_id}/twinkle'
        self.adapter_name = None
        self._training_ids: dict[str, str] = {}
        self._creating: set[str] = set()
        self._training_lock = threading.Lock()
        # Per-client monotonic sequence for idempotent dedup of stateful training ops:
        # the server dedups on (training_id, session, operation, seq_id), so each update is
        # applied at most once. Reserved once per call and reused on retry.
        self._seq_counter = itertools.count(1)
        self._seq_lock = threading.Lock()
        # The server-side component is created lazily on first use rather than in
        # __init__, so constructing the wrapper performs no network I/O.
        self._created = False
        self._create_lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Request plumbing
    # ------------------------------------------------------------------ #

    def _ensure_created(self) -> None:
        """Create the server-side model component once, on first use.

        Deferred out of ``__init__`` so construction has no side effect: a failed
        ``create`` surfaces from the first operation instead of leaving a
        half-initialised object published. Idempotent and thread-safe.
        """
        if self._created:
            return
        with self._create_lock:
            if self._created:
                return
            response = self._transport.post(f'{self.server_url}/create', json_data={'training_handles': True})
            if response.json().get('training_handles') is not True:
                raise RuntimeError('Server does not support training handles; upgrade Twinkle server and SDK together.')
            self._created = True

    _SEQUENCED_ENDPOINTS = frozenset({
        'forward_backward',
        'forward_backward_from_data_plane',
        'backward',
        'step',
        'lr_step',
        'clip_grad_and_step',
    })

    def _training_values(self, values: dict[str, Any]) -> dict[str, Any]:
        """Choose the label and its handle together, under one lock."""
        with self._training_lock:
            name = values.get('adapter_name', self.adapter_name)
            training_id = values.get('training_id') or self._training_ids.get(name)
        if training_id is None:
            raise RuntimeError('Create this training with add_adapter_to_model on this object before using it.')
        return {**values, 'adapter_name': name, 'training_id': training_id}

    def _submit(self, endpoint: str, model_cls, response_cls, **values):
        """Validate once, then send with a stable handle and retry identity."""
        if issubclass(model_cls, model_types.TrainingRequest):
            values = self._training_values(values)
        if endpoint in self._SEQUENCED_ENDPOINTS:
            if 'seq_id' in values:
                raise TypeError('seq_id is managed by the client')
            values['seq_id'] = self._next_seq_id()
        body = build_request(model_cls, **values)
        self._ensure_created()
        response = self._transport.post_model(f'{self.server_url}/{endpoint}', body)
        return self._await_task(response, response_cls)

    def _await_task(self, response, model_cls):
        """Resolve a Submit_Endpoint response through the Client_Future_Layer.

        Blocks until the task is terminal and returns the deserialized ``model_cls``
        result (or ``None``), raising ``TaskFailedError`` on a failed terminal state.
        Keeps every public method's synchronous signature unchanged.
        """
        from twinkle_client._future import resolve_response
        return resolve_response(response, model_cls, transport=self._transport)

    def _next_seq_id(self) -> int:
        """Reserve the next monotonic seq_id for a stateful op (dedup key with session)."""
        with self._seq_lock:
            return next(self._seq_counter)

    # ------------------------------------------------------------------ #
    # Adapter lifecycle
    # ------------------------------------------------------------------ #

    def add_adapter_to_model(
        self,
        adapter_name: str,
        config: LoraConfig | Mapping[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Add a new adapter to the model.

        Pass a peft ``LoraConfig`` (or its dict form) for LoRA training against a
        LoRA-mode deployment. Pass ``config=None`` for full-parameter training
        against a ``train_mode: full`` deployment.
        """
        if isinstance(config, Mapping):
            from peft import LoraConfig

            config = LoraConfig(**config)
        save_dir = kwargs.pop('save_dir', None)
        if save_dir:
            save_dir = Path(save_dir).expanduser().resolve().as_posix()
        with self._training_lock:
            if adapter_name in self._creating:
                raise RuntimeError('A training with this adapter_name is already being created on this object.')
            if adapter_name in self._training_ids:
                raise RuntimeError('A training with this adapter_name already exists on this object; release it first.')
            self._creating.add(adapter_name)
        try:
            result = self._submit(
                'add_adapter_to_model',
                model_types.AddAdapterRequest,
                model_types.TrainingHandleResponse,
                client_create_id=uuid.uuid4().hex,
                adapter_name=adapter_name,
                config=config,
                save_dir=save_dir,
                **kwargs)
            with self._training_lock:
                self._training_ids[adapter_name] = result.training_id
                self.adapter_name = adapter_name
        finally:
            with self._training_lock:
                self._creating.discard(adapter_name)

    def remove_adapter(self, adapter_name: str | None = None) -> None:
        """Release one client-owned adapter from the training component."""
        name = adapter_name or self.adapter_name
        with self._training_lock:
            training_id = self._training_ids.get(name)
        try:
            self._submit('remove_adapter', model_types.AdapterRequest, None, adapter_name=name, training_id=training_id)
        except TwinkleHTTPError as exc:
            if exc.status_code == 404:
                self._forget_training(name, training_id)
            # A missing handle is unusable, but does not certify device release.
            raise
        self._forget_training(name, training_id)

    def _forget_training(self, name: str, training_id: str) -> None:
        """A late response must never erase a newer same-name binding."""
        with self._training_lock:
            if self._training_ids.get(name) == training_id:
                self._training_ids.pop(name, None)
                if name == self.adapter_name:
                    self.adapter_name = None

    # ------------------------------------------------------------------ #
    # Inline forward family
    # ------------------------------------------------------------------ #

    def forward(self, inputs: Any, **kwargs) -> model_types.ForwardResponse:
        """Execute forward pass on inline model inputs."""
        return self._submit('forward', model_types.ForwardRequest, model_types.ForwardResponse, inputs=inputs, **kwargs)

    def forward_only(self, inputs: Any, **kwargs) -> model_types.ForwardResponse:
        """Execute forward pass without gradient computation on inline inputs."""
        return self._submit(
            'forward_only', model_types.ForwardOnlyRequest, model_types.ForwardResponse, inputs=inputs, **kwargs)

    def forward_backward(self, inputs: Any, **kwargs) -> model_types.ForwardBackwardResponse:
        """Execute combined forward and backward pass on inline inputs."""
        return self._submit(
            'forward_backward',
            model_types.ForwardBackwardTaskRequest,
            model_types.ForwardBackwardResponse,
            inputs=inputs,
            **kwargs)

    def calculate_loss(self, **kwargs) -> model_types.CalculateLossResponse:
        """Calculate loss from model outputs."""
        return self._submit('calculate_loss', model_types.AdapterRequest, model_types.CalculateLossResponse, **kwargs)

    def get_train_configs(self, **kwargs) -> model_types.GetTrainConfigsResponse:
        """Get training configs."""
        return self._submit('get_train_configs', model_types.AdapterRequest, model_types.GetTrainConfigsResponse,
                            **kwargs)

    def backward(self, **kwargs) -> None:
        """Execute backward pass."""
        self._submit('backward', model_types.AdapterRequest, None, **kwargs)

    # ------------------------------------------------------------------ #
    # Data-plane forward family
    # ------------------------------------------------------------------ #

    def forward_from_data_plane(
        self,
        inputs: DataRef | list[DataRef],
        *,
        input_field: str | None = None,
        kwarg_fields: dict[str, str] | None = None,
        **kwargs,
    ) -> model_types.ForwardResponse:
        """Execute forward using rows referenced from the server DataPlane."""
        return self._submit(
            'forward_from_data_plane',
            model_types.DataPlaneForwardRequest,
            model_types.ForwardResponse,
            input_refs=_data_refs(inputs),
            input_field=input_field,
            kwarg_fields=kwarg_fields or {},
            **kwargs)

    def forward_only_from_data_plane(
        self,
        inputs: DataRef | list[DataRef],
        *,
        input_field: str | None = None,
        kwarg_fields: dict[str, str] | None = None,
        output_ref: DataRef | None = None,
        output_fields: dict[str, str] | None = None,
        **kwargs,
    ) -> model_types.ForwardResponse | DataRef:
        """Execute forward-only using DataPlane rows and optionally append outputs."""
        result = self._submit(
            'forward_only_from_data_plane',
            model_types.DataPlaneForwardOnlyRequest,
            model_types.ForwardResponse,
            input_refs=_data_refs(inputs),
            input_field=input_field,
            kwarg_fields=kwarg_fields or {},
            output_ref=output_ref.model_dump() if output_ref is not None else None,
            output_fields=output_fields or {},
            **kwargs)
        if output_ref is not None:
            return DataRef(**result.result)
        return result

    def forward_backward_from_data_plane(
        self,
        inputs: DataRef | list[DataRef],
        *,
        input_field: str | None = None,
        kwarg_fields: dict[str, str] | None = None,
        **kwargs,
    ) -> model_types.ForwardBackwardResponse:
        """Execute forward/backward using rows referenced from the server DataPlane."""
        return self._submit(
            'forward_backward_from_data_plane',
            model_types.DataPlaneForwardRequest,
            model_types.ForwardBackwardResponse,
            input_refs=_data_refs(inputs),
            input_field=input_field,
            kwarg_fields=kwarg_fields or {},
            **kwargs)

    # ------------------------------------------------------------------ #
    # Optimizer / scheduler steps
    # ------------------------------------------------------------------ #

    def step(self, **kwargs) -> None:
        """Execute optimizer step."""
        self._submit('step', model_types.StepRequest, None, **kwargs)

    def zero_grad(self, **kwargs) -> None:
        """Zero out gradients."""
        self._submit('zero_grad', model_types.AdapterRequest, None, **kwargs)

    def lr_step(self, **kwargs) -> None:
        """Execute learning rate scheduler step."""
        self._submit('lr_step', model_types.LrStepRequest, None, **kwargs)

    def clip_grad_norm(self,
                       max_grad_norm: float = 1.0,
                       norm_type: int = 2,
                       **kwargs) -> model_types.ClipGradNormResponse:
        """Clip gradient norm."""
        return self._submit(
            'clip_grad_norm',
            model_types.ClipGradNormRequest,
            model_types.ClipGradNormResponse,
            max_grad_norm=max_grad_norm,
            norm_type=norm_type,
            **kwargs)

    def clip_grad_and_step(self, max_grad_norm: float = 1.0, norm_type: int = 2, **kwargs) -> None:
        """Clip gradient norm and execute optimizer step in one call."""
        self._submit(
            'clip_grad_and_step',
            model_types.ClipGradAndStepRequest,
            None,
            max_grad_norm=max_grad_norm,
            norm_type=norm_type,
            **kwargs)

    # ------------------------------------------------------------------ #
    # Plugin setters
    # ------------------------------------------------------------------ #

    def set_loss(self, loss_cls: str, **kwargs) -> None:
        """Set the loss function."""
        self._submit('set_loss', model_types.SetLossRequest, None, loss_cls=loss_cls, **kwargs)

    def set_optimizer(self, optimizer_cls: str, **kwargs) -> None:
        """Set the optimizer."""
        self._submit('set_optimizer', model_types.SetOptimizerRequest, None, optimizer_cls=optimizer_cls, **kwargs)

    def set_lr_scheduler(self, scheduler_cls: str, **kwargs) -> None:
        """Set the learning rate scheduler."""
        self._submit('set_lr_scheduler', model_types.SetLrSchedulerRequest, None, scheduler_cls=scheduler_cls, **kwargs)

    def set_template(self, template_cls: str, **kwargs) -> None:
        """Set the template for data processing.

        ``model_id`` is not injected here: the backend always overrides it with its own
        tokenizer id, so sending it made the request advertise a parameter that had no
        effect. A caller that passes it explicitly still reaches the template
        constructor through the passthrough region.
        """
        self._submit('set_template', model_types.SetTemplateRequest, None, template_cls=template_cls, **kwargs)

    def set_processor(self, processor_cls: str, **kwargs) -> None:
        """Set the input processor."""
        self._submit('set_processor', model_types.SetProcessorRequest, None, processor_cls=processor_cls, **kwargs)

    def add_metric(self, metric_cls: str, is_training: bool | None = None, **kwargs) -> None:
        """Add a metric to the model."""
        self._submit(
            'add_metric', model_types.AddMetricRequest, None, metric_cls=metric_cls, is_training=is_training, **kwargs)

    def apply_patch(self, patch_cls: str, **kwargs) -> None:
        """Apply a patch to the model."""
        self._submit('apply_patch', model_types.ApplyPatchRequest, None, patch_cls=patch_cls, **kwargs)

    def calculate_metric(self, is_training: bool = True, **kwargs) -> model_types.CalculateMetricResponse:
        """Calculate metrics from model outputs."""
        return self._submit(
            'calculate_metric',
            model_types.CalculateMetricRequest,
            model_types.CalculateMetricResponse,
            is_training=is_training,
            **kwargs)

    # ------------------------------------------------------------------ #
    # Checkpoint I/O
    # ------------------------------------------------------------------ #

    def save(self, name: str, **kwargs) -> model_types.SaveResponse:
        """Save model checkpoint."""
        return self._submit('save', model_types.SaveRequest, model_types.SaveResponse, name=name, **kwargs)

    def load(self, name: str, **kwargs) -> None:
        """Load model checkpoint."""
        self._submit('load', model_types.LoadRequest, None, name=name, **kwargs)

    def resume_from_checkpoint(self, name: str, *, resume_only_model: bool = False, **kwargs) -> dict[str, Any]:
        """Resume weights (and optionally optimizer state) from a checkpoint."""
        progress = self._submit(
            'resume_from_checkpoint',
            model_types.ResumeFromCheckpointRequest,
            model_types.TrainingProgressResponse,
            name=name,
            resume_only_model=resume_only_model,
            **kwargs)
        return progress.result

    def upload_to_hub(
        self,
        checkpoint_dir: str,
        hub_model_id: str,
        hub_token: str | None = None,
    ) -> None:
        """Upload model checkpoint to hub.

        Submits the upload task and blocks (via the Client_Future_Layer) until it
        finishes, raising ``TaskFailedError`` on failure.

        Args:
            checkpoint_dir: The directory path of the checkpoint to upload.
            hub_model_id: The hub model id.
            hub_token: The hub token (optional).
        """
        logger.info('[upload_to_hub] submitting upload, waiting for completion...')
        self._submit(
            'upload_to_hub',
            model_types.UploadToHubRequest,
            None,
            checkpoint_dir=checkpoint_dir,
            hub_model_id=hub_model_id,
            hub_token=hub_token)
        logger.info('[upload_to_hub] upload completed successfully.')
