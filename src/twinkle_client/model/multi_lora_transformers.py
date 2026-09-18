import itertools
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from twinkle_client._request_builder import build_request
from twinkle_client.http import http_post, http_post_model
from twinkle_client.types import model as model_types
from twinkle_client.types.component import DataRef

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
    """Client wrapper for TwinkleModel that calls server HTTP endpoints.

    This client manages adapters and sends training/inference requests to the model server.
    The server-side session (managed by TwinkleClient) keeps the model alive.

    Every method builds its endpoint's request model rather than a dict, so a
    misspelled or wrongly-typed argument fails here -- in the caller's own stack trace,
    with no request sent. Arguments that are not declared fields (loss inputs, plugin
    constructor arguments) are routed into that model's passthrough region, so public
    signatures stay ``**kwargs`` and callers are unchanged.
    """

    def __init__(self, model_id: str, **kwargs):
        """Initialize model client."""
        from twinkle_client.http import get_base_url
        self.server_url = get_base_url()
        kwargs.pop('data_plane_url', None)

        if '://' in model_id:
            model_id = model_id.split('://')[1]
        self.model_id = model_id
        self.server_url = f'{self.server_url}/model/{model_id}/twinkle'
        self.adapter_name = None
        # Per-client monotonic sequence for idempotent dedup of stateful training ops:
        # the server dedups on (session_id, seq_id) so a retried grad/step call is
        # applied at most once. Reserved once per call and reused on retry.
        self._seq_counter = itertools.count(1)
        self._seq_lock = threading.Lock()
        response = http_post(url=f'{self.server_url}/create', )
        response.raise_for_status()

    # ------------------------------------------------------------------ #
    # Request plumbing
    # ------------------------------------------------------------------ #

    def _submit(self, endpoint: str, model_cls, response_cls, **values):
        """Build, send, and resolve one twinkle-native request."""
        body = build_request(model_cls, **values)
        response = http_post_model(f'{self.server_url}/{endpoint}', body)
        return self._await_task(response, response_cls)

    @staticmethod
    def _await_task(response, model_cls):
        """Resolve a Submit_Endpoint response through the Client_Future_Layer.

        Blocks until the task is terminal and returns the deserialized ``model_cls``
        result (or ``None``), raising ``TaskFailedError`` on a failed terminal state.
        Keeps every public method's synchronous signature unchanged.
        """
        from twinkle_client._future import resolve_response
        return resolve_response(response, model_cls)

    def _next_seq_id(self) -> int:
        """Reserve the next monotonic seq_id for a stateful op (dedup key with session)."""
        with self._seq_lock:
            return next(self._seq_counter)

    # ------------------------------------------------------------------ #
    # Adapter lifecycle
    # ------------------------------------------------------------------ #

    def add_adapter_to_model(self, adapter_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Add a new adapter to the model.

        Pass a peft ``LoraConfig`` (or its dict form) for LoRA training against a
        LoRA-mode deployment. Pass ``config=None`` for full-parameter training
        against a ``train_mode: full`` deployment.
        """
        save_dir = kwargs.pop('save_dir', None)
        if save_dir:
            save_dir = Path(save_dir).expanduser().resolve().as_posix()
        self._submit(
            'add_adapter_to_model',
            model_types.AddAdapterRequest,
            None,
            adapter_name=adapter_name,
            config=config,
            save_dir=save_dir,
            **kwargs)
        self.adapter_name = adapter_name

    def remove_adapter(self, adapter_name: str | None = None) -> None:
        """Release one client-owned adapter from the training component."""
        name = adapter_name or self.adapter_name
        self._submit('remove_adapter', model_types.AdapterRequest, None, adapter_name=name)
        if name == self.adapter_name:
            self.adapter_name = None

    # ------------------------------------------------------------------ #
    # Inline forward family
    # ------------------------------------------------------------------ #

    def forward(self, inputs: Any, **kwargs) -> model_types.ForwardResponse:
        """Execute forward pass on inline model inputs."""
        return self._submit(
            'forward',
            model_types.ForwardRequest,
            model_types.ForwardResponse,
            inputs=inputs,
            adapter_name=self.adapter_name,
            **kwargs)

    def forward_only(self, inputs: Any, **kwargs) -> model_types.ForwardResponse:
        """Execute forward pass without gradient computation on inline inputs."""
        return self._submit(
            'forward_only',
            model_types.ForwardOnlyRequest,
            model_types.ForwardResponse,
            inputs=inputs,
            adapter_name=self.adapter_name,
            **kwargs)

    def forward_backward(self, inputs: Any, **kwargs) -> model_types.ForwardBackwardResponse:
        """Execute combined forward and backward pass on inline inputs."""
        return self._submit(
            'forward_backward',
            model_types.ForwardBackwardTaskRequest,
            model_types.ForwardBackwardResponse,
            inputs=inputs,
            adapter_name=self.adapter_name,
            seq_id=self._next_seq_id(),
            **kwargs)

    def calculate_loss(self, **kwargs) -> model_types.CalculateLossResponse:
        """Calculate loss from model outputs."""
        return self._submit(
            'calculate_loss',
            model_types.AdapterRequest,
            model_types.CalculateLossResponse,
            adapter_name=self.adapter_name,
            **kwargs)

    def get_train_configs(self, **kwargs) -> model_types.GetTrainConfigsResponse:
        """Get training configs."""
        return self._submit(
            'get_train_configs',
            model_types.AdapterRequest,
            model_types.GetTrainConfigsResponse,
            adapter_name=self.adapter_name,
            **kwargs)

    def backward(self, **kwargs) -> None:
        """Execute backward pass."""
        self._submit(
            'backward',
            model_types.AdapterRequest,
            None,
            adapter_name=self.adapter_name,
            seq_id=self._next_seq_id(),
            **kwargs)

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
            adapter_name=self.adapter_name,
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
            adapter_name=self.adapter_name,
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
            adapter_name=self.adapter_name,
            input_field=input_field,
            kwarg_fields=kwarg_fields or {},
            seq_id=self._next_seq_id(),
            **kwargs)

    # ------------------------------------------------------------------ #
    # Optimizer / scheduler steps
    # ------------------------------------------------------------------ #

    def step(self, **kwargs) -> None:
        """Execute optimizer step."""
        self._submit(
            'step', model_types.StepRequest, None, adapter_name=self.adapter_name, seq_id=self._next_seq_id(), **kwargs)

    def zero_grad(self, **kwargs) -> None:
        """Zero out gradients."""
        self._submit('zero_grad', model_types.AdapterRequest, None, adapter_name=self.adapter_name, **kwargs)

    def lr_step(self, **kwargs) -> None:
        """Execute learning rate scheduler step."""
        self._submit(
            'lr_step',
            model_types.LrStepRequest,
            None,
            adapter_name=self.adapter_name,
            seq_id=self._next_seq_id(),
            **kwargs)

    def clip_grad_norm(self,
                       max_grad_norm: float = 1.0,
                       norm_type: int = 2,
                       **kwargs) -> model_types.ClipGradNormResponse:
        """Clip gradient norm."""
        return self._submit(
            'clip_grad_norm',
            model_types.ClipGradNormRequest,
            model_types.ClipGradNormResponse,
            adapter_name=self.adapter_name,
            max_grad_norm=max_grad_norm,
            norm_type=norm_type,
            **kwargs)

    def clip_grad_and_step(self, max_grad_norm: float = 1.0, norm_type: int = 2, **kwargs) -> None:
        """Clip gradient norm and execute optimizer step in one call."""
        self._submit(
            'clip_grad_and_step',
            model_types.ClipGradAndStepRequest,
            None,
            adapter_name=self.adapter_name,
            max_grad_norm=max_grad_norm,
            norm_type=norm_type,
            seq_id=self._next_seq_id(),
            **kwargs)

    # ------------------------------------------------------------------ #
    # Plugin setters
    # ------------------------------------------------------------------ #

    def set_loss(self, loss_cls: str, **kwargs) -> None:
        """Set the loss function."""
        self._submit(
            'set_loss', model_types.SetLossRequest, None, loss_cls=loss_cls, adapter_name=self.adapter_name, **kwargs)

    def set_optimizer(self, optimizer_cls: str, **kwargs) -> None:
        """Set the optimizer."""
        self._submit(
            'set_optimizer',
            model_types.SetOptimizerRequest,
            None,
            optimizer_cls=optimizer_cls,
            adapter_name=self.adapter_name,
            **kwargs)

    def set_lr_scheduler(self, scheduler_cls: str, **kwargs) -> None:
        """Set the learning rate scheduler."""
        self._submit(
            'set_lr_scheduler',
            model_types.SetLrSchedulerRequest,
            None,
            scheduler_cls=scheduler_cls,
            adapter_name=self.adapter_name,
            **kwargs)

    def set_template(self, template_cls: str, **kwargs) -> None:
        """Set the template for data processing.

        ``model_id`` is not injected here: the backend always overrides it with its own
        tokenizer id, so sending it made the request advertise a parameter that had no
        effect. A caller that passes it explicitly still reaches the template
        constructor through the passthrough region.
        """
        self._submit(
            'set_template',
            model_types.SetTemplateRequest,
            None,
            template_cls=template_cls,
            adapter_name=self.adapter_name,
            **kwargs)

    def set_processor(self, processor_cls: str, **kwargs) -> None:
        """Set the input processor."""
        self._submit(
            'set_processor',
            model_types.SetProcessorRequest,
            None,
            processor_cls=processor_cls,
            adapter_name=self.adapter_name,
            **kwargs)

    def add_metric(self, metric_cls: str, is_training: Optional[bool] = None, **kwargs) -> None:
        """Add a metric to the model."""
        self._submit(
            'add_metric',
            model_types.AddMetricRequest,
            None,
            metric_cls=metric_cls,
            is_training=is_training,
            adapter_name=self.adapter_name,
            **kwargs)

    def apply_patch(self, patch_cls: str, **kwargs) -> None:
        """Apply a patch to the model."""
        self._submit(
            'apply_patch',
            model_types.ApplyPatchRequest,
            None,
            patch_cls=patch_cls,
            adapter_name=self.adapter_name,
            **kwargs)

    def calculate_metric(self, is_training: bool = True, **kwargs) -> model_types.CalculateMetricResponse:
        """Calculate metrics from model outputs."""
        return self._submit(
            'calculate_metric',
            model_types.CalculateMetricRequest,
            model_types.CalculateMetricResponse,
            is_training=is_training,
            adapter_name=self.adapter_name,
            **kwargs)

    # ------------------------------------------------------------------ #
    # Checkpoint I/O
    # ------------------------------------------------------------------ #

    def save(self, name: str, **kwargs) -> model_types.SaveResponse:
        """Save model checkpoint."""
        return self._submit(
            'save',
            model_types.SaveRequest,
            model_types.SaveResponse,
            name=name,
            adapter_name=self.adapter_name,
            **kwargs)

    def load(self, name: str, **kwargs) -> None:
        """Load model checkpoint."""
        self._submit('load', model_types.LoadRequest, None, name=name, adapter_name=self.adapter_name, **kwargs)

    def resume_from_checkpoint(self, name: str, *, resume_only_model: bool = False, **kwargs) -> Dict[str, Any]:
        """Resume weights (and optionally optimizer state) from a checkpoint."""
        progress = self._submit(
            'resume_from_checkpoint',
            model_types.ResumeFromCheckpointRequest,
            model_types.TrainingProgressResponse,
            name=name,
            adapter_name=self.adapter_name,
            resume_only_model=resume_only_model,
            **kwargs)
        return progress.result

    def upload_to_hub(
        self,
        checkpoint_dir: str,
        hub_model_id: str,
        hub_token: Optional[str] = None,
        async_upload: bool = True,
        poll_interval: float = 5.0,
    ) -> None:
        """Upload model checkpoint to hub.

        Submits the upload task and blocks (via the Client_Future_Layer) until it
        finishes, raising ``TaskFailedError`` on failure.

        Args:
            checkpoint_dir: The directory path of the checkpoint to upload.
            hub_model_id: The hub model id.
            hub_token: The hub token (optional).
            async_upload: Deprecated, has no effect. The server always runs the
                upload in the background and the client waits via the future layer.
            poll_interval: Deprecated, has no effect. Pacing is now owned by the
                server-side long-poll of the Retrieve_Endpoint.
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
