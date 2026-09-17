from typing import Any, Dict, Optional
import itertools
import logging
import threading
from pathlib import Path
from twinkle_client.http import http_post
from twinkle_client.common.json_utils import json_safe
from twinkle_client.types.component import DataRef
from twinkle_client.types.model import (
    CalculateLossResponse,
    CalculateMetricResponse,
    ClipGradNormResponse,
    ForwardBackwardResponse,
    ForwardResponse,
    GetTrainConfigsResponse,
    SaveResponse,
    TrainingProgressResponse,
)

logger = logging.getLogger('twinkle_client')


def _data_ref_payload(inputs: DataRef | list[DataRef]) -> dict[str, Any]:
    """Encode one or more opaque references for a DataPlane model endpoint."""
    refs = [inputs] if isinstance(inputs, DataRef) else list(inputs)
    if not refs:
        raise ValueError('at least one DataRef is required')
    if not all(isinstance(item, DataRef) for item in refs):
        raise TypeError('data-plane model inputs must contain only DataRef values')
    return {'input_refs': [item.model_dump() for item in refs]}


class MultiLoraTransformersModel:
    """Client wrapper for TwinkleModel that calls server HTTP endpoints.

    This client manages adapters and sends training/inference requests to the model server.
    The server-side session (managed by TwinkleClient) keeps the model alive.
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
        response = http_post(
            url=f'{self.server_url}/create',
        )
        response.raise_for_status()

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

    def add_adapter_to_model(self, adapter_name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Add a new adapter to the model.

        Pass a peft ``LoraConfig`` (or its dict form) for LoRA training against a
        LoRA-mode deployment. Pass ``config=None`` for full-parameter training
        against a ``train_mode: full`` deployment.
        """
        save_dir = kwargs.get('save_dir')
        if save_dir:
            kwargs['save_dir'] = Path(save_dir).expanduser().resolve().as_posix()
        response = http_post(
            url=f'{self.server_url}/add_adapter_to_model',
            json_data={'adapter_name': adapter_name, 'config': config, **kwargs}
        )
        self._await_task(response, None)
        self.adapter_name = adapter_name

    def remove_adapter(self, adapter_name: str | None = None) -> None:
        """Release one client-owned adapter from the training component."""
        name = adapter_name or self.adapter_name
        response = http_post(
            url=f'{self.server_url}/remove_adapter',
            json_data={'adapter_name': name},
        )
        self._await_task(response, None)
        if name == self.adapter_name:
            self.adapter_name = None

    def forward(self, inputs: Any, **kwargs) -> ForwardResponse:
        """Execute forward pass on inline model inputs."""
        response = http_post(
            url=f'{self.server_url}/forward',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs},
        )
        return self._await_task(response, ForwardResponse)

    def forward_only(self, inputs: Any, **kwargs) -> ForwardResponse:
        """Execute forward pass without gradient computation on inline inputs."""
        response = http_post(
            url=f'{self.server_url}/forward_only',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs},
        )
        return self._await_task(response, ForwardResponse)

    def forward_from_data_plane(
        self,
        inputs: DataRef | list[DataRef],
        *,
        input_field: str | None = None,
        kwarg_fields: dict[str, str] | None = None,
        **kwargs,
    ) -> ForwardResponse:
        """Execute forward using rows referenced from the server DataPlane."""
        response = http_post(
            url=f'{self.server_url}/forward_from_data_plane',
            json_data={
                **_data_ref_payload(inputs),
                'adapter_name': self.adapter_name,
                'input_field': input_field,
                'kwarg_fields': kwarg_fields or {},
                **json_safe(kwargs),
            },
        )
        return self._await_task(response, ForwardResponse)

    def forward_only_from_data_plane(
        self,
        inputs: DataRef | list[DataRef],
        *,
        input_field: str | None = None,
        kwarg_fields: dict[str, str] | None = None,
        output_ref: DataRef | None = None,
        output_fields: dict[str, str] | None = None,
        **kwargs,
    ) -> ForwardResponse | DataRef:
        """Execute forward-only using DataPlane rows and optionally append outputs."""
        body = {
            **_data_ref_payload(inputs),
            'adapter_name': self.adapter_name,
            'input_field': input_field,
            'kwarg_fields': kwarg_fields or {},
            'output_ref': output_ref.model_dump() if output_ref is not None else None,
            'output_fields': output_fields or {},
            **json_safe(kwargs),
        }
        response = http_post(
            url=f'{self.server_url}/forward_only_from_data_plane',
            json_data=body,
        )
        result = self._await_task(response, ForwardResponse)
        if output_ref is not None:
            return DataRef(**result.result)
        return result

    def calculate_loss(self, **kwargs) -> CalculateLossResponse:
        """Calculate loss from model outputs."""
        response = http_post(
            url=f'{self.server_url}/calculate_loss',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        return self._await_task(response, CalculateLossResponse)

    def get_train_configs(self, **kwargs) -> GetTrainConfigsResponse:
        """Get training configs."""
        response = http_post(
            url=f'{self.server_url}/get_train_configs',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        return self._await_task(response, GetTrainConfigsResponse)

    def backward(self, **kwargs) -> None:
        """Execute backward pass."""
        response = http_post(
            url=f'{self.server_url}/backward',
            json_data={'adapter_name': self.adapter_name, 'seq_id': self._next_seq_id(), **kwargs}
        )
        self._await_task(response, None)

    def forward_backward(self, inputs: Any, **kwargs) -> ForwardBackwardResponse:
        """Execute combined forward and backward pass on inline inputs."""
        response = http_post(
            url=f'{self.server_url}/forward_backward',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, 'seq_id': self._next_seq_id(), **kwargs},
        )
        return self._await_task(response, ForwardBackwardResponse)

    def forward_backward_from_data_plane(
        self,
        inputs: DataRef | list[DataRef],
        *,
        input_field: str | None = None,
        kwarg_fields: dict[str, str] | None = None,
        **kwargs,
    ) -> ForwardBackwardResponse:
        """Execute forward/backward using rows referenced from the server DataPlane."""
        response = http_post(
            url=f'{self.server_url}/forward_backward_from_data_plane',
            json_data={
                **_data_ref_payload(inputs),
                'adapter_name': self.adapter_name,
                'input_field': input_field,
                'kwarg_fields': kwarg_fields or {},
                'seq_id': self._next_seq_id(),
                **json_safe(kwargs),
            },
        )
        return self._await_task(response, ForwardBackwardResponse)

    def step(self, **kwargs) -> None:
        """Execute optimizer step."""
        response = http_post(
            url=f'{self.server_url}/step',
            json_data={'adapter_name': self.adapter_name, 'seq_id': self._next_seq_id(), **kwargs}
        )
        self._await_task(response, None)

    def zero_grad(self, **kwargs) -> None:
        """Zero out gradients."""
        response = http_post(
            url=f'{self.server_url}/zero_grad',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        self._await_task(response, None)

    def lr_step(self, **kwargs) -> None:
        """Execute learning rate scheduler step."""
        response = http_post(
            url=f'{self.server_url}/lr_step',
            json_data={'adapter_name': self.adapter_name, 'seq_id': self._next_seq_id(), **kwargs}
        )
        self._await_task(response, None)

    def clip_grad_norm(self, max_grad_norm: float = 1.0, norm_type: int = 2, **kwargs) -> ClipGradNormResponse:
        """Clip gradient norm."""
        response = http_post(
            url=f'{self.server_url}/clip_grad_norm',
            json_data={
                'max_grad_norm': max_grad_norm,
                'norm_type': norm_type,
                'adapter_name': self.adapter_name,
                **kwargs
            }
        )
        return self._await_task(response, ClipGradNormResponse)

    def clip_grad_and_step(self, max_grad_norm: float = 1.0, norm_type: int = 2, **kwargs) -> None:
        """Clip gradient norm and execute optimizer step in one call."""
        response = http_post(
            url=f'{self.server_url}/clip_grad_and_step',
            json_data={
                'max_grad_norm': max_grad_norm,
                'norm_type': norm_type,
                'adapter_name': self.adapter_name,
                'seq_id': self._next_seq_id(),
                **kwargs
            }
        )
        self._await_task(response, None)

    def set_loss(self, loss_cls: str, **kwargs) -> None:
        """Set the loss function."""
        response = http_post(
            url=f'{self.server_url}/set_loss',
            json_data={'loss_cls': loss_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        self._await_task(response, None)

    def set_optimizer(self, optimizer_cls: str, **kwargs) -> None:
        """Set the optimizer."""
        response = http_post(
            url=f'{self.server_url}/set_optimizer',
            json_data={'optimizer_cls': optimizer_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        self._await_task(response, None)

    def set_lr_scheduler(self, scheduler_cls: str, **kwargs) -> None:
        """Set the learning rate scheduler."""
        response = http_post(
            url=f'{self.server_url}/set_lr_scheduler',
            json_data={'scheduler_cls': scheduler_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        self._await_task(response, None)

    def save(self, name: str, **kwargs) -> SaveResponse:
        """Save model checkpoint."""
        response = http_post(
            url=f'{self.server_url}/save',
            json_data={'name': name, 'adapter_name': self.adapter_name, **kwargs}
        )
        return self._await_task(response, SaveResponse)

    def load(self, name: str, **kwargs) -> None:
        """Load model checkpoint."""
        response = http_post(
            url=f'{self.server_url}/load',
            json_data={'name': name, 'adapter_name': self.adapter_name, **kwargs}
        )
        self._await_task(response, None)

    def resume_from_checkpoint(self, name: str, *, resume_only_model: bool = False, **kwargs) -> Dict[str, Any]:
        response = http_post(
            url=f'{self.server_url}/resume_from_checkpoint',
            json_data={'name': name, 'adapter_name': self.adapter_name,
                       'resume_only_model': resume_only_model, **kwargs}
        )
        return self._await_task(response, TrainingProgressResponse).result

    def apply_patch(self, patch_cls: str, **kwargs) -> None:
        """Apply a patch to the model."""
        response = http_post(
            url=f'{self.server_url}/apply_patch',
            json_data={'patch_cls': patch_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        self._await_task(response, None)

    def add_metric(self, metric_cls: str, is_training: Optional[bool] = None, **kwargs) -> None:
        """Add a metric to the model."""
        response = http_post(
            url=f'{self.server_url}/add_metric',
            json_data={
                'metric_cls': metric_cls,
                'is_training': is_training,
                'adapter_name': self.adapter_name,
                **kwargs
            }
        )
        self._await_task(response, None)

    def set_template(self, template_cls: str, **kwargs) -> None:
        """Set the template for data processing."""
        response = http_post(
            url=f'{self.server_url}/set_template',
            json_data={
                'template_cls': template_cls,
                'adapter_name': self.adapter_name,
                'model_id': self.model_id,
                **kwargs
            }
        )
        self._await_task(response, None)

    def set_processor(self, processor_cls: str, **kwargs) -> None:
        """Set the input processor."""
        response = http_post(
            url=f'{self.server_url}/set_processor',
            json_data={'processor_cls': processor_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        self._await_task(response, None)

    def calculate_metric(self, is_training: bool = True, **kwargs) -> CalculateMetricResponse:
        """Calculate metrics from model outputs."""
        response = http_post(
            url=f'{self.server_url}/calculate_metric',
            json_data={'is_training': is_training, 'adapter_name': self.adapter_name, **kwargs}
        )
        return self._await_task(response, CalculateMetricResponse)

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
        response = http_post(
            url=f'{self.server_url}/upload_to_hub',
            json_data={
                'checkpoint_dir': checkpoint_dir,
                'hub_model_id': hub_model_id,
                'hub_token': hub_token,
            }
        )
        logger.info('[upload_to_hub] upload submitted, waiting for completion...')
        self._await_task(response, None)
        logger.info('[upload_to_hub] upload completed successfully.')
