import asyncio
from dataclasses import asdict
from peft import PeftConfig
from typing import Any, Dict, List, Optional, Union

from twinkle.data_format import InputFeature, SamplingParams, Trajectory
from twinkle.protocol.json_utils import json_safe
from twinkle.protocol.types.component import DataPlaneSampleRequest, DataRef, UnloadAdapterPathsRequest
from twinkle.protocol.types.sampler import (SamplerAddAdapterRequest, SamplerAddAdapterResponse, SampleRequest,
                                            SampleResponseModel, SampleResponseModelList, SamplerSetTemplateRequest,
                                            SamplerSetTemplateResponse)
from twinkle_client._request_builder import build_request
from twinkle_client.http import ClientTransport
from twinkle_client.http.context import capture_transport


# Intentionally does NOT subclass ``twinkle.sampler.base.Sampler``: importing
# that base pulls ``twinkle.sampler.__init__`` → ``VLLMEngine`` → torch + zmq,
# which the mock / CPU-only client environments don't have.
def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy arrays / torch tensors to JSON-serialisable lists.

    ``sample()`` accepts pre-encoded ``InputFeature`` dicts (e.g. from a multi-turn
    rollout's ``template.encode``) whose values are numpy arrays or torch tensors;
    these are not JSON-serialisable and would break the HTTP POST. Detection is by
    duck-typing (``.tolist()``) so this stays free of a hard torch/numpy import,
    honouring the CPU-only client contract noted above.
    """
    return json_safe(obj)


class vLLMSampler:
    """Client wrapper for Sampler that calls server HTTP endpoints.

    This client manages sampling operations and adapter synchronization with the sampler server.
    The server-side session (managed by TwinkleClient) keeps the sampler alive.
    """

    def __init__(
        self,
        model_id: str,
        *,
        transport: ClientTransport | None = None,
        **kwargs,
    ):
        """Create the sampler instance on server with one captured transport."""
        self._transport = capture_transport(transport)
        from twinkle_client.data_plane import DataPlaneClient
        self.data_plane = DataPlaneClient(kwargs.pop('data_plane_url', None), transport=self._transport)

        self.adapter_name = None
        if '://' in model_id:
            model_id = model_id.split('://')[1]
        self.model_id = model_id
        self.server_url = f'{self._transport.context.base_url}/sampler/{model_id}/twinkle'
        self._transport.post(f'{self.server_url}/create', json_data=kwargs)

    def _await_task(self, response, model_cls):
        """Resolve a Submit_Endpoint response through the Client_Future_Layer."""
        from twinkle_client._future import resolve_response
        return resolve_response(response, model_cls, transport=self._transport)

    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig, **kwargs) -> SamplerAddAdapterResponse:
        """Add a new adapter to the sampler."""
        if isinstance(config, PeftConfig):
            config = config.__dict__
        body = build_request(SamplerAddAdapterRequest, adapter_name=adapter_name, config=config, **kwargs)
        response = self._transport.post_model(f'{self.server_url}/add_adapter_to_sampler', body)
        self.adapter_name = adapter_name
        return SamplerAddAdapterResponse(**response.json())

    def sample(
        self,
        inputs: Union[List[Trajectory], List[InputFeature]],
        sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]] = None,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        num_samples: int = 1,
    ) -> List[SampleResponseModel]:
        """Sample from the model.

        Args:
            inputs: List of Trajectory or InputFeature to sample from.
            sampling_params: Sampling parameters mapping or Twinkle ``SamplingParams``.
            adapter_name: Adapter name for LoRA inference.
            adapter_uri: Adapter URI (twinkle:// path or local path) for LoRA inference.
            num_samples: Number of completions to generate per prompt.

        Returns:
            SampleResponseModel with 'sequences' list, each containing tokens, logprobs, stop_reason.
        """
        response = self._transport.post_model(
            f'{self.server_url}/sample',
            build_request(
                SampleRequest,
                inputs=_json_safe(inputs),
                sampling_params=self._sampling_params(sampling_params, num_samples),
                adapter_name=adapter_name,
                adapter_uri=adapter_uri))
        return self._await_task(response, SampleResponseModelList).samples

    @staticmethod
    def _sampling_params(sampling_params: Optional[Union[SamplingParams, Dict[str, Any]]],
                         num_samples: int) -> Dict[str, Any]:
        """Normalise sampling parameters into the single dict the server builds from."""
        if isinstance(sampling_params, SamplingParams):
            sampling_params = asdict(sampling_params)
        else:
            sampling_params = dict(sampling_params or {})
        if num_samples != 1 and sampling_params.setdefault('num_samples', num_samples) != num_samples:
            raise ValueError('num_samples conflicts with sampling_params.num_samples')
        return _json_safe(sampling_params)

    def sample_to_data_plane(
        self,
        inputs: Union[List[Trajectory], List[InputFeature], DataRef],
        sampling_params: Optional[Dict[str, Any]] = None,
        *,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        policy_version: int | None = None,
        group_ids: list[str] | None = None,
        num_samples: int = 1,
    ) -> DataRef:
        """Generate complete prompt groups and keep their rows in the server DataPlane."""
        source = ({'input_ref': inputs.model_dump()} if isinstance(inputs, DataRef) else {'inputs': _json_safe(inputs)})
        body = build_request(
            DataPlaneSampleRequest,
            sampling_params=_json_safe(sampling_params) if sampling_params else None,
            adapter_name=adapter_name,
            adapter_uri=adapter_uri,
            policy_version=policy_version,
            group_ids=group_ids,
            num_samples=num_samples,
            **source)
        response = self._transport.post_model(f'{self.server_url}/sample_to_data_plane', body)
        return self._await_task(response, DataRef)

    async def asample(
        self,
        inputs: Union[List[Trajectory], List[InputFeature]],
        sampling_params: Optional[Dict[str, Any]] = None,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        num_samples: int = 1,
    ) -> List[SampleResponseModel]:
        """Asynchronous convenience wrapper for the materialized sample API."""
        return await asyncio.to_thread(
            self.sample,
            inputs,
            sampling_params,
            adapter_name=adapter_name,
            adapter_uri=adapter_uri,
            num_samples=num_samples,
        )

    async def asample_to_data_plane(
        self,
        inputs: Union[List[Trajectory], List[InputFeature], DataRef],
        sampling_params: Optional[Dict[str, Any]] = None,
        *,
        adapter_name: str = '',
        adapter_uri: Optional[str] = None,
        policy_version: int | None = None,
        group_ids: list[str] | None = None,
        num_samples: int = 1,
    ) -> DataRef:
        """Asynchronously sample and return the opaque server-side result reference."""
        return await asyncio.to_thread(
            self.sample_to_data_plane,
            inputs,
            sampling_params,
            adapter_name=adapter_name,
            adapter_uri=adapter_uri,
            policy_version=policy_version,
            group_ids=group_ids,
            num_samples=num_samples,
        )

    def unload_adapter_paths(self, adapter_paths: list[str]) -> None:
        """Evict policy snapshots that are no longer referenced by this client."""
        self._transport.post_model(f'{self.server_url}/unload_adapter_paths',
                                   build_request(UnloadAdapterPathsRequest, adapter_paths=adapter_paths))

    def set_template(self, template_cls: str, adapter_name: str = '', **kwargs) -> SamplerSetTemplateResponse:
        """Set the template for encoding trajectories."""
        body = build_request(SamplerSetTemplateRequest, template_cls=template_cls, adapter_name=adapter_name, **kwargs)
        response = self._transport.post_model(f'{self.server_url}/set_template', body)
        return SamplerSetTemplateResponse(**response.json())

    def apply_patch(self, patch_cls: str, **kwargs) -> None:
        """Apply a patch to the model."""
        from twinkle.protocol.types.model import ApplyPatchRequest
        body = build_request(ApplyPatchRequest, patch_cls=patch_cls, adapter_name=self.adapter_name or '', **kwargs)
        self._transport.post_model(f'{self.server_url}/apply_patch', body)
