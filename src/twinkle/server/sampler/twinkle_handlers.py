# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Twinkle-native sampler handler mixin.

Provides /twinkle/* sampler endpoints.
"""
from __future__ import annotations

import asyncio
import json
import traceback
import uuid
from collections.abc import Callable
from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import SamplerManagement

import numpy as np

import twinkle_client.types as types
from twinkle.data_format import SamplingParams
from twinkle.server.exceptions import EndpointUnavailableError, RequestRejectedError
from twinkle.server.lifecycle.submit import backend_kwargs, resolve_twinkle_adapter_name, to_backend_inputs
from twinkle.server.sampler.weights import resolve_sampler_weights
from twinkle.server.telemetry.correlation import MODEL_ID
from twinkle.server.telemetry.tracing import traced_operation
from twinkle.server.utils.task_errors import task_error_payload
from twinkle.utils.logger import get_logger
from twinkle_client.common.json_utils import json_safe
from twinkle_client.types import sampler as sampler_types

logger = get_logger()


def _serialize_input_feature(feature: dict) -> dict:
    """Convert numpy arrays / torch tensors in an InputFeature to plain Python lists."""
    result = {}
    for k, v in feature.items():
        if isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            try:
                import torch
                if isinstance(v, torch.Tensor):
                    result[k] = v.tolist()
                    continue
            except ImportError:
                pass
            result[k] = v
    return result


def _get_twinkle_sampler_adapter_name(request: Request, adapter_name: str | None) -> str | None:
    """Per-session adapter name; delegates to the shared lifecycle resolver."""
    return resolve_twinkle_adapter_name(request, adapter_name)


def _build_rollout_rows_and_tags(
    sample_models: list[types.SampleResponseModel],
    *,
    group_ids: list[str] | None,
    policy_version: int | None,
    adapter_uri: str | None,
) -> tuple[list[dict], list[dict]]:
    """Flatten sampler output to one TQ row per generated sequence."""
    resolved_group_ids = group_ids or [uuid.uuid4().hex for _ in sample_models]
    if len(resolved_group_ids) != len(sample_models):
        raise ValueError(f'group_ids contains {len(resolved_group_ids)} values for '
                         f'{len(sample_models)} sampler inputs')
    rows = []
    tags = []
    for prompt_index, (response, group_id) in enumerate(zip(sample_models, resolved_group_ids)):
        for generation_idx, sequence in enumerate(response.sequences):
            sampled_logprobs = [
                0.0 if not position else float(position[0][1]) for position in (sequence.logprobs or [])
            ]
            rows.append({
                'train_input': sequence.new_input_feature,
                'sampled_logprobs': sampled_logprobs,
                'tokens': sequence.tokens,
                'decoded': sequence.decoded,
                'stop_reason': sequence.stop_reason,
                'prompt_logprobs': response.prompt_logprobs,
                'topk_prompt_logprobs': response.topk_prompt_logprobs,
            })
            tags.append({
                'record_type': 'sample',
                'group_id': group_id,
                'prompt_index': prompt_index,
                'generation_idx': generation_idx,
                'rollout_status': 'ROLLOUT_DONE',
                'rollout_policy_version': policy_version,
                'rollout_adapter_uri': adapter_uri,
            })
    return rows, tags


def _to_sample_response_models(responses) -> list[types.SampleResponseModel]:
    """Convert internal sampler responses to the HTTP response schema."""
    sample_models = []
    for response in responses:
        sequences = [
            types.SampledSequenceModel(
                stop_reason=sequence.stop_reason,
                tokens=list(sequence.tokens),
                logprobs=list(sequence.logprobs) if sequence.logprobs is not None else None,
                decoded=sequence.decoded,
                new_input_feature=(_serialize_input_feature(sequence.new_input_feature)
                                   if sequence.new_input_feature is not None else None),
            ) for sequence in response.sequences
        ]
        sample_models.append(
            types.SampleResponseModel(
                sequences=sequences,
                # The prompt's own ids, which is what makes a reply served over
                # HTTP trainable: paired with a sequence's tokens they are the
                # exact ids the model ran on, so a caller never has to re-encode
                # the text and hope the tokenizer agrees with itself.
                prompt_token_ids=(list(response.prompt_token_ids) if response.prompt_token_ids is not None else None),
                prompt_logprobs=response.prompt_logprobs,
                topk_prompt_logprobs=response.topk_prompt_logprobs,
            ))
    return sample_models


def _submission_states(value) -> list[dict]:
    """Normalize Twinkle's single-worker unwrapping to a list of states."""
    return value if isinstance(value, list) else [value]


async def _stream_queue(q, sentinel, request_id: str, total_timeout: float, single_get_timeout: float = 60.0):
    loop = asyncio.get_running_loop()
    start = loop.time()
    try:
        while True:
            remaining = total_timeout - (loop.time() - start)
            if remaining <= 0:
                payload = task_error_payload(
                    'sample_stream exceeded the execution time bound', request_id=request_id, error_code=504)
                yield json.dumps(payload) + '\n'
                break
            try:
                item = await asyncio.wait_for(
                    loop.run_in_executor(None, q.get), timeout=min(single_get_timeout, remaining))
            except asyncio.TimeoutError:
                payload = task_error_payload(
                    'sample_stream timed out waiting for the next token', request_id=request_id, error_code=504)
                yield json.dumps(payload) + '\n'
                break
            if item == sentinel:
                break
            if isinstance(item, Exception):
                payload = task_error_payload(f'{type(item).__name__}: {item}', request_id=request_id, error_code=500)
                yield json.dumps(payload) + '\n'
                break
            delta, reason = item
            yield json.dumps({'delta': delta, 'finish_reason': reason}) + '\n'
    finally:
        try:
            q.shutdown(force=True)
        except Exception:
            pass


async def _await_generation(service: SamplerManagement, submission_id: str, timeout: float):
    """Poll one admitted generation through the backend boundary."""
    collected = False

    async def poll():
        nonlocal collected
        poll_interval = 0.01
        while True:
            try:
                states = _submission_states(await service.call_backend(service.sampler.get_generation_status,
                                                                       submission_id))
            except Exception as error:
                from ray.exceptions import TaskCancelledError
                if not isinstance(error, TaskCancelledError):
                    raise
                await asyncio.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, 0.25)
                continue
            failed = next(
                (state for state in states if state.get('status') not in ('running', 'completed')),
                None,
            )
            if failed is not None:
                error = failed.get('error') or failed.get('status', 'unknown failure')
                raise RuntimeError(f'generation {submission_id} failed: {error}')
            if states and all(state.get('status') == 'completed' for state in states):
                responses = await service.call_backend(service.sampler.collect_generation, submission_id)
                collected = True
                return responses
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 0.25)

    try:
        return await asyncio.wait_for(poll(), timeout=timeout)
    finally:
        if not collected:
            try:
                await asyncio.wait_for(
                    service.call_backend(service.sampler.cancel_generation, submission_id), timeout=4.0)
            except Exception:
                logger.warning('Failed to cancel generation %s', submission_id, exc_info=True)


def _register_twinkle_sampler_routes(app: FastAPI, self_fn: Callable[[], SamplerManagement]) -> None:
    """Register all /twinkle/* sampler routes on the given FastAPI app.

    self_fn is a zero-argument callable returning the current SamplerManagement replica instance.
    It is wired in via Depends so it is resolved lazily at request time.
    """

    @app.post('/twinkle/create', response_model=sampler_types.SamplerCreateResponse)
    async def create(
        request: Request, self: SamplerManagement = Depends(self_fn)) -> sampler_types.SamplerCreateResponse:
        """Health check / session creation endpoint."""
        return sampler_types.SamplerCreateResponse()

    @app.post('/twinkle/sample', response_model=types.TaskEnvelope)
    async def sample(request: Request, body: types.SampleRequest,
                     self: SamplerManagement = Depends(self_fn)) -> types.TaskEnvelope:
        """Sample completions from the model.

        Supports Trajectory or InputFeature inputs, with optional LoRA adapter.
        """
        token = await self._on_request_start(request)

        async def _task():
            # Resolve adapter
            adapter_path = None
            adapter_name = body.adapter_name or ''
            full_adapter_name = _get_twinkle_sampler_adapter_name(request, adapter_name) or ''

            if body.adapter_uri:
                from twinkle.server.checkpoint import create_checkpoint_manager
                checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
                _, resolved_uri = checkpoint_manager.parse_adapter_uri(body.adapter_uri)
                # Reset prefix cache only when new weights are loaded.
                await self.call_backend(self.sampler.reset_prefix_cache)
                adapter_path = await resolve_sampler_weights(self, resolved_uri)

            # Parse inputs (shared seam; batch form)
            inputs = to_backend_inputs(body.inputs)

            # Build sampling params
            params = None
            if body.sampling_params:
                params = SamplingParams.from_dict(body.sampling_params)

            # Sample
            responses = await self.call_backend(
                self.sampler.sample,
                inputs,
                params,
                adapter_name=full_adapter_name,
                adapter_path=adapter_path,
            )
            return types.SampleResponseModelList(samples=_to_sample_response_models(responses)).model_dump()

        # Calculate metrics for queue scheduling. The body is wire-validated, so the
        # entries are models and ``input_ids`` is absent or a list of ints.
        input_tokens = sum(len(getattr(entry, 'input_ids', None) or ()) for entry in body.inputs)
        return await self.submit_and_peek(_task, token=token, input_tokens=input_tokens, task_type='sample')

    @app.post('/twinkle/sample_to_data_plane', response_model=types.TaskEnvelope)
    async def sample_to_data_plane(
            request: Request,
            body: types.DataPlaneSampleRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> types.TaskEnvelope:
        """Generate a complete group, store it server-side, and return a Task_Envelope
        whose result is the stored group's DataRef."""
        token = await self._on_request_start(request)
        if not self.data_plane.enabled:
            raise EndpointUnavailableError('sample_to_data_plane requires data_plane_url')
        if not callable(getattr(self.sampler, 'submit_generation', None)):
            raise EndpointUnavailableError('sampler_type must be vllm_async')

        adapter_path = None
        full_adapter_name = _get_twinkle_sampler_adapter_name(request, body.adapter_name) or ''
        if body.adapter_uri:
            from twinkle.server.checkpoint import create_checkpoint_manager
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            _, adapter_path = checkpoint_manager.parse_adapter_uri(body.adapter_uri)

        inputs = (await self.data_plane.get(body.input_ref) if body.input_ref is not None else body.inputs)
        inputs = to_backend_inputs(inputs)

        params_dict = dict(body.sampling_params or {})
        params_dict['num_samples'] = body.num_samples
        params = SamplingParams.from_dict(params_dict)
        submission_id = uuid.uuid4().hex

        async def _generate_and_store():
            # vLLM async engine owns generation concurrency, so the whole
            # admit -> await -> store sequence runs as one background future
            # (outside the serial compute queue) and its result is the DataRef.
            await self.call_backend(
                self.sampler.submit_generation,
                submission_id,
                inputs,
                params,
                adapter_name=full_adapter_name,
                adapter_path=adapter_path,
            )
            responses = await _await_generation(self, submission_id,
                                                self._task_queue_config.effective_execution_timeout)
            rows, tags = _build_rollout_rows_and_tags(
                _to_sample_response_models(responses),
                group_ids=body.group_ids,
                policy_version=body.policy_version,
                adapter_uri=body.adapter_uri,
            )
            ref = await self.data_plane.put([json_safe(item) for item in rows], kind='rollout', tags=tags)
            return ref.model_dump()

        return await self.submit_background_and_peek(
            _generate_and_store, model_id=full_adapter_name or None, task_type='sample_to_data_plane')

    @app.post('/twinkle/unload_adapter_paths')
    async def unload_adapter_paths(
            request: Request,
            body: types.UnloadAdapterPathsRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> dict[str, str]:
        """Best-effort eviction of published LoRA snapshots from sampler caches."""
        token = await self._on_request_start(request)
        resolved_paths = []
        for adapter_path in body.adapter_paths:
            if adapter_path.startswith('twinkle://'):
                from twinkle.server.checkpoint import create_checkpoint_manager
                checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
                _, adapter_path = checkpoint_manager.parse_adapter_uri(adapter_path)
            resolved_paths.append(adapter_path)
        unload = getattr(self.sampler, 'unload_adapter_paths', None)
        if unload is not None:
            await self.call_backend(unload, resolved_paths)
        return {'status': 'ok'}

    @app.post('/twinkle/set_template', response_model=sampler_types.SamplerSetTemplateResponse)
    async def set_template(
            request: Request,
            body: sampler_types.SamplerSetTemplateRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> sampler_types.SamplerSetTemplateResponse:
        """Set the chat template for encoding Trajectory inputs."""
        with traced_operation('sampler.set_template'):
            await self.call_backend(self.sampler.set_template, body.template_cls, **backend_kwargs(body))
        return sampler_types.SamplerSetTemplateResponse()

    @app.post('/twinkle/add_adapter_to_sampler', response_model=sampler_types.SamplerAddAdapterResponse)
    async def add_adapter_to_sampler(
            request: Request,
            body: sampler_types.SamplerAddAdapterRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> sampler_types.SamplerAddAdapterResponse:
        """Add a LoRA adapter to the sampler."""
        # Raised, not asserted: decidable from the request body alone, so it owes the caller
        # a real 400 rather than an AssertionError surfacing as a 500 -- and a bare assert
        # would vanish under `python -O`, letting an empty adapter_name reach the backend.
        if not body.adapter_name:
            raise RequestRejectedError('`adapter_name` is required and must be non-empty.')
        full_adapter_name = _get_twinkle_sampler_adapter_name(request, body.adapter_name)

        from peft import LoraConfig
        config = LoraConfig(**body.config) if isinstance(body.config, dict) else body.config

        with traced_operation('sampler.add_adapter_to_sampler', attrs={MODEL_ID: self.model_id}):
            await self.call_backend(self.sampler.add_adapter_to_sampler, full_adapter_name, config)

        return sampler_types.SamplerAddAdapterResponse(adapter_name=full_adapter_name)

    @app.post('/twinkle/apply_patch')
    async def apply_patch(
            request: Request,
            body: types.ApplyPatchRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> None:
        from twinkle_client.common.serialize import deserialize_object
        patch_cls = deserialize_object(body.patch_cls)
        with traced_operation('sampler.apply_patch'):
            await self.call_backend(self.sampler.apply_patch, patch_cls, **backend_kwargs(body))

    @app.post('/twinkle/sample_stream')
    async def sample_stream(
            request: Request,
            body: types.SampleRequest,
            self: SamplerManagement = Depends(self_fn),
    ):
        """Stream token deltas as newline-delimited JSON.

        Each line is a JSON object: {"delta": "text", "finish_reason": null|"stop"|"length"}.

        Uses ``ray.util.queue.Queue`` to bridge the sampler's Actor process
        boundary — the sampler pushes deltas into the queue as they are
        generated, and this handler yields them to the HTTP response.
        """
        token = await self._on_request_start(request)

        adapter_path = None
        adapter_name = body.adapter_name or ''
        full_adapter_name = _get_twinkle_sampler_adapter_name(request, adapter_name) or ''

        if body.adapter_uri:
            from twinkle.server.checkpoint import create_checkpoint_manager
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            _, resolved_uri = checkpoint_manager.parse_adapter_uri(body.adapter_uri)
            await self.call_backend(self.sampler.reset_prefix_cache)
            adapter_path = await resolve_sampler_weights(self, resolved_uri)

        # Streaming accepts exactly one input; the shared seam enforces that and
        # returns a single parsed object. Its ValueError maps to the same 400 this
        # endpoint has always returned.
        try:
            inputs_parsed = to_backend_inputs(body.inputs, single=True)
        except ValueError as e:
            raise RequestRejectedError(str(e))

        params = None
        if body.sampling_params:
            params = SamplingParams.from_dict(body.sampling_params)

        from ray.util.queue import Queue

        from .backends import STREAM_SENTINEL

        request_id = f'req_{uuid.uuid4().hex}'
        actors = self.sampler._actors
        if not actors:

            async def _no_actor_generator():
                payload = task_error_payload('No available sampler actor', request_id=request_id, error_code=503)
                yield json.dumps(payload) + '\n'

            return StreamingResponse(_no_actor_generator(), media_type='application/x-ndjson')
        q = Queue(maxsize=128)
        actor = actors[0]
        actor.sample_stream_to_queue.remote(
            q,
            inputs_parsed,
            params,
            adapter_name=full_adapter_name,
            adapter_path=adapter_path,
        )

        return StreamingResponse(
            _stream_queue(
                q,
                STREAM_SENTINEL,
                request_id,
                self._task_queue_config.effective_execution_timeout,
            ),
            media_type='application/x-ndjson',
        )
