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
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import TYPE_CHECKING

from twinkle_client.common.serialize import deserialize_object

if TYPE_CHECKING:
    from .app import SamplerManagement

import numpy as np

import twinkle_client.types as types
from twinkle.data_format import InputFeature, SamplingParams, Trajectory
from twinkle.server.telemetry.correlation import MODEL_ID, TOKEN_ID
from twinkle.server.telemetry.tracing import traced_operation
from twinkle.server.utils.validation import get_session_id_from_request
from twinkle.utils.logger import get_logger
from twinkle_client.common.json_utils import json_safe

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
    """Build a stable per-session adapter name, falling back to request_id for older clients."""
    if adapter_name is None or adapter_name == '':
        return None
    owner_id = get_session_id_from_request(request) or request.state.request_id
    return owner_id + '-' + adapter_name


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
                prompt_logprobs=response.prompt_logprobs,
                topk_prompt_logprobs=response.topk_prompt_logprobs,
            ))
    return sample_models


def _submission_states(value) -> list[dict]:
    """Normalize Twinkle's single-worker unwrapping to a list of states."""
    return value if isinstance(value, list) else [value]


async def _await_generation(
    sampler,
    submission_id: str,
):
    """Poll an admitted generation without occupying the sampler admission queue."""
    collected = False
    try:
        poll_interval = 0.01
        while True:
            try:
                states = _submission_states(await asyncio.to_thread(sampler.get_generation_status, submission_id))
            except Exception as error:
                # A pending read-only actor call can be cancelled by Ray while
                # the generation submitted just above remains alive.  Treating
                # that as a generation failure makes the finally block discard
                # otherwise valid rollout work.  Retry only Ray's explicit task
                # cancellation; actor death and application errors must still
                # propagate immediately.
                from ray.exceptions import TaskCancelledError
                if not isinstance(error, TaskCancelledError):
                    raise
                logger.warning(
                    'Generation status poll was cancelled; retrying submission %s',
                    submission_id,
                )
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
                responses = await asyncio.to_thread(sampler.collect_generation, submission_id)
                collected = True
                return responses
            await asyncio.sleep(poll_interval)
            poll_interval = min(poll_interval * 1.5, 0.25)
    finally:
        if not collected:
            try:
                await asyncio.to_thread(sampler.cancel_generation, submission_id)
            except Exception:
                logger.warning('Failed to cancel generation %s', submission_id, exc_info=True)


def _register_twinkle_sampler_routes(app: FastAPI, self_fn: Callable[[], SamplerManagement]) -> None:
    """Register all /twinkle/* sampler routes on the given FastAPI app.

    self_fn is a zero-argument callable returning the current SamplerManagement replica instance.
    It is wired in via Depends so it is resolved lazily at request time.
    """

    async def run_task(coro):
        """Await a schedule_task_and_wait coroutine and surface any exception as a
        structured HTTP 500 response so the client receives the full traceback instead
        of an opaque connection-level error.

        Note: HTTPException is re-raised directly to preserve its status code and detail.
        """
        try:
            return await coro
        except HTTPException:
            raise
        except Exception:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post('/twinkle/create', response_model=types.CreateResponse)
    async def create(request: Request, self: SamplerManagement = Depends(self_fn)) -> types.CreateResponse:
        """Health check / session creation endpoint."""
        return types.CreateResponse()

    @app.post('/twinkle/sample', response_model=types.SampleResponseModelList)
    async def sample(
        request: Request, body: types.SampleRequest,
        self: SamplerManagement = Depends(self_fn)) -> types.SampleResponseModelList:
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
                import os

                from twinkle.server.checkpoint import create_checkpoint_manager
                checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
                _, resolved_uri = checkpoint_manager.parse_adapter_uri(body.adapter_uri)
                # Reset prefix cache only when new weights are loaded
                self.sampler.reset_prefix_cache()
                # LoRA adapter dir (has adapter_config.json) vs full-parameter
                # HF checkpoint. Full checkpoints replace the sampler base model.
                if resolved_uri and os.path.exists(os.path.join(resolved_uri, 'adapter_config.json')):
                    adapter_path = resolved_uri
                elif resolved_uri:
                    self.sampler.load_full_weights_from_path(resolved_uri)

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

            # Sample
            responses = self.sampler.sample(
                inputs,
                params,
                adapter_name=full_adapter_name,
                adapter_path=adapter_path,
            )
            return types.SampleResponseModelList(samples=_to_sample_response_models(responses))

        # Calculate metrics for queue scheduling
        inputs_list = body.inputs if isinstance(body.inputs, list) else [body.inputs]
        input_tokens = sum(len(inp.get('input_ids', [])) if isinstance(inp, dict) else 0 for inp in inputs_list)
        return await run_task(
            self.schedule_task_and_wait(
                _task,
                token=token,
                input_tokens=input_tokens,
                task_type='sample',
            ))

    @app.post('/twinkle/sample_to_data_plane', response_model=types.DataRef)
    async def sample_to_data_plane(
            request: Request,
            body: types.DataPlaneSampleRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> types.DataRef:
        """Generate a complete group, store it server-side, and return its DataRef."""
        token = await self._on_request_start(request)
        if not self.data_plane.enabled:
            raise HTTPException(status_code=503, detail='sample_to_data_plane requires data_plane_url')
        if not callable(getattr(self.sampler, 'submit_generation', None)):
            raise HTTPException(status_code=503, detail='sampler_type must be vllm_async')

        adapter_path = None
        full_adapter_name = _get_twinkle_sampler_adapter_name(request, body.adapter_name) or ''
        if body.adapter_uri:
            from twinkle.server.checkpoint import create_checkpoint_manager
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            _, adapter_path = checkpoint_manager.parse_adapter_uri(body.adapter_uri)

        inputs = (await self.data_plane.get(body.input_ref) if body.input_ref is not None else body.inputs)
        if isinstance(inputs, list) and inputs:
            first = inputs[0]
            if isinstance(first, dict) and 'input_ids' in first:
                inputs = [InputFeature(**item) for item in inputs]
            else:
                inputs = [Trajectory(**item) for item in inputs]
        elif isinstance(inputs, dict):
            inputs = [InputFeature(**inputs)] if 'input_ids' in inputs else [Trajectory(**inputs)]

        params_dict = dict(body.sampling_params or {})
        params_dict['num_samples'] = body.num_samples
        params = SamplingParams.from_dict(params_dict)
        submission_id = uuid.uuid4().hex

        async def _admit():
            await asyncio.to_thread(
                self.sampler.submit_generation,
                submission_id,
                inputs,
                params,
                adapter_name=full_adapter_name,
                adapter_path=adapter_path,
            )
            return submission_id

        inline_inputs = body.inputs if isinstance(body.inputs, list) else [body.inputs]
        input_tokens = (
            body.input_ref.num_tokens if body.input_ref is not None else sum(
                len(item.get('input_ids', [])) for item in inline_inputs if isinstance(item, dict)))
        await run_task(
            self.schedule_task_and_wait(
                _admit,
                model_id=full_adapter_name or None,
                token=token,
                input_tokens=input_tokens,
                task_type='sample_admission',
            ))

        responses = await _await_generation(self.sampler, submission_id)
        rows, tags = _build_rollout_rows_and_tags(
            _to_sample_response_models(responses),
            group_ids=body.group_ids,
            policy_version=body.policy_version,
            adapter_uri=body.adapter_uri,
        )
        return await self.data_plane.put(
            [json_safe(item) for item in rows],
            kind='rollout',
            tags=tags,
        )

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
            unload(resolved_paths)
        return {'status': 'ok'}

    @app.post('/twinkle/set_template', response_model=types.SetTemplateResponse)
    async def set_template(
            request: Request,
            body: types.SetTemplateRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> types.SetTemplateResponse:
        """Set the chat template for encoding Trajectory inputs."""
        extra_kwargs = body.model_extra or {}
        with traced_operation('sampler.set_template'):
            self.sampler.set_template(body.template_cls, **extra_kwargs)
        return types.SetTemplateResponse()

    @app.post('/twinkle/add_adapter_to_sampler', response_model=types.AddAdapterResponse)
    async def add_adapter_to_sampler(
            request: Request,
            body: types.AddAdapterRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> types.AddAdapterResponse:
        """Add a LoRA adapter to the sampler."""
        assert body.adapter_name, 'You need to specify a valid `adapter_name`'
        full_adapter_name = _get_twinkle_sampler_adapter_name(request, body.adapter_name)

        from peft import LoraConfig
        config = LoraConfig(**body.config) if isinstance(body.config, dict) else body.config

        with traced_operation('sampler.add_adapter_to_sampler', attrs={MODEL_ID: self.model_id}):
            self.sampler.add_adapter_to_sampler(full_adapter_name, config)

        return types.AddAdapterResponse(adapter_name=full_adapter_name)

    @app.post('/twinkle/apply_patch')
    async def apply_patch(
            request: Request,
            body: types.ApplyPatchRequest,
            self: SamplerManagement = Depends(self_fn),
    ) -> None:
        extra_kwargs = body.model_extra or {}
        patch_cls = deserialize_object(body.patch_cls)
        with traced_operation('sampler.apply_patch'):
            self.sampler.apply_patch(patch_cls, **extra_kwargs)

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
            import os

            from twinkle.server.checkpoint import create_checkpoint_manager
            checkpoint_manager = create_checkpoint_manager(token, client_type='twinkle')
            _, resolved_uri = checkpoint_manager.parse_adapter_uri(body.adapter_uri)
            self.sampler.reset_prefix_cache()
            if resolved_uri and os.path.exists(os.path.join(resolved_uri, 'adapter_config.json')):
                adapter_path = resolved_uri
            elif resolved_uri:
                self.sampler.load_full_weights_from_path(resolved_uri)

        inputs = body.inputs
        if isinstance(inputs, list):
            if len(inputs) != 1:
                raise HTTPException(status_code=400, detail='Streaming only supports a single input')
            inputs = inputs[0]
        if isinstance(inputs, dict):
            if 'input_ids' in inputs:
                inputs_parsed = InputFeature(**inputs)
            else:
                inputs_parsed = Trajectory(**inputs)
        else:
            inputs_parsed = inputs

        params = None
        if body.sampling_params:
            params = SamplingParams.from_dict(body.sampling_params)

        from ray.util.queue import Queue

        from .backends import STREAM_SENTINEL

        q = Queue(maxsize=128)
        actor = self.sampler._actors[0]
        actor.sample_stream_to_queue.remote(
            q,
            inputs_parsed,
            params,
            adapter_name=full_adapter_name,
            adapter_path=adapter_path,
        )

        async def _stream_generator():
            loop = asyncio.get_event_loop()
            while True:
                item = await loop.run_in_executor(None, q.get)
                if item == STREAM_SENTINEL:
                    break
                if isinstance(item, Exception):
                    yield json.dumps({'error': str(item)}) + '\n'
                    break
                delta, reason = item
                yield json.dumps({'delta': delta, 'finish_reason': reason}) + '\n'

        return StreamingResponse(_stream_generator(), media_type='application/x-ndjson')
