# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from concurrent.futures import Future
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from twinkle import DeviceMesh, get_logger, remote_class, remote_function
from twinkle.data_format import SampleResponse, SamplingParams, user_data_get
from twinkle.metric import MetricBuffer, MetricRecord
from twinkle.sampler.vllm_sampler import vLLMSampler
from .data_plane import TQDataPlane
from .generation_submissions import GenerationSubmissionMixin
from .metrics import rollout_metrics
from .types import PromptGroup, RLContext, RolloutOutput, RolloutPolicy
from .utils import resolve_adapter_path, sample_responses_to_rollout_rows

logger = get_logger()


def _path_component(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '_', value).strip('._') or 'unknown'


def _compute_rewards(
    reward_registry: dict[str, Any],
    context: RLContext,
    rollout_rows: list[RolloutOutput],
) -> list[float] | None:
    reward_fn = reward_registry.get(context.key)
    if reward_fn is None:
        return None
    return list(reward_fn(rollout_rows, context=context))


def _compute_reward_metrics(
    reward_registry: dict[str, Any],
    context: RLContext,
    rollout_rows: list[RolloutOutput],
    rewards: list[float],
) -> dict[str, Any]:
    reward_fn = reward_registry.get(context.key)
    metric_payload = getattr(reward_fn, 'metric_payload', None)
    if metric_payload is None:
        return {}
    return dict(metric_payload(rollout_rows, rewards=rewards, context=context))


@dataclass(frozen=True)
class _GeneratedSample:
    response: SampleResponse
    policies: tuple[RolloutPolicy, ...]
    attempts: int
    was_aborted: bool
    resumed_partial_output: bool

    @property
    def initial_policy(self) -> RolloutPolicy:
        return self.policies[0]

    @property
    def final_policy(self) -> RolloutPolicy:
        return self.policies[-1]

    @property
    def retry_count(self) -> int:
        return self.attempts - 1


@dataclass(frozen=True)
class _PromptGroupRolloutStats:
    completion_lengths: tuple[int, ...]
    stop_reasons: tuple[str | None, ...]
    policy_versions: tuple[int, ...]


@remote_class()
class VLLMSamplerTQ(vLLMSampler, GenerationSubmissionMixin):
    """vLLM sampler that writes async RL rollout results directly to TransferQueue.

    ``sample()`` is intentionally fire-and-forget: it schedules generation work
    on the sampler actor's vLLM event loop and returns submission metadata
    without waiting for any prompt group to finish.

    ``vLLMSampler`` already mixes in the (re-exported, single-identity) submission mixin, so it MUST stay
    ahead of the explicit mixin base here: listing the mixin first would repeat a base the other parent
    already inherits and fail C3 linearisation at import.
    """

    def __init__(
        self,
        model_id: str,
        engine_args: dict[str, Any] | None = None,
        device_mesh: DeviceMesh | None = None,
        *,
        context_manager: Any | None = None,
        reward_registry: dict[str, Any] | None = None,
        rollout_max_retries: int = 2,
        rollout_retry_delay_s: float = 0.5,
        rollout_output_dir: str | None = None,
        rollout_output_include_token_ids: bool = False,
        **kwargs,
    ):
        self.context_manager = context_manager
        super().__init__(model_id=model_id, engine_args=engine_args, device_mesh=device_mesh, **kwargs)
        # Native YAML async-RL writes rollout groups to TransferQueue. The C/S
        # component mode only uses submit_generation/collect_generation and
        # stores results through the server DataPlane deployment.
        self.data_plane = TQDataPlane() if context_manager is not None else None
        self.reward_registry = dict(reward_registry or {})
        self.rollout_max_retries = int(rollout_max_retries)
        self.rollout_retry_delay_s = float(rollout_retry_delay_s)
        self.rollout_output_dir = (
            Path(rollout_output_dir).expanduser().resolve() if rollout_output_dir is not None else None)
        self.rollout_output_include_token_ids = bool(rollout_output_include_token_ids)
        if self.rollout_max_retries < 0:
            raise ValueError(f'rollout_max_retries must be non-negative, got {self.rollout_max_retries}')
        if self.rollout_retry_delay_s < 0:
            raise ValueError(f'rollout_retry_delay_s must be non-negative, got {self.rollout_retry_delay_s}')
        self._background_submissions: dict[str, Future] = {}
        # Generation submissions are used by the client-orchestrated server
        # path. Unlike ``_background_submissions`` above, their results must
        # remain available until SamplerManagement collects them and writes
        # them to the opaque client DataPlane.
        self._generation_submissions: dict[str, Future[list[SampleResponse]]] = {}
        self.metric_buffer = MetricBuffer()
        self._failure: str | None = None

    def _record_metrics(
        self,
        group: PromptGroup,
        values: dict[str, Any],
        *,
        status: str = 'completed',
        attributes: dict[str, Any] | None = None,
        policy_version: int | None = None,
    ) -> None:
        self.metric_buffer.record(
            MetricRecord(
                stage='rollout',
                values=dict(values),
                context_key=group.context.key,
                partition_id=group.partition_id,
                partition_index=group.partition.step,
                policy_version=policy_version,
                status=status,
                attributes=dict(attributes or {}),
            ))

    @remote_function(dispatch='all', collect='flatten', lazy_collect=False)
    def drain_metric_records(self) -> list[MetricRecord]:
        return self.metric_buffer.drain()

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def check_health(self) -> None:
        if self._failure is not None:
            raise RuntimeError(self._failure)

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def register_reward(self, context_key: str, reward: Any) -> None:
        if context_key in self.reward_registry:
            raise KeyError(f'reward already registered for {context_key}')
        self.reward_registry[context_key] = reward

    @remote_function(dispatch='all', collect='none', lazy_collect=False)
    def unregister_reward(self, context_key: str) -> None:
        self.reward_registry.pop(context_key, None)

    @remote_function(dispatch='slice_dp', collect='none', lazy_collect=False)
    def submit_prompt_groups(
        self,
        groups: list[PromptGroup],
        sampling_params: SamplingParams,
        allow_partial_rollout: bool = False,
    ) -> dict[str, Any]:
        """Schedule this DP worker's complete prompt groups and return immediately."""
        if self.context_manager is None:
            raise RuntimeError('context_manager is required for native TQ prompt-group sampling')
        submission_id = str(uuid.uuid4())
        submitted_at = time.perf_counter()
        future = self._submit_in_loop(
            self._sample_prompt_groups(
                submission_id,
                groups,
                sampling_params,
                bool(allow_partial_rollout),
                submitted_at,
            ))
        self._background_submissions[submission_id] = future
        future.add_done_callback(self._on_submission_done(submission_id))
        return {
            'submission_id': submission_id,
            'submitted_prompt_groups': len(groups),
            'submitted_samples': sum(group.num_samples for group in groups),
        }

    @remote_function(dispatch='slice_dp', collect='flatten', lazy_collect=False)
    def evaluate(
        self,
        inputs: list[dict[str, Any]],
        sampling_params: SamplingParams,
        adapter_name: str,
        adapter_path: str,
    ) -> list[SampleResponse]:
        """Synchronously evaluate one adapter without writing results to TQ."""
        return super().sample(
            inputs,
            sampling_params,
            adapter_name=adapter_name,
            adapter_path=adapter_path,
        )

    def _on_submission_done(self, submission_id: str):

        def callback(future: Future) -> None:
            self._background_submissions.pop(submission_id, None)
            error = future.exception()
            if error is not None:
                self._failure = f'{type(error).__name__}: {error}'
                logger.warning('VLLMSamplerTQ background submission failed: submission=%s error=%s', submission_id,
                               error)

        return callback

    async def _sample_prompt_groups(
        self,
        submission_id: str,
        groups: list[PromptGroup],
        sampling_params: SamplingParams,
        allow_partial_rollout: bool,
        submitted_at: float,
    ) -> None:
        results = await asyncio.gather(
            *(self._run_prompt_group(
                submission_id=submission_id,
                group=group,
                sampling_params=sampling_params,
                allow_partial_rollout=allow_partial_rollout,
            ) for group in groups),
            return_exceptions=True)
        failed_group = next(
            ((group, result) for group, result in zip(groups, results) if isinstance(result, Exception)), None)
        if failed_group is not None:
            group, error = failed_group
            self._record_metrics(
                group,
                {},
                status='failed',
                attributes={
                    'scope': 'group',
                    'group_id': group.group_id,
                    'error': str(error)
                },
            )
            raise RuntimeError(f'rollout failed for {group.group_id}: {error}') from error

        rollout_stats = [result for result in results if isinstance(result, _PromptGroupRolloutStats)]
        metric_rows = [{
            'completion_length': completion_length,
            'stop_reason': stop_reason,
        } for stats in rollout_stats
                       for completion_length, stop_reason in zip(stats.completion_lengths, stats.stop_reasons)]
        policy_versions = [version for stats in rollout_stats for version in stats.policy_versions]
        first_group = groups[0]
        dp_size = self.device_mesh.dp_world_size or 1
        self._record_metrics(
            first_group,
            {
                'prompt_group_count':
                len(groups),
                **rollout_metrics(
                    completion_lengths=[row['completion_length'] for row in metric_rows],
                    stop_reasons=[row['stop_reason'] for row in metric_rows],
                    rollout_latency_s=time.perf_counter() - submitted_at,
                ),
                'policy_version_min':
                min(policy_versions),
                'policy_version_max':
                max(policy_versions),
                'sampler_dp_size':
                dp_size,
            },
            attributes={'scope': 'partition' if dp_size == 1 else 'shard'},
            policy_version=max(policy_versions),
        )

    async def _run_prompt_group(
        self,
        *,
        submission_id: str,
        group: PromptGroup,
        sampling_params: SamplingParams,
        allow_partial_rollout: bool,
    ) -> _PromptGroupRolloutStats:
        """Sample all generations for one group, then write that group once."""
        started = asyncio.get_running_loop().time()
        num_generations = group.num_samples
        sources = [{
            **group.prompt, 'group_id': group.group_id,
            'generation_idx': generation_idx
        } for generation_idx in range(num_generations)]
        generated_samples = await self._generate_group_samples(
            group.context, sources, sampling_params, allow_partial_rollout=allow_partial_rollout)
        rows = []
        for source, generated in zip(sources, generated_samples):
            sample_rows = sample_responses_to_rollout_rows([source], [generated.response],
                                                           policy_version=generated.final_policy.version)
            if len(sample_rows) != 1:
                raise ValueError(f'generation {source["generation_idx"]} produced {len(sample_rows)} samples')
            row = sample_rows[0]
            versions = [policy.version for policy in generated.policies]
            row.update({
                'rollout_policy_version': generated.final_policy.version,
                'rollout_adapter_path': generated.final_policy.adapter_path,
                'rollout_policy_versions': versions,
                'initial_policy_version': generated.initial_policy.version,
                'final_policy_version': generated.final_policy.version,
                'policy_version_span': generated.final_policy.version - generated.initial_policy.version,
            })
            rows.append(row)
        if len(rows) != num_generations:
            raise ValueError(f'group {group.group_id} expected {num_generations} rollout samples, got {len(rows)}')

        rewards = _compute_rewards(self.reward_registry, group.context, rows)
        if rewards is None:
            raise ValueError(f'no reward function registered for context {group.context.key}')
        reward_metrics = _compute_reward_metrics(self.reward_registry, group.context, rows, rewards)
        if self.data_plane is None:
            raise RuntimeError('native TQ data plane is required for prompt-group sampling')
        await self.data_plane.complete_rollout_group(
            group,
            rollout_rows=rows,
            rewards=rewards,
            submission_id=submission_id,
            tag_metrics=reward_metrics,
        )

        rollout_latency_s = asyncio.get_running_loop().time() - started
        policy_versions = [policy.version for sample in generated_samples for policy in sample.policies]
        if self.rollout_output_dir is not None:
            try:
                await asyncio.to_thread(
                    self._write_rollout_group,
                    submission_id,
                    group,
                    generated_samples,
                    rows,
                    rewards,
                )
            except Exception as error:
                logger.warning('Failed to write rollout output for %s: %s', group.group_id, error)
        self._record_metrics(
            group,
            {
                **rollout_metrics(
                    rewards={'reward': rewards},
                    completion_lengths=[int(row['completion_length']) for row in rows],
                    stop_reasons=[row.get('stop_reason') for row in rows],
                    rollout_latency_s=rollout_latency_s,
                ),
                'retry_count':
                sum(sample.retry_count for sample in generated_samples),
                'aborted_sample_count':
                sum(sample.was_aborted for sample in generated_samples),
                'partial_resumed_sample_count':
                sum(sample.resumed_partial_output for sample in generated_samples),
                'policy_version_min':
                min(policy_versions),
                'policy_version_max':
                max(policy_versions),
                **reward_metrics,
            },
            attributes={
                'scope': 'group',
                'group_id': group.group_id
            },
            policy_version=max(policy_versions),
        )
        return _PromptGroupRolloutStats(
            completion_lengths=tuple(int(row['completion_length']) for row in rows),
            stop_reasons=tuple(row.get('stop_reason') for row in rows),
            policy_versions=tuple(policy_versions),
        )

    def _write_rollout_group(
        self,
        submission_id: str,
        group: PromptGroup,
        generated_samples: list[_GeneratedSample],
        rows: list[RolloutOutput],
        rewards: list[float],
    ) -> None:
        policy_version = max(int(row['rollout_policy_version']) for row in rows)
        partition_name = _path_component(group.partition_id.rsplit('/', 1)[-1])
        group_name = _path_component(group.group_id.rsplit('/', 1)[-1])
        output_dir = self.rollout_output_dir.joinpath(
            _path_component(group.context.tenant_id),
            _path_component(group.context.training_run_id),
            _path_component(group.context.policy_slot),
            f'policy_{policy_version}',
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{partition_name}-{group_name}.jsonl'
        temporary_path = output_path.with_suffix(f'.jsonl.{uuid.uuid4().hex}.tmp')
        ground_truth = user_data_get(group.prompt.get('user_data'), 'ground_truth')

        with temporary_path.open('w', encoding='utf-8') as stream:
            for generated, row, reward in zip(generated_samples, rows, rewards):
                response = generated.response
                sequence = response.sequences[0]
                prompt_token_ids = list(response.prompt_token_ids or [])
                completion_token_ids = list(sequence.tokens)
                record = {
                    'submission_id': submission_id,
                    'context_key': group.context.key,
                    'tenant_id': group.context.tenant_id,
                    'training_run_id': group.context.training_run_id,
                    'adapter_name': group.context.adapter_name,
                    'partition_id': group.partition_id,
                    'group_id': group.group_id,
                    'sample_idx': int(row['generation_idx']),
                    'seqlen': len(prompt_token_ids) + len(completion_token_ids),
                    'prompt_len': len(prompt_token_ids),
                    'completion_len': len(completion_token_ids),
                    'head_version': int(row['initial_policy_version']),
                    'tail_version': int(row['final_policy_version']),
                    'policy_versions': list(row['rollout_policy_versions']),
                    'adapter_path': row.get('rollout_adapter_path'),
                    'reward': float(reward),
                    'ground_truth': ground_truth,
                    'stop_reason': row.get('stop_reason'),
                    'retry_count': generated.retry_count,
                    'was_aborted': generated.was_aborted,
                    'resumed_partial_output': generated.resumed_partial_output,
                    'prompt': self.template.decode(prompt_token_ids, skip_special_tokens=False),
                    'completion': self.template.decode(completion_token_ids, skip_special_tokens=False),
                }
                if self.rollout_output_include_token_ids:
                    record.update({
                        'prompt_token_ids': prompt_token_ids,
                        'completion_token_ids': completion_token_ids,
                        'logprobs': list(row['logprobs']),
                    })
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')
        os.replace(temporary_path, output_path)

    async def _load_lora_for_policy(self, policy: RolloutPolicy) -> Any:
        """Load the adapter selected for one group's rollout snapshot."""
        if policy.adapter_path is None:
            return None
        local_path = await asyncio.to_thread(resolve_adapter_path, policy.adapter_path)
        lora_request = await self.engine._get_or_load_lora(local_path)
        if lora_request is None:
            raise RuntimeError(f'failed to load LoRA adapter from {local_path}')
        return lora_request

    async def _generate_group_samples(
        self,
        context: RLContext,
        sources: list[dict[str, Any]],
        sampling_params: SamplingParams,
        *,
        allow_partial_rollout: bool,
    ) -> list[_GeneratedSample]:
        logprobs_only = False
        if sampling_params.max_tokens == 0:
            sampling_params = copy(sampling_params)
            sampling_params.max_tokens = 1
            logprobs_only = True

        is_trajectory = 'input_ids' not in sources[0]
        multi_modal_data_list = [self._extract_multi_modal_data(source) for source in sources]
        if is_trajectory:
            template = self.template
            assert template is not None, 'Use set_template before sampling trajectories'
            encoded_inputs = [
                self.encode_trajectory_for_vllm(source, context.adapter_name or '', not logprobs_only)
                for source in sources
            ]
        else:
            encoded_inputs = sources
        tasks = [
            self._generate_sample(
                context,
                feat,
                sampling_params,
                multi_modal_data=multi_modal_data,
                logprobs_only=logprobs_only,
                allow_partial_rollout=allow_partial_rollout,
            ) for feat, multi_modal_data in zip(encoded_inputs, multi_modal_data_list)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        failures = [result for result in results if isinstance(result, Exception)]
        if failures:
            raise failures[0]
        return results

    async def _generate_sample(
        self,
        context: RLContext,
        original_input: dict[str, Any],
        sampling_params: SamplingParams,
        *,
        multi_modal_data: dict[str, Any] | None,
        logprobs_only: bool,
        allow_partial_rollout: bool,
    ) -> _GeneratedSample:
        """Generate one sample, resuming across aborts via the shared backend-agnostic kernel.

        The retry/resume/merge loop lives in :class:`~twinkle.sampler.partial_rollout.PartialRolloutMixin`
        so vLLM and SGLang share one implementation; what is specific to this TransferQueue path is the
        per-attempt policy pinning, supplied here as ``run_attempt`` -- acquire the rollout policy for this
        context, sample under its (possibly per-version) adapter, then release the pin. Each segment's
        policy is threaded back out as the kernel's opaque attempt state so the merged sample can report the
        versions it spanned.
        """

        async def run_attempt(current_input: dict[str, Any], attempt_params: SamplingParams,
                              attempt: int) -> tuple[SampleResponse, RolloutPolicy]:
            policy = await self.context_manager.acquire_rollout_policy.remote(context)
            try:
                response = await self._sample_single(
                    current_input,
                    attempt_params,
                    lora_request=await self._load_lora_for_policy(policy),
                    multi_modal_data=multi_modal_data,
                    logprobs_only=logprobs_only,
                )
                return response, policy
            finally:
                await self.context_manager.release_rollout_policy.remote(policy)

        outcome = await self._run_partial_rollout(
            original_input,
            sampling_params,
            run_attempt=run_attempt,
            allow_partial_rollout=allow_partial_rollout,
            max_retries=self.rollout_max_retries,
            retry_delay_s=self.rollout_retry_delay_s,
        )
        return _GeneratedSample(
            outcome.response,
            outcome.attempt_states,
            outcome.attempts,
            outcome.was_aborted,
            outcome.resumed_partial_output,
        )
