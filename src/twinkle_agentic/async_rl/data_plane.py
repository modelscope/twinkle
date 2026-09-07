# Copyright (c) ModelScope Contributors. All rights reserved.
"""The only async-RL layer that speaks native TransferQueue BatchMeta."""

from __future__ import annotations

from typing import Any, Sequence

from .native_tq import (AsyncTQClient, append_fields, batch_size_for_groups, clear_partition, fetch_ready_batch,
                        metadata_size, preallocate_partition, set_sample_tags, split_batch_meta)
from .tq_utils import REQUIRED_MODEL_INPUT_FIELDS, ROLLOUT_TRAIN_FIELDS, columns_to_tq_fields, rows_to_tq_fields
from .types import ClaimedBatch, LoraContext, PartitionAdmission, PreparedPartition, PromptGroup, RolloutOutput

_REQUIRED_ROLLOUT_FIELDS = frozenset((*REQUIRED_MODEL_INPUT_FIELDS, 'logprobs', 'rewards'))


def build_rollout_group_sample_write(
    group: PromptGroup,
    samples: Sequence[RolloutOutput],
    *,
    rewards: list[float] | None = None,
    expected_num_generations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    group_samples = [dict(sample) for sample in samples]
    if expected_num_generations <= 0:
        raise ValueError(f'expected_num_generations must be positive, got {expected_num_generations}')
    if len(group_samples) != expected_num_generations:
        raise ValueError(f'group {group.group_id} expected {expected_num_generations} rollout samples, '
                         f'got {len(group_samples)}')
    if rewards is not None and len(rewards) != len(group_samples):
        raise ValueError(f'reward count {len(rewards)} does not match sample count {len(group_samples)}')

    sample_fields: list[dict[str, Any]] = []
    sample_tags: list[dict[str, Any]] = []
    generation_indices: list[int] = []
    reward_iter = iter(rewards or [])
    for sample_index, trajectory in enumerate(group_samples):
        sample = dict(trajectory)
        if rewards is not None:
            sample['rewards'] = float(next(reward_iter))
        generation_idx = int(sample.get('generation_idx', sample_index))
        sample_key = f'samples/{group.group_id}/{generation_idx}'
        generation_indices.append(generation_idx)
        logprobs = _require_rollout_logprobs(sample, sample_key=sample_key)
        sample['logprobs'] = logprobs
        sample_fields.append(_rollout_sample_fields(sample))
        sample_tags.append(
            _sample_tag(
                context=group.context,
                group=group,
                sample=sample,
                sample_key=sample_key,
                generation_idx=generation_idx,
                logprobs=logprobs,
            ))

    expected_indices = list(range(expected_num_generations))
    if generation_indices != expected_indices:
        raise ValueError(f'group {group.group_id} generation_idx must be 0..{expected_num_generations - 1} '
                         f'in order, got {generation_indices}')
    return sample_fields, sample_tags


def _require_rollout_logprobs(sample: dict[str, Any], *, sample_key: str) -> list[float]:
    logprobs = sample.get('logprobs')
    if not isinstance(logprobs, list):
        raise TypeError(f'rollout sample {sample_key!r} logprobs must be list[float], got {type(logprobs)!r}')
    values: list[float] = []
    for index, value in enumerate(logprobs):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f'rollout sample {sample_key!r} logprobs[{index}] must be a float, got {type(value)!r}')
        values.append(float(value))
    labels = sample.get('labels')
    if labels is not None:
        trainable_tokens = sum(1 for label in labels if label != -100)
        if len(values) != trainable_tokens:
            raise ValueError(f'rollout sample {sample_key!r} logprobs length must match trainable labels: '
                             f'{len(values)} != {trainable_tokens}')
    return values


def _rollout_sample_fields(sample: dict[str, Any]) -> dict[str, Any]:
    return {field_name: sample[field_name] for field_name in ROLLOUT_TRAIN_FIELDS if field_name in sample}


def _sample_tag(
    *,
    context: LoraContext,
    group: PromptGroup,
    sample: dict[str, Any],
    sample_key: str,
    generation_idx: int,
    logprobs: list[float],
) -> dict[str, Any]:
    tag = {
        'record_type': 'sample',
        'sample_status': 'success',
        'context_key': context.key,
        'tenant_id': context.tenant_id,
        'training_run_id': context.training_run_id,
        'adapter_name': context.adapter_name,
        'sample_id': sample.get('sample_id', sample_key),
        'group_id': group.group_id,
        'generation_idx': generation_idx,
        'rollout_policy_version': int(sample['rollout_policy_version']),
        'rollout_adapter_path': sample.get('rollout_adapter_path'),
        'logprobs_length': len(logprobs),
    }
    for field_name in ('rollout_policy_versions', 'initial_policy_version', 'final_policy_version',
                       'policy_version_span'):
        if field_name in sample:
            tag[field_name] = sample[field_name]
    trainable_tokens = _trainable_token_count(sample.get('labels'))
    if trainable_tokens is not None:
        tag['trainable_tokens'] = trainable_tokens
    for sample_field, tag_field in (
        ('input_ids', 'input_length'),
        ('labels', 'label_length'),
        ('attention_mask', 'attention_length'),
    ):
        length = _safe_len(sample.get(sample_field))
        if length is not None:
            tag[tag_field] = length
    for field_name in ('stop_reason', 'truncated', 'turns'):
        if field_name in sample:
            tag[field_name] = sample[field_name]
    if 'completion_length' in sample:
        tag['completion_length'] = int(sample['completion_length'])
    return tag


def _trainable_token_count(labels: Any) -> int | None:
    if labels is None:
        return None
    return sum(1 for label in labels if label != -100)


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


class TQDataPlane:
    """Maps partition-level training operations to native TQ operations."""

    def __init__(self, client: AsyncTQClient | None = None):
        self._client = client

    @property
    def client(self) -> AsyncTQClient:
        if self._client is None:
            import transfer_queue as tq
            tq.init()
            self._client = tq.get_client()
        return self._client

    async def prepare_rollout_partition(
        self,
        admission: PartitionAdmission,
        prompts: Sequence[dict[str, Any]],
        sampling_params: Any,
    ) -> PreparedPartition:
        if len(prompts) != admission.target_groups:
            raise ValueError(f'{admission.partition_id} expected {admission.target_groups} prompts, got {len(prompts)}')
        rows = [dict(prompt) for prompt in prompts for _ in range(admission.num_generations)]
        metadata = await preallocate_partition(
            self.client, partition_id=admission.partition_id, prompt_fields=rows_to_tq_fields(rows))
        group_batch_metas = split_batch_meta(metadata, admission.num_generations)
        groups = []
        for index, (prompt, batch_meta) in enumerate(zip(prompts, group_batch_metas)):
            group_id = f'{admission.partition_id}/group_{index}'
            await set_sample_tags(self.client, batch_meta, [{
                'group_id': group_id,
                'generation_idx': generation_idx,
                'rollout_status': 'PENDING',
            } for generation_idx in range(admission.num_generations)])
            groups.append(
                PromptGroup(
                    context=admission.context,
                    partition=admission,
                    group_id=group_id,
                    prompt=dict(prompt),
                    batch_meta=batch_meta,
                ))
        return PreparedPartition(admission, tuple(groups), sampling_params)

    async def complete_rollout_group(
        self,
        group: PromptGroup,
        *,
        rollout_rows: Sequence[RolloutOutput],
        rewards: Sequence[float],
        submission_id: str,
        tag_metrics: dict[str, Any] | None = None,
    ) -> None:
        expected = group.partition.num_generations
        sample_fields, sample_tags = build_rollout_group_sample_write(
            group,
            rollout_rows,
            rewards=list(rewards),
            expected_num_generations=expected,
        )
        for index, fields in enumerate(sample_fields):
            missing = sorted(_REQUIRED_ROLLOUT_FIELDS - set(fields))
            if missing:
                raise ValueError(f'rollout sample {group.group_id}/{index} is missing training fields {missing}')
        metrics = dict(tag_metrics or {})
        completed_tags = []
        for tag in sample_tags:
            completed_tag = dict(tag)
            completed_tag.update(metrics)
            completed_tag.update({'rollout_status': 'ROLLOUT_DONE', 'submission_id': submission_id})
            completed_tags.append(completed_tag)
        await set_sample_tags(self.client, group.batch_meta, completed_tags)
        await append_fields(self.client, rows_to_tq_fields(sample_fields), group.batch_meta)

    async def claim_advantage_batch(self, admission: PartitionAdmission, group_count: int) -> ClaimedBatch | None:
        metadata = await self._claim(admission, group_count, ['input_ids', 'logprobs', 'rewards'],
                                     self._advantage_task(admission))
        if metadata is None:
            return None
        return ClaimedBatch(
            admission=admission,
            data=await self.client.async_get_data(metadata.select_fields(['rewards'])),
            batch_meta=metadata,
        )

    async def write_advantages(self, batch: ClaimedBatch, *, advantages: Any, returns: Any) -> None:
        size = metadata_size(batch.batch_meta)
        fields = columns_to_tq_fields({'advantages': list(advantages), 'returns': list(returns)}, size)
        await append_fields(self.client, fields, batch.batch_meta)

    async def claim_training_batch(self, admission: PartitionAdmission, group_count: int) -> ClaimedBatch | None:
        metadata = await self._claim(
            admission,
            group_count,
            [*REQUIRED_MODEL_INPUT_FIELDS, 'logprobs', 'rewards', 'advantages', 'returns'],
            self._trainer_task(admission),
        )
        if metadata is None:
            return None
        return ClaimedBatch(
            admission=admission,
            data=await self.client.async_get_data(metadata),
            batch_meta=metadata,
            sample_tags=tuple(metadata.get_all_custom_meta()),
        )

    async def is_training_consumed(self, admission: PartitionAdmission) -> bool:
        return await self.client.async_check_consumption_status(self._trainer_task(admission), admission.partition_id)

    async def clear_partition(self, admission: PartitionAdmission) -> None:
        await clear_partition(self.client, admission.partition_id)

    async def _claim(self, admission: PartitionAdmission, groups: int, fields: list[str], task: str) -> Any | None:
        return await fetch_ready_batch(
            self.client,
            data_fields=fields,
            batch_size=batch_size_for_groups(groups, admission.num_generations),
            partition_id=admission.partition_id,
            task_name=task,
            num_generations=admission.num_generations,
        )

    @staticmethod
    def _advantage_task(admission: PartitionAdmission) -> str:
        return f'async_rl/advantage/{admission.context.key}'

    @staticmethod
    def _trainer_task(admission: PartitionAdmission) -> str:
        return f'async_rl/trainer/{admission.context.key}'
