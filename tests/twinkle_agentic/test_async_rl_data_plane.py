from __future__ import annotations

import asyncio

import pytest

from twinkle_agentic.async_rl import LoraContext, TQDataPlane
from twinkle_agentic.async_rl.data_plane import build_rollout_group_sample_write
from twinkle_agentic.async_rl.types import PartitionAdmission, PromptGroup


def _context() -> LoraContext:
    return LoraContext('tenant', 'run_adapter', 'model', 'adapter')


def test_rollout_sample_tags_use_new_context_descriptor_only():
    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 2, 0)
    group = PromptGroup(context, admission, f'{admission.partition_id}/group_0', {}, batch_meta=None)
    fields, tags = build_rollout_group_sample_write(
        group,
        [
            {
                'generation_idx': 0,
                'labels': [-100, 1],
                'logprobs': [-.1],
                'rollout_policy_version': 3,
                'rollout_adapter_path': 'adapter-v3',
            },
            {
                'generation_idx': 1,
                'labels': [-100, 2],
                'logprobs': [-.2],
                'rollout_policy_version': 4,
                'rollout_adapter_path': 'adapter-v4',
            },
        ],
        rewards=[1., 0.],
        expected_num_generations=2,
    )
    assert [row['rewards'] for row in fields] == [1., 0.]
    assert [tag['generation_idx'] for tag in tags] == [0, 1]
    assert all(tag['context_key'] == context.key for tag in tags)
    assert [tag['rollout_policy_version'] for tag in tags] == [3, 4]


def test_data_plane_completes_rollout_with_full_training_trajectory():
    class Metadata:
        def __init__(self):
            self.size = 2
            self.custom_meta = [{}, {}]

        def update_custom_meta(self, updates):
            for tag, update in zip(self.custom_meta, updates):
                tag.update(update)

    class Client:
        def __init__(self):
            self.written = None
            self.calls = []

        async def async_put(self, data, metadata=None, partition_id=None):
            self.calls.append('fields')
            self.written = data
            return metadata

        async def async_set_custom_meta(self, _metadata):
            self.calls.append('tags')

    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 2, 0)
    metadata = Metadata()
    group = PromptGroup(context, admission, f'{admission.partition_id}/group_0', {}, metadata)
    client = Client()
    rows = [{
        'input_ids': [1, 2, token],
        'labels': [-100, -100, token],
        'attention_mask': [1, 1, 1],
        'position_ids': [0, 1, 2],
        'logprobs': [-.1],
        'generation_idx': generation_idx,
        'rollout_policy_version': 3,
        'rollout_policy_versions': [3],
        'initial_policy_version': 3,
        'final_policy_version': 3,
        'policy_version_span': 0,
        'rollout_adapter_path': 'adapter-v3',
        'completion_length': 1,
    } for generation_idx, token in enumerate((7, 8))]

    asyncio.run(
        TQDataPlane(client).complete_rollout_group(
            group,
            rollout_rows=rows,
            rewards=[1., 0.],
            submission_id='submission',
        ))

    assert set(client.written.keys()) == {
        'input_ids', 'labels', 'attention_mask', 'position_ids', 'logprobs', 'rewards'
    }
    assert client.calls == ['tags', 'fields']
    assert [tag['rollout_status'] for tag in metadata.custom_meta] == ['ROLLOUT_DONE', 'ROLLOUT_DONE']
    assert [tag['submission_id'] for tag in metadata.custom_meta] == ['submission', 'submission']


def test_data_plane_rejects_rollout_without_complete_model_inputs():
    context = _context()
    admission = PartitionAdmission(context, context.partition_id(0), 0, 1, 1, 0)
    metadata = type('Metadata', (), {'size': 1})()
    group = PromptGroup(context, admission, f'{admission.partition_id}/group_0', {}, metadata)
    row = {
        'input_ids': [1, 2],
        'labels': [-100, 2],
        'logprobs': [-.1],
        'generation_idx': 0,
        'rollout_policy_version': 0,
    }

    with pytest.raises(ValueError) as error:
        asyncio.run(
            TQDataPlane(object()).complete_rollout_group(
                group,
                rollout_rows=[row],
                rewards=[1.],
                submission_id='submission',
            ))

    assert 'attention_mask' in str(error.value)
    assert 'position_ids' in str(error.value)
