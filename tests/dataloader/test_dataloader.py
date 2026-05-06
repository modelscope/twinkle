# Copyright (c) ModelScope Contributors. All rights reserved.
import concurrent.futures
import numpy as np
import os
import pytest
from pathlib import Path
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import IterableDataset as TorchIterableDataset
from unittest.mock import MagicMock

import twinkle
import twinkle.hub.hub as _hub_module
from twinkle import DeviceMesh
from twinkle.data_format import Message
from twinkle.dataloader import DataLoader
from twinkle.dataset import Dataset, DatasetMeta, IterableDataset
from twinkle.processor import InputProcessor

twinkle.initialize(mode='local')


@pytest.fixture(autouse=True)
def _disable_process_pool(monkeypatch):
    mock_executor = MagicMock()
    mock_executor.submit.side_effect = RuntimeError('Process pool is disabled in this test environment.')
    monkeypatch.setattr(_hub_module, '_executor', mock_executor)


TEST_DATA_DIR = Path(__file__).parent.parent / 'dataset' / 'test_data'
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def convert_to_messages(example):
    text = example.get('text', '')
    return {'messages': [Message(role='user', content=text), Message(role='assistant', content='Response')]}


def _build_resume_rows():
    return [
        {
            'text': 'Hello world'
        },
        {
            'text': 'Test data'
        },
        {
            'text': 'Another example'
        },
        {
            'text': 'Sample text'
        },
    ]


class _InMemoryDataset(TorchDataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


class _InMemoryIterableDataset(TorchIterableDataset):

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class TestDataLoaderBasic:

    def test_dataloader_basic(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        dataloader = DataLoader(dataset=dataset, batch_size=2)

        assert len(dataloader) == 2

        batches = list(dataloader)
        assert len(batches) == 2
        assert len(batches[0]) == 2

    def test_dataloader_with_dataset_callable(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')

        def create_dataset():
            return Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        dataloader = DataLoader(dataset=create_dataset, batch_size=2)

        assert len(dataloader) == 2
        batches = list(dataloader)
        assert len(batches) == 2


class TestDataCollator:
    """Test data_collator (InputProcessor) functionality"""

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_dataloader_with_collator(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f'Failed to setup dataset (may need network): {e}')

        dataloader = DataLoader(dataset=dataset, batch_size=2)
        dataloader.set_processor(InputProcessor, padding_side='right')

        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 2

    def test_dataloader_without_collator(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        dataloader = DataLoader(dataset=dataset, batch_size=2)

        batch = next(iter(dataloader))
        assert isinstance(batch, list)
        assert len(batch) == 2

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_collator_padding_side(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f'Failed to setup dataset (may need network): {e}')

        dataloader_right = DataLoader(dataset=dataset, batch_size=2)
        dataloader_right.set_processor(InputProcessor, padding_side='right')

        batch_right = next(iter(dataloader_right))
        assert 'input_ids' in batch_right
        assert 'attention_mask' in batch_right


class TestDeviceMeshSampler:

    def test_device_mesh_sampler_basic(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        device_mesh = DeviceMesh(device_type='cpu', mesh=np.array([0, 1]), mesh_dim_names=('dp', ))

        dataloader = DataLoader(dataset=dataset, batch_size=4, device_mesh=device_mesh)

        batches = list(dataloader)
        assert len(batches) > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_device_mesh_sampler_with_encode(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f'Failed to setup dataset (may need network): {e}')

        device_mesh = DeviceMesh(device_type='cpu', mesh=np.array([0, 1]), mesh_dim_names=('dp', ))

        dataloader = DataLoader(dataset=dataset, batch_size=4, device_mesh=device_mesh)
        dataloader.set_processor(InputProcessor, padding_side='right')

        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 2


class TestRetrySampler:

    def test_retry_sampler_with_valid_data(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        dataloader = DataLoader(dataset=dataset, batch_size=2, max_retries=5)

        batches = list(dataloader)
        assert len(batches) == 2

    def test_retry_sampler_length(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        original_len = len(dataset)

        dataloader = DataLoader(dataset=dataset, batch_size=2, max_retries=10)

        total_samples = sum(len(batch) for batch in dataloader)
        assert total_samples == original_len


class TestResumeSkip:

    def test_dataloader_skip_consumed_samples_for_map_style_dataset(self):
        dataset = _InMemoryDataset(_build_resume_rows())
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)

        dataloader.skip_consumed_samples(2)
        batches = list(dataloader)

        texts = [item['text'] for batch in batches for item in batch]
        assert texts[0] == 'Another example'

    def test_dataloader_warns_when_skip_requested_for_iterable_dataset(self, recwarn):
        dataset = _InMemoryIterableDataset(_build_resume_rows())
        dataloader = DataLoader(dataset=dataset, batch_size=2, num_workers=0)

        dataloader.skip_consumed_samples(2)
        next(iter(dataloader))

        assert 'does not support consumed-data skipping' in str(recwarn.pop(UserWarning).message)
