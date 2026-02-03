# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
import numpy as np
from pathlib import Path

import twinkle
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle.processor import InputProcessor
from twinkle.data_format import Message
from twinkle import DeviceMesh

twinkle.initialize(mode='local')

TEST_DATA_DIR = Path(__file__).parent.parent / "dataset" / "test_data"
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def convert_to_messages(example):
    text = example.get('text', '')
    return {
        'messages': [
            Message(role='user', content=text),
            Message(role='assistant', content='Response')
        ]
    }


class TestDataLoaderBasic:
    def test_dataloader_basic(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        dataloader = DataLoader(dataset=dataset, batch_size=2)
        
        assert len(dataloader) == 2
        
        batches = list(dataloader)
        assert len(batches) == 2
        assert len(batches[0]) == 2

    def test_dataloader_with_dataset_callable(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        
        def create_dataset():
            return Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        dataloader = DataLoader(dataset=create_dataset, batch_size=2)
        
        assert len(dataloader) == 2
        batches = list(dataloader)
        assert len(batches) == 2


class TestDataCollator:
    """测试data_collator（InputProcessor）功能"""
    
    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_dataloader_with_collator(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        dataloader = DataLoader(dataset=dataset, batch_size=2)
        dataloader.set_processor(InputProcessor, padding_side='right')
        
        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 2

    def test_dataloader_without_collator(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        dataloader = DataLoader(dataset=dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        assert isinstance(batch, list)
        assert len(batch) == 2

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_collator_padding_side(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        dataloader_right = DataLoader(dataset=dataset, batch_size=2)
        dataloader_right.set_processor(InputProcessor, padding_side='right')
        
        batch_right = next(iter(dataloader_right))
        assert 'input_ids' in batch_right
        assert 'attention_mask' in batch_right


class TestDeviceMeshSampler:
    def test_device_mesh_sampler_basic(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        device_mesh = DeviceMesh(
            device_type='cpu',
            mesh=np.array([0, 1]),
            mesh_dim_names=('dp',)
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=4,
            device_mesh=device_mesh
        )
        
        batches = list(dataloader)
        assert len(batches) > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_device_mesh_sampler_with_encode(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        device_mesh = DeviceMesh(
            device_type='cpu',
            mesh=np.array([0, 1]),
            mesh_dim_names=('dp',)
        )
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=4,
            device_mesh=device_mesh
        )
        dataloader.set_processor(InputProcessor, padding_side='right')
        
        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 2


class TestRetrySampler:
    def test_retry_sampler_with_valid_data(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=2,
            max_retries=5
        )
        
        batches = list(dataloader)
        assert len(batches) == 2

    def test_retry_sampler_補齐长度(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        original_len = len(dataset)
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=2,
            max_retries=10
        )
        
        total_samples = sum(len(batch) for batch in dataloader)
        assert total_samples == original_len
