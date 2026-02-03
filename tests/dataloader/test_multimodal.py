# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
from pathlib import Path

import twinkle
from twinkle.dataset import Dataset, LazyDataset, DatasetMeta
from twinkle.dataloader import DataLoader
from twinkle.processor import InputProcessor

twinkle.initialize(mode='local')

TEST_DATA_DIR = Path(__file__).parent.parent / "dataset" / "test_data"
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def create_multimodal_messages(example):
    text = example.get('text', '')
    return {
        'messages': [
            {'role': 'user', 'content': f'<image>\n{text}'},
            {'role': 'assistant', 'content': 'Response'}
        ]
    }


class TestDataLoaderMultimodal:
    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_dataloader_multimodal_with_lazy_dataset(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(create_multimodal_messages)
        
        try:
            dataset.set_template('Qwen3VLTemplate', model_id='Qwen/Qwen2-VL-7B-Instruct')
        except Exception as e:
            pytest.skip(f"Failed to load Qwen3VLTemplate (may need network): {e}")
        
        dataset.encode()
        
        dataloader = DataLoader(dataset=dataset, batch_size=2)
        dataloader.set_processor(InputProcessor, padding_side='right')
        
        batch = next(iter(dataloader))
        assert 'input_ids' in batch
        assert batch['input_ids'].shape[0] == 2

    def test_dataloader_multimodal_placeholder(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(create_multimodal_messages)
        
        dataloader = DataLoader(dataset=dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        assert len(batch) == 2
        assert 'messages' in batch[0]
        user_content = batch[0]['messages'][0]['content']
        assert '<image>' in user_content
