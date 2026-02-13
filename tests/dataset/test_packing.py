# Copyright (c) ModelScope Contributors. All rights reserved.
"""Test dataset packing: normal packing, iterable packing (cyclic=True/False)"""
import os
import pytest
from pathlib import Path

try:
    import binpacking  # noqa: F401
    HAS_BINPACKING = True
except ImportError:
    HAS_BINPACKING = False

from twinkle.data_format import Message
from twinkle.dataset import DatasetMeta, IterablePackingDataset, PackingDataset

TEST_DATA_DIR = Path(__file__).parent / 'test_data'
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def convert_to_messages(example):
    text = example.get('text', '') or str(example.get('question', example.get('title', '')))
    return {'messages': [Message(role='user', content=text), Message(role='assistant', content='Response')]}


@pytest.mark.skipif(not HAS_BINPACKING, reason='binpacking not installed')
@pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
class TestPackingDataset:
    """Normal packing"""

    def test_packing_dataset_basic(self):
        """encode -> pack_dataset -> index packed samples"""
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = PackingDataset(dataset_meta=DatasetMeta(dataset_id=csv_path), packing_num_proc=1)
        dataset.map(convert_to_messages)
        dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=64)
        dataset.encode(batched=True, load_from_cache_file=False)
        dataset.pack_dataset()

        assert len(dataset) >= 1
        sample = dataset[0]
        assert 'input_ids' in sample
        assert len(sample['input_ids']) > 0
        assert len(sample['input_ids']) <= 64  # Each pack <= max_length


@pytest.mark.skipif(not HAS_BINPACKING, reason='binpacking not installed')
@pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
class TestIterablePackingDataset:
    """Iterable packing (cyclic=True/False)"""

    def _iter_take(self, dataset, n: int):
        items = []
        for i, item in enumerate(dataset):
            items.append(item)
            if i >= n - 1:
                break
        return items

    def test_iterable_packing_cyclic_false(self):
        """cyclic=False: stop when dataset exhausted"""
        jsonl_path = str(TEST_DATA_DIR / 'packing_messages.jsonl')
        dataset = IterablePackingDataset(
            dataset_meta=DatasetMeta(dataset_id=jsonl_path),
            packing_interval=8,
            cyclic=False,
            packing_num_proc=1,
        )
        dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=64)
        dataset.pack_dataset()

        items = self._iter_take(dataset, 4)
        assert len(items) >= 1
        assert 'input_ids' in items[0]

    def test_iterable_packing_cyclic_true(self):
        """cyclic=True: cycle from start when exhausted, can yield more than original count"""
        jsonl_path = str(TEST_DATA_DIR / 'packing_messages.jsonl')
        dataset = IterablePackingDataset(
            dataset_meta=DatasetMeta(dataset_id=jsonl_path),
            packing_interval=4,
            cyclic=True,
            packing_num_proc=1,
        )
        dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=64)
        dataset.pack_dataset()

        items = self._iter_take(dataset, 6)
        assert len(items) >= 1
        assert 'input_ids' in items[0]
