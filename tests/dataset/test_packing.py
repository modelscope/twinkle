# Copyright (c) ModelScope Contributors. All rights reserved.
"""测试 dataset packing 功能：普通 packing、iterable packing (cyclic=True/False)"""
import os
import pytest
from pathlib import Path

try:
    import binpacking  # noqa: F401
    HAS_BINPACKING = True
except ImportError:
    HAS_BINPACKING = False

from twinkle.data_format import Message
from twinkle.dataset import PackingDataset, IterablePackingDataset, DatasetMeta

TEST_DATA_DIR = Path(__file__).parent / "test_data"
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def convert_to_messages(example):
    text = example.get('text', '') or str(example.get('question', example.get('title', '')))
    return {'messages': [Message(role='user', content=text), Message(role='assistant', content='Response')]}


@pytest.mark.skipif(not HAS_BINPACKING, reason="binpacking not installed")
@pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
class TestPackingDataset:
    """普通 packing"""

    def test_packing_dataset_basic(self):
        """encode -> pack_dataset -> 索引 packed 样本"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = PackingDataset(dataset_meta=DatasetMeta(dataset_id=csv_path), packing_num_proc=1)
        dataset.map(convert_to_messages)
        dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=64)
        dataset.encode(batched=True, load_from_cache_file=False)
        dataset.pack_dataset()

        assert len(dataset) >= 1
        sample = dataset[0]
        assert 'input_ids' in sample
        assert len(sample['input_ids']) > 0
        assert len(sample['input_ids']) <= 64  # 每包不超过 max_length


@pytest.mark.skipif(not HAS_BINPACKING, reason="binpacking not installed")
@pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
class TestIterablePackingDataset:
    """iterable packing (cyclic=True/False)"""

    def _iter_take(self, dataset, n: int):
        items = []
        for i, item in enumerate(dataset):
            items.append(item)
            if i >= n - 1:
                break
        return items

    def test_iterable_packing_cyclic_false(self):
        """cyclic=False：迭代到数据集结束即停止"""
        jsonl_path = str(TEST_DATA_DIR / "packing_messages.jsonl")
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
        """cyclic=True：数据耗尽后从头循环，可产出超过原始条数"""
        jsonl_path = str(TEST_DATA_DIR / "packing_messages.jsonl")
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
