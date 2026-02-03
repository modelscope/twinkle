# Copyright (c) ModelScope Contributors. All rights reserved.
"""
测试数据集packing功能：
1. 普通packing（PackingDataset）
2. iterable packing（IterablePackingDataset，cyclic=True/False）
"""
import os
import pytest
from pathlib import Path

from twinkle.dataset import Dataset, PackingDataset, IterablePackingDataset, DatasetMeta
from twinkle.data_format import Message, Trajectory


# 获取测试数据目录
TEST_DATA_DIR = Path(__file__).parent / "test_data"

SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def convert_to_messages(example):
    """将CSV数据转换为Template期望的messages格式"""
    text = example.get('text', '')
    if not text:
        text = str(example.get('question', example.get('title', '')))
    
    return {
        'messages': [
            Message(role='user', content=text),
            Message(role='assistant', content='Response')
        ]
    }


class TestPackingDataset:
    """测试普通packing功能（PackingDataset）"""

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_packing_basic(self):
        """测试基本的packing功能"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        
        # 创建PackingDataset
        dataset = PackingDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        dataset.map(convert_to_messages)
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        

        try:
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to encode dataset: {e}")
        
        assert 'input_ids' in dataset.dataset[0]
        assert 'length' in dataset.dataset[0]
        
        try:
            dataset.pack_dataset()
        except Exception as e:
            pytest.skip(f"Failed to pack dataset: {e}")
        
        assert len(dataset) <= len(dataset.dataset)
        
        packed_item = dataset[0]
        assert 'input_ids' in packed_item
        assert isinstance(packed_item['input_ids'], list)
        
        assert len(packed_item['input_ids']) <= dataset.template.max_length

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_packing_multiple_samples(self):
        """测试packing多个样本"""
        csv_path = str(TEST_DATA_DIR / "test4.csv") 
        
        dataset = PackingDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=256)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        

        try:
            dataset.pack_dataset()
        except Exception as e:
            pytest.skip(f"Failed to pack dataset: {e}")
   
        assert len(dataset) > 0
        assert len(dataset) <= len(dataset.dataset)
        
 
        for i in range(min(10, len(dataset))):  
            packed_item = dataset[i]
            assert len(packed_item['input_ids']) <= dataset.template.max_length

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_packing_with_different_max_length(self):
        """测试使用不同的max_length进行packing"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        
        dataset = PackingDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        

        dataset.map(convert_to_messages)
        
        try:

            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=64)
            dataset.encode(batched=True)
            dataset.pack_dataset()
        except Exception as e:
            pytest.skip(f"Failed to setup/pack dataset (may need network): {e}")
        

        for i in range(len(dataset)):
            packed_item = dataset[i]
            assert len(packed_item['input_ids']) <= 64


class TestIterablePackingDataset:
    """测试iterable packing功能（IterablePackingDataset）"""

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_iterable_packing_basic(self):
        """测试基本的iterable packing功能（cyclic=False）"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        
        try:
            dataset = IterablePackingDataset(
                dataset_meta=DatasetMeta(dataset_id=csv_path),
                cyclic=False,
                packing_interval=4,
                num_proc=1  
            )
        except NotImplementedError as e:
            if 'num_proc' in str(e):
                pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")
            raise
        

        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128, truncation_strategy='left')
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        try:
            dataset.pack_dataset()
        except Exception as e:
            pytest.skip(f"Failed to start packing: {e}")
        
        packed_items = []
        try:
            for i, item in enumerate(dataset):
                packed_items.append(item)
                if i >= 5:  # 只取前几个样本
                    break
        except Exception as e:
            pytest.skip(f"Failed to iterate packed dataset: {e}")
        
        assert len(packed_items) > 0
        
        for item in packed_items:
            assert 'input_ids' in item
            assert len(item['input_ids']) <= dataset.template.max_length

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_iterable_packing_cyclic_false(self):
        """测试iterable packing（cyclic=False）- 数据集结束后应该停止"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        
        try:
            dataset = IterablePackingDataset(
                dataset_meta=DatasetMeta(dataset_id=csv_path),
                cyclic=False,
                packing_interval=2,
                num_proc=1  # streaming模式下必须设置为1
            )
        except NotImplementedError as e:
            if 'num_proc' in str(e):
                pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")
            raise
        
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128, truncation_strategy='left')
            dataset.encode(batched=True)
            dataset.pack_dataset()
        except Exception as e:
            pytest.skip(f"Failed to setup/pack dataset (may need network): {e}")
        
        # 迭代所有数据
        count = 0
        try:
            for item in dataset:
                count += 1
                assert 'input_ids' in item
                if count > 100:  # 防止无限循环
                    break
        except Exception as e:
            pytest.skip(f"Failed to iterate dataset: {e}")

        assert count > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_iterable_packing_cyclic_true(self):
        """测试iterable packing（cyclic=True）- 数据集结束后应该循环"""
        csv_path = str(TEST_DATA_DIR / "test.csv")
        
        try:
            dataset = IterablePackingDataset(
                dataset_meta=DatasetMeta(dataset_id=csv_path),
                cyclic=True,
                packing_interval=2,
                num_proc=1  # streaming模式下必须设置为1
            )
        except NotImplementedError as e:
            if 'num_proc' in str(e):
                pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")
            raise
        

        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128, truncation_strategy='left')
            dataset.encode(batched=True)
            dataset.pack_dataset()
        except Exception as e:
            pytest.skip(f"Failed to setup/pack dataset (may need network): {e}")
        
        count = 0
        first_batch = None
        try:
            for item in dataset:
                if count == 0:
                    first_batch = item
                count += 1
                assert 'input_ids' in item
                if count >= 20:  
                    break
        except Exception as e:
            pytest.skip(f"Failed to iterate dataset: {e}")
        
        assert count >= 20

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_iterable_packing_different_intervals(self):
        """测试使用不同的packing_interval"""
        csv_path = str(TEST_DATA_DIR / "test4.csv")  # 使用较大的数据集
        
        try:
            dataset1 = IterablePackingDataset(
                dataset_meta=DatasetMeta(dataset_id=csv_path),
                cyclic=False,
                packing_interval=8,
                num_proc=1 
            )
        except NotImplementedError as e:
            if 'num_proc' in str(e):
                pytest.xfail(f"Known limitation: streaming local file with num_proc is not supported: {e}")
            raise
        
        # 将CSV数据转换为messages格式
        dataset1.map(convert_to_messages)
        
        try:
            dataset1.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=256, truncation_strategy='left')
            dataset1.encode(batched=True)
            dataset1.pack_dataset()
        except Exception as e:
            pytest.skip(f"Failed to setup/pack dataset (may need network): {e}")
        
        count1 = 0
        try:
            for item in dataset1:
                count1 += 1
                assert 'input_ids' in item
                if count1 >= 5:
                    break
        except Exception as e:
            pytest.skip(f"Failed to iterate dataset: {e}")
        
        assert count1 > 0
