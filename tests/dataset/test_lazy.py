# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
from pathlib import Path

from twinkle.dataset import LazyDataset, DatasetMeta
from twinkle.data_format import Message

TEST_DATA_DIR = Path(__file__).parent / "test_data"
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def convert_to_messages(example):
    text = example.get('text', '')
    if not text:
        text = str(example.get('question', example.get('title', '')))
    
    return {
        'messages': [
            Message(role='user', content=text),
            Message(role='assistant', content='Response')
        ]
    }


class TestLazyDataset:
    def test_lazy_dataset_basic(self):
        #基本功能测试 
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        assert len(dataset) == 4
        assert dataset.do_encode == False
        assert dataset.do_check == False
        
        item = dataset[0]
        assert 'text' in item
        assert item['text'] == "Hello world"

    def test_lazy_dataset_encode_flag(self):
        #懒加载编码标志测试 
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        assert dataset.do_encode == False
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        dataset.encode()
        
        assert dataset.do_encode == True
        assert 'messages' in dataset.dataset[0]
        assert 'input_ids' not in dataset.dataset[0]

    def test_lazy_dataset_encode_on_access(self):
        #懒加载编码执行测试
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        dataset.encode()
        
        item = dataset[0]
        assert 'input_ids' in item
        assert 'length' in item
        assert len(item['input_ids']) > 0

    def test_lazy_dataset_check_flag(self):
        #懒加载检查标志测试,验证 check() 只设置标志，不实际执行检查
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        assert dataset.do_check == False
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        dataset.check()
        
        assert dataset.do_check == True

    def test_lazy_dataset_check_on_access(self):
        #懒加载检查执行测试,验证在访问数据时才执行检查
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        dataset.check()
        
        item = dataset[0]
        assert item is not None
        assert 'messages' in item or item is None
        
    def test_lazy_dataset_encode_requires_template(self):
        #编码要求模板测试,验证未设置模板时抛出异常
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        with pytest.raises(AssertionError):
            dataset.encode()

    def test_lazy_dataset_check_requires_template(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        with pytest.raises(AssertionError):
            dataset.check()

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_lazy_dataset_no_split_strategy(self):
        #编码不支持split策略测试,验证未设置模板时抛出异常
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128, truncation_strategy='split')
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        with pytest.raises(AssertionError, match="Lazy tokenize does not support truncation_strategy==`split`"):
            dataset.encode()

    def test_lazy_dataset_multiple_items(self):
        #验证多个数据项的懒加载编码
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        dataset.encode()
        
        for i in range(len(dataset)):
            item = dataset[i]
            assert 'input_ids' in item
            assert len(item['input_ids']) > 0
