# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
from pathlib import Path

from twinkle.data_format import Message
from twinkle.dataset import DatasetMeta, LazyDataset

TEST_DATA_DIR = Path(__file__).parent / 'test_data'
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def convert_to_messages(example):
    text = example.get('text', '')
    if not text:
        text = str(example.get('question', example.get('title', '')))

    return {'messages': [Message(role='user', content=text), Message(role='assistant', content='Response')]}


class TestLazyDataset:

    def test_lazy_dataset_basic(self):
        # Basic functionality test
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        assert len(dataset) == 4
        assert not dataset.do_encode
        assert not dataset.do_check

        item = dataset[0]
        assert 'text' in item
        assert item['text'] == 'Hello world'

    def test_lazy_dataset_encode_flag(self):
        # Lazy encode flag test
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        assert not dataset.do_encode

        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        dataset.encode()

        # Lazy load: encode() only sets flag, actual encoding on access; raw dataset has no input_ids
        assert 'messages' in dataset.dataset[0]
        assert 'input_ids' not in dataset.dataset[0]
        item = dataset[0]
        assert 'input_ids' in item

    def test_lazy_dataset_encode_on_access(self):
        # Lazy encode execution test
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        dataset.encode()

        item = dataset[0]
        assert 'input_ids' in item
        assert 'length' in item
        assert len(item['input_ids']) > 0

    def test_lazy_dataset_check_flag(self):
        # Lazy check flag test: check() only sets flag, does not execute check
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        assert not dataset.do_check

        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        dataset.check()

        # Lazy load: check() only sets flag, actual check on access
        item = dataset[0]
        assert item is not None

    def test_lazy_dataset_check_on_access(self):
        # Lazy check execution test: check runs on data access
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        dataset.check()

        item = dataset[0]
        assert item is not None
        assert 'messages' in item or item is None

    def test_lazy_dataset_encode_requires_template(self):
        # Encode requires template: raises when template not set
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        with pytest.raises(AssertionError):
            dataset.encode()

    def test_lazy_dataset_check_requires_template(self):
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        with pytest.raises(AssertionError):
            dataset.check()

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_lazy_dataset_no_split_strategy(self):
        # Encode does not support split strategy: raises when template not set
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        try:
            dataset.set_template(
                'Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128, truncation_strategy='split')
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        with pytest.raises(AssertionError, match='Lazy tokenize does not support truncation_strategy==`split`'):
            dataset.encode()

    def test_lazy_dataset_multiple_items(self):
        # Lazy encode for multiple items
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)

        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        dataset.encode()

        for i in range(len(dataset)):
            item = dataset[i]
            assert 'input_ids' in item
            assert len(item['input_ids']) > 0
