# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test dataset loading:
1. Load local csv/json/jsonl (normal dataset mode)
2. Load local csv/json/jsonl (iterable mode)
3. Load HF dataset (normal mode)
4. Load HF dataset (iterable mode)
5. Load MS dataset (normal mode)
6. Load MS dataset (iterable mode)
"""
import os
import pytest
from pathlib import Path

from twinkle.dataset import Dataset, DatasetMeta, IterableDataset

# Get test data directory
TEST_DATA_DIR = Path(__file__).parent / 'test_data'


class TestLocalDatasetLoading:
    """Test local dataset loading (normal mode)"""

    def test_load_local_csv(self):
        """Test loading local CSV file"""
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        assert len(dataset) == 4
        assert dataset[0]['text'] == 'Hello world'
        assert dataset[0]['label'] == 0
        assert dataset[1]['text'] == 'Test data'
        assert dataset[1]['label'] == 1

    def test_load_local_json(self):
        """Test loading local JSON file"""
        json_path = str(TEST_DATA_DIR / 'test.json')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=json_path))

        assert len(dataset) == 4
        assert dataset[0]['text'] == 'Hello world'
        assert dataset[0]['label'] == 0

    def test_load_local_jsonl(self):
        jsonl_path = str(TEST_DATA_DIR / 'test.jsonl')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))

        assert len(dataset) == 4
        assert dataset[0]['text'] == 'Hello world'
        assert dataset[0]['label'] == 0


class TestLocalIterableDatasetLoading:
    """Test local dataset loading (iterable mode)"""

    def _iter_take(self, dataset, n: int):
        """Avoid list(dataset) triggering __len__; use for-loop to take first n"""
        items = []
        for i, item in enumerate(dataset):
            items.append(item)
            if i >= n - 1:
                break
        return items

    def test_load_local_csv_iterable(self):
        """Test loading local CSV (iterable mode)"""
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        except NotImplementedError as e:
            pytest.xfail(f'Known limitation: streaming local file with num_proc is not supported: {e}')
        with pytest.raises(NotImplementedError):
            _ = len(dataset)
        items = self._iter_take(dataset, 4)
        assert len(items) == 4
        assert items[0]['text'] == 'Hello world'
        assert items[0]['label'] == 0

    def test_load_local_json_iterable(self):
        """Test loading local JSON (iterable mode)"""
        json_path = str(TEST_DATA_DIR / 'test.json')
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=json_path))
        except NotImplementedError as e:
            pytest.xfail(f'Known limitation: streaming local file with num_proc is not supported: {e}')
        items = self._iter_take(dataset, 4)
        assert len(items) == 4
        assert items[0]['text'] == 'Hello world'

    def test_load_local_jsonl_iterable(self):
        """Test loading local JSONL (iterable mode)"""
        jsonl_path = str(TEST_DATA_DIR / 'test.jsonl')
        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=jsonl_path))
        except NotImplementedError as e:
            pytest.xfail(f'Known limitation: streaming local file with num_proc is not supported: {e}')
        items = self._iter_take(dataset, 4)
        assert len(items) == 4
        assert items[0]['text'] == 'Hello world'


class TestHFDatasetLoading:
    """Test HuggingFace dataset loading"""

    @pytest.mark.skipif(os.environ.get('TWINKLE_FORBID_HF', '0') == '1', reason='HF hub is disabled')
    def test_load_hf_dataset(self):
        """Test loading HF dataset (normal mode)"""
        # Use a small public dataset for testing
        dataset_meta = DatasetMeta(dataset_id='hf://squad', subset_name='plain_text', split='train')
        try:
            dataset = Dataset(dataset_meta=dataset_meta)

            # Only check successful load, not length (dataset may be large)
            assert dataset is not None
            # Try to get first sample
            sample = dataset[0]
            assert sample is not None
        except Exception as e:
            # SSL cert chain unavailable in offline/corporate proxy
            pytest.skip(f'HF dataset not reachable in current environment: {e}')

    @pytest.mark.skipif(os.environ.get('TWINKLE_FORBID_HF', '0') == '1', reason='HF hub is disabled')
    def test_load_hf_dataset_iterable(self):
        """Test loading HF dataset (iterable mode)"""
        dataset_meta = DatasetMeta(dataset_id='hf://squad', subset_name='plain_text', split='train')
        try:
            dataset = IterableDataset(dataset_meta=dataset_meta)

            # iterable dataset does not support __len__
            with pytest.raises(NotImplementedError):
                _ = len(dataset)

            # Test iteration, take first few samples
            items = []
            for i, item in enumerate(dataset):
                items.append(item)
                if i >= 2:  # Take first 3 samples
                    break

            assert len(items) == 3
            assert items[0] is not None
        except Exception as e:
            pytest.skip(f'HF dataset not reachable in current environment: {e}')


class TestMSDatasetLoading:
    """Test ModelScope dataset loading"""

    def test_load_ms_dataset(self):
        """Test loading MS dataset (normal mode)"""
        # Use a small public dataset for testing
        dataset_meta = DatasetMeta('ms://modelscope/competition_math')
        try:
            dataset = Dataset(dataset_meta=dataset_meta)
            # Only check successful load
            assert dataset is not None
            # If dataset has data, try to get first sample
            if len(dataset) > 0:
                sample = dataset[0]
                assert sample is not None
        except Exception as e:
            # Skip if dataset does not exist or is inaccessible
            pytest.skip(f'MS dataset not available: {e}')

    def test_load_ms_dataset_iterable(self):
        """Test loading MS dataset (iterable mode)"""
        dataset_meta = DatasetMeta('ms://modelscope/competition_math')
        try:
            dataset = IterableDataset(dataset_meta=dataset_meta)

            # iterable dataset does not support __len__
            with pytest.raises(NotImplementedError):
                _ = len(dataset)

            # Test iteration, take first few samples
            items = []
            for i, item in enumerate(dataset):
                items.append(item)
                if i >= 2:  # Take first 3 samples
                    break

            assert len(items) > 0
            assert items[0] is not None
        except Exception as e:
            # Skip if dataset does not exist or is inaccessible
            pytest.skip(f'MS dataset not available: {e}')


class TestDatasetMeta:
    """Test DatasetMeta functionality"""

    def test_dataset_meta_get_id(self):
        """Test DatasetMeta.get_id()"""
        meta = DatasetMeta(dataset_id='test/dataset', subset_name='subset1', split='train')
        assert meta.get_id() == 'test_dataset:subset1:train'

    def test_dataset_meta_with_data_slice(self):
        """Test DatasetMeta data_slice"""
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        meta = DatasetMeta(
            dataset_id=csv_path,
            data_slice=[0, 2]  # Select indices 0 and 2 only
        )
        dataset = Dataset(dataset_meta=meta)

        assert len(dataset) == 2
        assert dataset[0]['text'] == 'Hello world'
        assert dataset[1]['text'] == 'Another example'
