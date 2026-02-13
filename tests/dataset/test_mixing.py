# Copyright (c) ModelScope Contributors. All rights reserved.
"""
Test dataset mixing:
1. add_dataset - add multiple datasets
2. mix_dataset - interleave mode
3. mix_dataset - concat mode
"""
import pytest
from pathlib import Path

from twinkle.dataset import Dataset, DatasetMeta, IterableDataset

# Get test data directory
TEST_DATA_DIR = Path(__file__).parent / 'test_data'


class TestDatasetMixing:
    """Test dataset mixing (normal dataset mode)"""

    def test_add_multiple_datasets(self):
        """Test adding multiple datasets"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))

        assert len(dataset.datasets) == 2
        assert len(dataset.dataset) == 4

    def test_mix_dataset_interleave(self):
        """Test mixing datasets with interleave"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.mix_dataset(interleave=True)

        assert len(dataset.dataset) == 6

        samples = [dataset.dataset[i] for i in range(len(dataset.dataset))]
        texts = [s['text'] for s in samples]
        assert any('Hello' in t or 'Test' in t or 'Another' in t or 'Sample' in t for t in texts)  # from test.csv
        assert any('Dataset 2' in t for t in texts)  # from test2.csv

    def test_mix_dataset_concat(self):
        """Test mixing datasets with concat"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.mix_dataset(interleave=False)

        assert len(dataset.dataset) == 7

        assert dataset.dataset[0]['text'] == 'Hello world'
        assert dataset.dataset[3]['text'] == 'Sample text'

        assert dataset.dataset[4]['text'] == 'Dataset 2 item 1'
        assert dataset.dataset[6]['text'] == 'Dataset 2 item 3'

    def test_mix_three_datasets_interleave(self):
        """Test interleaving three datasets"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')
        csv_path3 = str(TEST_DATA_DIR / 'test3.csv')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path3))
        dataset.mix_dataset(interleave=True)

        assert len(dataset.dataset) == 6

        # Verify data from three datasets
        texts = [dataset.dataset[i]['text'] for i in range(len(dataset.dataset))]
        assert any('Hello' in t or 'Test' in t or 'Another' in t or 'Sample' in t for t in texts)  # from test.csv
        assert any('Dataset 2' in t for t in texts)  # from test2.csv
        assert any('Dataset 3' in t for t in texts)  # from test3.csv

    def test_mix_three_datasets_concat(self):
        """Test concat of three datasets"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')
        csv_path3 = str(TEST_DATA_DIR / 'test3.csv')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path3))
        dataset.mix_dataset(interleave=False)

        assert len(dataset.dataset) == 9

        assert dataset.dataset[0]['text'] == 'Hello world'
        assert dataset.dataset[3]['text'] == 'Sample text'

        assert dataset.dataset[4]['text'] == 'Dataset 2 item 1'
        assert dataset.dataset[6]['text'] == 'Dataset 2 item 3'

        assert dataset.dataset[7]['text'] == 'Dataset 3 item 1'
        assert dataset.dataset[8]['text'] == 'Dataset 3 item 2'

    def test_mix_large_datasets_interleave(self):
        """Test interleaving large datasets"""
        csv_path4 = str(TEST_DATA_DIR / 'test4.csv')
        csv_path5 = str(TEST_DATA_DIR / 'test5.csv')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path4))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path5))
        dataset.mix_dataset(interleave=True)

        assert len(dataset.dataset) == 224

        texts = []
        for i in range(len(dataset.dataset)):
            item = dataset.dataset[i]
            text = item.get('text') or item.get('question') or ''
            if text:
                texts.append(str(text))

        assert any('Complex example' in t or 'Extended metadata' in t for t in texts)
        assert any('capital of France' in t or 'quantum mechanics' in t for t in texts)

    def test_mix_large_datasets_concat(self):
        """Test concat of large datasets"""
        csv_path4 = str(TEST_DATA_DIR / 'test4.csv')  #
        csv_path5 = str(TEST_DATA_DIR / 'test5.csv')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path4))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path5))
        dataset.mix_dataset(interleave=False)

        assert len(dataset.dataset) == 281

        assert 'Complex example' in str(dataset.dataset[0].get('text', ''))
        assert 'Multiplayer sync tick' in str(dataset.dataset[111].get('text', ''))

        assert 'capital of France' in str(dataset.dataset[112].get('question', ''))

        assert 'democracy' in str(dataset.dataset[121].get('question', ''))

        last_item = dataset.dataset[280]
        last_text = str(last_item.get('text') or last_item.get('id') or last_item.get('question') or '')
        assert 'Multiplayer sync tick' in last_text or 'tick_rate_64' in last_text

    def test_mix_different_formats_csv_json(self):
        """Test mixing different formats (CSV + JSON)"""
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        json_path = str(TEST_DATA_DIR / 'test6.json')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.add_dataset(DatasetMeta(dataset_id=json_path))
        dataset.mix_dataset(interleave=True)

        assert len(dataset.dataset) == 8

        has_csv_data = False
        has_json_data = False
        for item in dataset.dataset:
            text = item.get('text')
            if text and ('Hello' in str(text) or 'Test' in str(text)):
                has_csv_data = True
            title = item.get('title')
            if title and 'Article' in str(title):
                has_json_data = True

        assert has_csv_data
        assert has_json_data

    def test_mix_different_formats_csv_jsonl(self):
        """Test mixing different formats (CSV + JSONL)"""
        csv_path = str(TEST_DATA_DIR / 'test2.csv')
        jsonl_path = str(TEST_DATA_DIR / 'test7.jsonl')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.add_dataset(DatasetMeta(dataset_id=jsonl_path))
        dataset.mix_dataset(interleave=False)

        assert len(dataset.dataset) == 15

        assert 'Dataset 2' in dataset.dataset[0].get('text', '')

        assert 'user_id' in dataset.dataset[3]
        assert 'action' in dataset.dataset[3]

    def test_mix_multiple_large_datasets(self):
        """Test mixing multiple large datasets (CSV only for large_string alignment)"""
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')
        csv_path3 = str(TEST_DATA_DIR / 'test3.csv')
        csv_path4 = str(TEST_DATA_DIR / 'test4.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path3))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path4))
        dataset.mix_dataset(interleave=False)  # concat keeps all samples
        assert len(dataset.dataset) == 121  # 4+3+2+112
        all_texts = [str(item.get('text', '')) for item in dataset.dataset]
        assert any('Hello' in t or 'Test' in t for t in all_texts)
        assert any('Dataset 2' in t for t in all_texts)
        assert any('Dataset 3' in t for t in all_texts)
        assert any('Complex example' in t or 'Multiplayer' in t for t in all_texts)

    def test_mix_very_large_datasets_concat(self):
        """Test concat of very large datasets (alignable schema)"""
        csv_path4 = str(TEST_DATA_DIR / 'test4.csv')
        csv_path5 = str(TEST_DATA_DIR / 'test5.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path4))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path5))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
        dataset.mix_dataset(interleave=False)
        assert len(dataset.dataset) == 284  # 112 + 169 + 3
        assert 'Complex example' in str(dataset.dataset[0].get('text', ''))
        assert 'capital of France' in str(dataset.dataset[112].get('question', ''))
        assert 'Dataset 2' in str(dataset.dataset[281].get('text', ''))

    def test_mix_complex_fields_interleave(self):
        """Test interleaving datasets with complex fields"""
        csv_path4 = str(TEST_DATA_DIR / 'test4.csv')
        csv_path8 = str(TEST_DATA_DIR / 'test8.csv')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path4))
        dataset.add_dataset(DatasetMeta(dataset_id=csv_path8))
        dataset.mix_dataset(interleave=True)

        assert len(dataset.dataset) == 24

        # Verify complex fields exist
        has_metadata = any('metadata' in item for item in dataset.dataset)
        has_product_fields = any('product_id' in item and 'price' in item for item in dataset.dataset)
        assert has_metadata
        assert has_product_fields

    def test_mix_all_formats_concat(self):
        """Test concat of all formats"""
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        json_path = str(TEST_DATA_DIR / 'test6.json')
        jsonl_path = str(TEST_DATA_DIR / 'test7.jsonl')

        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.add_dataset(DatasetMeta(dataset_id=json_path))
        dataset.add_dataset(DatasetMeta(dataset_id=jsonl_path))
        dataset.mix_dataset(interleave=False)

        assert len(dataset.dataset) == 121  # 4 + 105 + 12

        assert 'text' in dataset.dataset[0]
        assert 'title' in dataset.dataset[4]
        assert 'user_id' in dataset.dataset[109]


class TestIterableDatasetMixing:
    """Test dataset mixing (iterable mode)"""

    def test_add_multiple_datasets_iterable(self):
        """Test adding multiple datasets (iterable mode)"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')

        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
            dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))

            assert len(dataset.datasets) == 2

            with pytest.raises((NotImplementedError, TypeError)):
                _ = len(dataset.dataset)
        except NotImplementedError as e:
            pytest.xfail(f'Known limitation: streaming local file with num_proc is not supported: {e}')

    def test_mix_dataset_interleave_iterable(self):
        """Test interleaving datasets (iterable mode)"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')

        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
            dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
            dataset.mix_dataset(interleave=True)

            with pytest.raises((NotImplementedError, TypeError)):
                _ = len(dataset.dataset)
            items = []
            for i, item in enumerate(dataset):
                items.append(item)
                if i >= 5:
                    break
            assert len(items) == 6  # interleave first_exhausted: stop when shorter dataset (3) exhausted
            texts = [item['text'] for item in items]
            assert any('Hello' in t or 'Test' in t or 'Another' in t for t in texts)
            assert any('Dataset 2' in t for t in texts)
        except NotImplementedError as e:
            pytest.xfail(f'Known limitation: streaming local file with num_proc is not supported: {e}')

    def test_mix_dataset_concat_iterable(self):
        """Test concat of datasets (iterable mode)"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')

        try:
            dataset = IterableDataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
            dataset.add_dataset(DatasetMeta(dataset_id=csv_path2))
            dataset.mix_dataset(interleave=False)

            with pytest.raises((NotImplementedError, TypeError)):
                _ = len(dataset.dataset)
            items = []
            for i, item in enumerate(dataset):
                items.append(item)
                if i >= 6:
                    break
            assert len(items) == 7
            assert items[0]['text'] == 'Hello world'
            assert items[3]['text'] == 'Sample text'
            assert items[4]['text'] == 'Dataset 2 item 1'
            assert items[6]['text'] == 'Dataset 2 item 3'
        except NotImplementedError as e:
            pytest.xfail(f'Known limitation: streaming local file with num_proc is not supported: {e}')


class TestDatasetMixingEdgeCases:
    """Test dataset mixing edge cases"""

    def test_mix_single_dataset(self):
        """Test mix_dataset with single dataset"""
        csv_path = str(TEST_DATA_DIR / 'test.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))

        # With single dataset, mix_dataset should not change dataset
        original_len = len(dataset.dataset)
        dataset.mix_dataset(interleave=True)

        # dataset should remain unchanged
        assert len(dataset.dataset) == original_len
        assert dataset.dataset[0]['text'] == 'Hello world'

    def test_mix_datasets_with_different_streaming_modes_error(self):
        """Test mixing streaming and non-streaming datasets should raise"""
        csv_path1 = str(TEST_DATA_DIR / 'test.csv')
        csv_path2 = str(TEST_DATA_DIR / 'test2.csv')
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path1))
        try:
            dataset.add_dataset(DatasetMeta(dataset_id=csv_path2), streaming=True)
            with pytest.raises((AssertionError, ValueError),
                               match=r'(All datasets must be all streaming|Unable to interleave)'):
                dataset.mix_dataset(interleave=True)
        except NotImplementedError:
            pytest.xfail('Known limitation: streaming local file with num_proc is not supported')
