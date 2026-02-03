# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
from pathlib import Path
from torch.utils.data import SequentialSampler, RandomSampler

import twinkle
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.dataloader import DataLoader

twinkle.initialize(mode='local')

TEST_DATA_DIR = Path(__file__).parent.parent / "dataset" / "test_data"


class TestSequentialSampler:
    def test_sequential_sampler_basic(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=5, sampler=sampler)
        
        batches = list(dataloader)
        dataset_size = len(dataset)
        expected_batches = (dataset_size + 4) // 5
        
        assert len(batches) == expected_batches
        
        first_batch = batches[0]
        assert len(first_batch) == 5
        
        assert first_batch[0]['text'] == "Hello world"
        assert first_batch[1]['text'] == "Test data"
        assert first_batch[2]['text'] == "Another example"
        assert first_batch[3]['text'] == "Sample text"
        assert first_batch[4]['text'] == "Machine learning is fascinating"

    def test_sequential_sampler_batch_size_1(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=1, sampler=sampler)
        
        batches = list(dataloader)
        dataset_size = len(dataset)
        
        assert len(batches) == dataset_size
        
        assert batches[0][0]['text'] == "Hello world"
        assert batches[1][0]['text'] == "Test data"
        assert batches[2][0]['text'] == "Another example"
        assert batches[3][0]['text'] == "Sample text"

    def test_sequential_sampler_multiple_epochs(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=3, sampler=sampler)
        
        epoch1 = list(dataloader)
        epoch2 = list(dataloader)
        
        assert len(epoch1) == len(epoch2)
        assert epoch1[0][0]['text'] == epoch2[0][0]['text'] == "Hello world"
        assert epoch1[0][1]['text'] == epoch2[0][1]['text'] == "Test data"
        assert epoch1[0][2]['text'] == epoch2[0][2]['text'] == "Another example"


class TestRandomSampler:
    def test_random_sampler_basic(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset=dataset, batch_size=7, sampler=sampler)
        
        batches = list(dataloader)
        dataset_size = len(dataset)
        expected_batches = (dataset_size + 6) // 7
        
        assert len(batches) == expected_batches
        
        all_texts = [item['text'] for batch in batches for item in batch]
        assert len(all_texts) == dataset_size
        assert len(set(all_texts)) == dataset_size
        
        all_items = [item for batch in batches for item in batch]
        expected_texts = {item['text'] for item in dataset}
        assert set(all_texts) == expected_texts

    def test_random_sampler_different_order(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        sampler1 = RandomSampler(dataset)
        sampler2 = RandomSampler(dataset)
        
        dataloader1 = DataLoader(dataset=dataset, batch_size=5, sampler=sampler1)
        dataloader2 = DataLoader(dataset=dataset, batch_size=5, sampler=sampler2)
        
        batches1 = list(dataloader1)
        batches2 = list(dataloader2)
        
        texts1 = [item['text'] for batch in batches1 for item in batch]
        texts2 = [item['text'] for batch in batches2 for item in batch]
        
        assert set(texts1) == set(texts2)
        assert len(texts1) == len(texts2) == len(dataset)
        
        different_order = texts1 != texts2
        assert different_order or len(texts1) == 1

    def test_random_sampler_with_replacement(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset_size = len(dataset)
        
        num_samples = dataset_size
        sampler = RandomSampler(dataset, replacement=True, num_samples=num_samples)
        dataloader = DataLoader(dataset=dataset, batch_size=5, sampler=sampler, max_retries=50)
        
        batches = list(dataloader)
        expected_batches = (num_samples + 4) // 5
        
        assert len(batches) == expected_batches
        
        all_texts = [item['text'] for batch in batches for item in batch]
        assert len(all_texts) == num_samples
        
        all_indices = [item for batch in batches for item in batch]
        assert len(all_indices) == num_samples


class TestSamplerComparison:
    def test_sequential_vs_random_order(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        seq_sampler = SequentialSampler(dataset)
        rand_sampler = RandomSampler(dataset)
        
        seq_dataloader = DataLoader(dataset=dataset, batch_size=10, sampler=seq_sampler)
        rand_dataloader = DataLoader(dataset=dataset, batch_size=10, sampler=rand_sampler)
        
        seq_batches = list(seq_dataloader)
        rand_batches = list(rand_dataloader)
        
        seq_texts = [item['text'] for batch in seq_batches for item in batch]
        rand_texts = [item['text'] for batch in rand_batches for item in batch]
        
        assert set(seq_texts) == set(rand_texts)
        assert len(seq_texts) == len(rand_texts) == len(dataset)
        
        assert seq_texts[0] == "Hello world"
        assert seq_texts[1] == "Test data"
        assert seq_texts[2] == "Another example"
        assert seq_texts[3] == "Sample text"
        
        different = seq_texts != rand_texts
        assert different or len(seq_texts) == 1
