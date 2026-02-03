# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
import torch
import numpy as np
from pathlib import Path

import twinkle
from twinkle.dataset import Dataset, DatasetMeta
from twinkle.processor import InputProcessor, GRPOLossProcessor
from twinkle.data_format import Message

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


class TestNormalProcessor:
    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_normal_padding_mode(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        processor = InputProcessor(padding_free=False, padding_side='right')
        
        batch_items = [dataset[i] for i in range(3)]
        result = processor.collate_fn(batch_items)
        
        assert 'input_ids' in result
        assert 'attention_mask' in result
        assert result['input_ids'].shape[0] == 3
        assert result['attention_mask'].shape[0] == 3
        
        max_len = result['input_ids'].shape[1]
        assert all(result['input_ids'].shape[1] == max_len for _ in range(3))
        assert all(result['attention_mask'].shape[1] == max_len for _ in range(3))

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_padding_side_left(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        processor = InputProcessor(padding_free=False, padding_side='left')
        
        batch_items = [dataset[i] for i in range(3)]
        result = processor.collate_fn(batch_items)
        
        assert 'input_ids' in result
        assert result['input_ids'].shape[0] == 3


class TestPaddingFreeProcessor:
    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_padding_free_mode(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        processor = InputProcessor(padding_free=True)
        
        batch_items = [dataset[i] for i in range(3)]
        result = processor.collate_fn(batch_items)
        
        assert 'input_ids' in result
        assert result['input_ids'].shape[0] == 1
        
        total_length = sum(len(item['input_ids']) for item in batch_items)
        assert result['input_ids'].shape[1] == total_length

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_padding_free_with_different_lengths(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        processor = InputProcessor(padding_free=True)
        
        items = [dataset[i] for i in range(4)]
        lengths = [len(item['input_ids']) for item in items]
        
        result = processor.collate_fn(items)
        
        assert result['input_ids'].shape[1] == sum(lengths)
        assert result['input_ids'].shape[0] == 1


class TestMicroBatchProcessor:
    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_micro_batch_split_fixed_length(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        processor = InputProcessor(padding_free=False)
        
        batch_items = [dataset[i] for i in range(6)]
        micro_batch_size = 2
        
        results = processor.collate_fn(batch_items, micro_batch_size=micro_batch_size, variable_seq_lengths=False)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        for result in results:
            assert 'input_ids' in result
            assert result['input_ids'].shape[0] == micro_batch_size

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_micro_batch_split_variable_length(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        processor = InputProcessor(padding_free=False)
        
        batch_items = [dataset[i] for i in range(6)]
        micro_batch_size = 2
        
        results = processor.collate_fn(batch_items, micro_batch_size=micro_batch_size, variable_seq_lengths=True)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        for result in results:
            assert 'input_ids' in result
            assert result['input_ids'].shape[0] == micro_batch_size


class TestMultimodalProcessor:
    def test_multimodal_fields_concatenation(self):
        processor = InputProcessor()
        
        pixel1 = torch.randn(1, 3, 224, 224)
        pixel2 = torch.randn(1, 3, 224, 224)
        
        batch_items = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'pixel_values': pixel1,
                'image_grid_thw': torch.tensor([[1, 14, 14]])
            },
            {
                'input_ids': torch.tensor([4, 5]),
                'pixel_values': pixel2,
                'image_grid_thw': torch.tensor([[1, 14, 14]])
            }
        ]
        
        vlm_fields = {k: [] for k in processor.VLM_CONCAT_FIELDS}
        for inp in batch_items:
            inp = dict(inp)
            for field in processor.VLM_CONCAT_FIELDS:
                if field in inp:
                    vlm_fields[field].append(inp.pop(field))
        
        assert len(vlm_fields['pixel_values']) == 2
        assert len(vlm_fields['image_grid_thw']) == 2
        
        pixel_values_concat = torch.cat(vlm_fields['pixel_values'], dim=0)
        image_grid_thw_concat = torch.cat(vlm_fields['image_grid_thw'], dim=0)
        
        assert pixel_values_concat.shape[0] == 2
        assert image_grid_thw_concat.shape[0] == 2
        
        result = processor._collate_macro_batch(batch_items)
        assert 'input_ids' in result

    def test_multimodal_with_padding_free(self):
        processor = InputProcessor(padding_free=True)
        
        pixel1 = torch.randn(1, 3, 224, 224)
        pixel2 = torch.randn(1, 3, 224, 224)
        
        batch_items = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'pixel_values': pixel1,
                'image_grid_thw': torch.tensor([[1, 14, 14]])
            },
            {
                'input_ids': torch.tensor([4, 5]),
                'pixel_values': pixel2,
                'image_grid_thw': torch.tensor([[1, 14, 14]])
            }
        ]
        
        vlm_fields = {k: [] for k in processor.VLM_CONCAT_FIELDS}
        for inp in batch_items:
            inp = dict(inp)
            for field in processor.VLM_CONCAT_FIELDS:
                if field in inp:
                    vlm_fields[field].append(inp.pop(field))
        
        assert len(vlm_fields['pixel_values']) == 2
        pixel_values_concat = torch.cat(vlm_fields['pixel_values'], dim=0)
        assert pixel_values_concat.shape[0] == 2
        
        result = processor._collate_macro_batch(batch_items)
        assert 'input_ids' in result

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_multimodal_processor_with_dataset(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        def create_multimodal_messages(example):
            text = example.get('text', '')
            return {
                'messages': [
                    {'role': 'user', 'content': f'<image>\n{text}'},
                    {'role': 'assistant', 'content': 'Response'}
                ]
            }
        
        dataset.map(create_multimodal_messages)
        
        try:
            dataset.set_template('Qwen3VLTemplate', model_id='Qwen/Qwen2-VL-7B-Instruct')
            dataset.encode()
        except Exception as e:
            pytest.skip(f"Failed to setup multimodal dataset (may need network): {e}")
        
        processor = InputProcessor()
        
        batch_items = [dataset[i] for i in range(2)]
        result = processor.collate_fn(batch_items)
        
        assert 'input_ids' in result
        if 'pixel_values' in result:
            assert result['pixel_values'].shape[0] == 2


class TestGRPOProcessor:
    def test_grpo_processor_basic(self):
        processor = GRPOLossProcessor()
        
        batch_items = [
            {
                'input_ids': torch.tensor([1, 2, 3, 4, 5]),
                'labels': torch.tensor([-100, -100, 10, 11, 12])
            },
            {
                'input_ids': torch.tensor([6, 7, 8]),
                'labels': torch.tensor([-100, 20, 21])
            }
        ]
        
        result = processor._collate_macro_batch(batch_items)
        processed = processor.prepare_inputs(result)
        
        assert 'completion_mask' in processed
        assert 'logits_to_keep' in processed
        assert 'num_items_in_batch' in processed
        
        assert isinstance(processed['logits_to_keep'], int)
        assert isinstance(processed['num_items_in_batch'], int)
        assert processed['completion_mask'].shape[0] == 2

    def test_grpo_processor_all_prompt(self):
        processor = GRPOLossProcessor()
        
        batch_items = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([-100, -100, -100])
            }
        ]
        
        result = processor._collate_macro_batch(batch_items)
        processed = processor.prepare_inputs(result)
        
        assert 'completion_mask' in processed
        assert processed['num_items_in_batch'] == 0

    def test_grpo_processor_all_completion(self):
        processor = GRPOLossProcessor()
        
        batch_items = [
            {
                'input_ids': torch.tensor([1, 2, 3]),
                'labels': torch.tensor([10, 11, 12])
            }
        ]
        
        result = processor._collate_macro_batch(batch_items)
        processed = processor.prepare_inputs(result)
        
        assert 'completion_mask' in processed
        assert processed['num_items_in_batch'] == 3

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_grpo_processor_with_dataset(self):
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(convert_to_messages)
        
        try:
            dataset.set_template('Template', model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=128)
            dataset.encode(batched=True)
        except Exception as e:
            pytest.skip(f"Failed to setup dataset (may need network): {e}")
        
        processor = GRPOLossProcessor()
        
        batch_items = [dataset[i] for i in range(3)]
        result = processor.collate_fn(batch_items)
        processed = processor.prepare_inputs(result)
        
        assert 'completion_mask' in processed
        assert 'logits_to_keep' in processed
        assert 'num_items_in_batch' in processed
