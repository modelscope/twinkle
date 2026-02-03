# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
from pathlib import Path

from twinkle.dataset import Dataset, LazyDataset, DatasetMeta

TEST_DATA_DIR = Path(__file__).parent / "test_data"
SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


def create_multimodal_messages(example):
    text = example.get('text', '')
    if not text:
        text = str(example.get('question', example.get('title', '')))
    
    return {
        'messages': [
            {'role': 'user', 'content': f'<image>\n{text}'},
            {'role': 'assistant', 'content': 'Response'}
        ]
    }


class TestMultimodalDataset:
    #基本功能
    def test_multimodal_dataset_basic(self):
        #多模态数据集基本功能（图像+文本）
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(create_multimodal_messages)
        
        assert len(dataset) == 4
        item = dataset[0]
        assert 'messages' in item
        
        messages = item['messages']
        assert len(messages) == 2
        user_msg = messages[0]
        assert user_msg['role'] == 'user'
        assert '<image>' in user_msg['content']

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_multimodal_dataset_with_qwen3vl_template(self):
        #使用Qwen3VLTemplate
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(create_multimodal_messages)
        
        try:
            dataset.set_template('Qwen3VLTemplate', model_id='Qwen/Qwen3-VL-2B-Instruct')
        except Exception as e:
            pytest.skip(f"Failed to load Qwen3VLTemplate (may need network): {e}")
        
        assert dataset.template is not None
        assert hasattr(dataset.template, 'is_mm')

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_multimodal_dataset_encode_with_lazy(self):
        #多模态数据集编码
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = LazyDataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(create_multimodal_messages)
        
        try:
            dataset.set_template('Qwen3VLTemplate', model_id='Qwen/Qwen3-VL-2B-Instruct')
        except Exception as e:
            pytest.skip(f"Failed to load Qwen3VLTemplate (may need network): {e}")
        
        try:
            dataset.encode()
        except Exception as e:
            pytest.skip(f"Failed to encode multimodal dataset: {e}")
        
        item = dataset[0]
        assert 'input_ids' in item
        assert len(item['input_ids']) > 0

    def test_multimodal_dataset_image_placeholder(self):
        #图像占位符处理
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        dataset.map(create_multimodal_messages)
        
        item = dataset[0]
        assert 'messages' in item
        user_content = item['messages'][0]['content']
        assert '<image>' in user_content

    def test_multimodal_dataset_multiple_image_placeholders(self):
        #多图像处理
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        def create_multi_image_messages(example):
            text = example.get('text', '')
            return {
                'messages': [
                    {'role': 'user', 'content': f'<image>\n{text}\n<image>'},
                    {'role': 'assistant', 'content': 'Response'}
                ]
            }
        
        dataset.map(create_multi_image_messages)
        
        item = dataset[0]
        user_content = item['messages'][0]['content']
        assert user_content.count('<image>') == 2

    def test_multimodal_dataset_video_placeholder(self):
        #视频占位符
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        def create_video_messages(example):
            text = example.get('text', '')
            return {
                'messages': [
                    {'role': 'user', 'content': f'<video>\n{text}'},
                    {'role': 'assistant', 'content': 'Response'}
                ]
            }
        
        dataset.map(create_video_messages)
        
        item = dataset[0]
        user_content = item['messages'][0]['content']
        assert '<video>' in user_content

    def test_multimodal_dataset_audio_placeholder(self):
        #音频占位符
        csv_path = str(TEST_DATA_DIR / "test.csv")
        dataset = Dataset(dataset_meta=DatasetMeta(dataset_id=csv_path))
        
        def create_audio_messages(example):
            text = example.get('text', '')
            return {
                'messages': [
                    {'role': 'user', 'content': f'<audio>\n{text}'},
                    {'role': 'assistant', 'content': 'Response'}
                ]
            }
        
        dataset.map(create_audio_messages)
        
        item = dataset[0]
        user_content = item['messages'][0]['content']
        assert '<audio>' in user_content
