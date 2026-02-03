# Copyright (c) ModelScope Contributors. All rights reserved.
import os
import pytest
from pathlib import Path
from PIL import Image
import numpy as np

import twinkle
from twinkle.data_format import Message, Trajectory
from twinkle.template import Template

twinkle.initialize(mode='local')

SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


class TestTextTemplate:
    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_qwen25_text_template_basic(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        messages = [
            Message(role='user', content='How are you?'),
            Message(role='assistant', content='I am fine, thank you!')
        ]
        trajectory = Trajectory(messages=messages)
        
        encoded = template.batch_encode([trajectory])
        
        assert len(encoded) == 1
        assert 'input_ids' in encoded[0]
        assert 'labels' in encoded[0]
        assert len(encoded[0]['input_ids']) > 0
        assert len(encoded[0]['labels']) == len(encoded[0]['input_ids'])
        
        input_ids = encoded[0]['input_ids']
        labels = encoded[0]['labels']
        
        assert isinstance(input_ids, np.ndarray)
        assert isinstance(labels, np.ndarray)
        
        assert (labels == -100).sum() > 0
        assert (labels != -100).sum() > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_qwen25_text_template_multiple_messages(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        messages = [
            Message(role='user', content='What is 1+1?'),
            Message(role='assistant', content='2'),
            Message(role='user', content='What is 2+2?'),
            Message(role='assistant', content='4')
        ]
        trajectory = Trajectory(messages=messages)
        
        encoded = template.batch_encode([trajectory])
        
        assert len(encoded) == 1
        assert 'input_ids' in encoded[0]
        assert 'labels' in encoded[0]
        assert len(encoded[0]['input_ids']) > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_qwen25_text_template_labels_correctness(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        messages = [
            Message(role='user', content='Hello'),
            Message(role='assistant', content='Hi there')
        ]
        trajectory = Trajectory(messages=messages)
        
        encoded = template.batch_encode([trajectory])
        
        input_ids = encoded[0]['input_ids']
        labels = encoded[0]['labels']
        
        assert len(input_ids) == len(labels)
        
        prompt_mask = (labels == -100)
        completion_mask = (labels != -100)
        
        assert prompt_mask.sum() > 0
        assert completion_mask.sum() > 0
        
        completion_tokens = input_ids[completion_mask]
        assert len(completion_tokens) > 0


class TestMultimodalTemplate:
    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_qwen2vl_multimodal_template_basic(self):
        try:
            template = Template(model_id='Qwen/Qwen2-VL-7B-Instruct', max_length=8192, truncation_strategy='right')
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        assert template.is_mm == True
        
        image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
        messages = [
            Message(role='user', content='<image>\nWhat is in this image?', images=[image_url]),
            Message(role='assistant', content='This is a test image.')
        ]
        trajectory = Trajectory(messages=messages)
        
        encoded = template.batch_encode([trajectory])
        
        assert len(encoded) == 1
        assert 'input_ids' in encoded[0]
        assert 'labels' in encoded[0]
        assert len(encoded[0]['input_ids']) > 0
        assert len(encoded[0]['labels']) == len(encoded[0]['input_ids'])
        
        input_ids = encoded[0]['input_ids']
        labels = encoded[0]['labels']
        
        assert isinstance(input_ids, np.ndarray)
        assert isinstance(labels, np.ndarray)

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_qwen2vl_multimodal_template_with_placeholder(self):
        try:
            template = Template(model_id='Qwen/Qwen2-VL-7B-Instruct', max_length=8192, truncation_strategy='right')
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
        messages = [
            Message(role='user', content='<image>\nDescribe this image.'),
            Message(role='assistant', content='The image shows a beautiful landscape.')
        ]
        trajectory = Trajectory(messages=messages, images=[image_url])
        
        encoded = template.batch_encode([trajectory])
        
        assert len(encoded) == 1
        assert 'input_ids' in encoded[0]
        assert 'labels' in encoded[0]
        
        if 'pixel_values' in encoded[0]:
            assert encoded[0]['pixel_values'].shape[0] > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_qwen2vl_multimodal_template_labels_correctness(self):
        try:
            template = Template(model_id='Qwen/Qwen2-VL-7B-Instruct', max_length=8192, truncation_strategy='right')
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
        messages = [
            Message(role='user', content='<image>\nWhat do you see?', images=[image_url]),
            Message(role='assistant', content='I see an image.')
        ]
        trajectory = Trajectory(messages=messages)
        
        encoded = template.batch_encode([trajectory])
        
        input_ids = encoded[0]['input_ids']
        labels = encoded[0]['labels']
        
        assert len(input_ids) == len(labels)
        
        prompt_mask = (labels == -100)
        completion_mask = (labels != -100)
        
        assert prompt_mask.sum() > 0
        assert completion_mask.sum() > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_qwen2vl_multimodal_template_multiple_images(self):
        try:
            template = Template(model_id='Qwen/Qwen2-VL-7B-Instruct', max_length=16384, truncation_strategy='right')
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
        messages = [
            Message(role='user', content='<image>\n<image>\nCompare these images.', images=[image_url, image_url]),
            Message(role='assistant', content='Both images are similar.')
        ]
        trajectory = Trajectory(messages=messages)
        
        encoded = template.batch_encode([trajectory])
        
        assert len(encoded) == 1
        assert 'input_ids' in encoded[0]
        assert 'labels' in encoded[0]


class TestTemplateEdgeCases:
    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_text_template_empty_assistant(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        messages = [
            Message(role='user', content='Hello')
        ]
        trajectory = Trajectory(messages=messages)
        
        encoded = template.batch_encode([trajectory])
        
        assert len(encoded) == 1
        assert 'input_ids' in encoded[0]
        assert 'labels' in encoded[0]

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason="Skipping tests that require model download")
    def test_text_template_max_length_truncation(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=50, truncation_strategy='right')
        except Exception as e:
            pytest.skip(f"Failed to load template (may need network): {e}")
        
        long_text = 'Hello ' * 100
        messages = [
            Message(role='user', content=long_text),
            Message(role='assistant', content='Response')
        ]
        trajectory = Trajectory(messages=messages)
        
        encoded = template.batch_encode([trajectory])
        
        assert len(encoded) == 1
        assert len(encoded[0]['input_ids']) <= 50
