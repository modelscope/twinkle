# Copyright (c) ModelScope Contributors. All rights reserved.
import numpy as np
import os
import pytest

import twinkle
from twinkle.data_format import Message, Trajectory
from twinkle.template import Template
from twinkle.template.contract import MediaBundle, coerce_tool_call_arguments, split_reasoning

twinkle.initialize(mode='local')

SKIP_MODEL_DOWNLOAD = os.getenv('SKIP_MODEL_DOWNLOAD', 'false').lower() == 'true'


@pytest.mark.parametrize(
    'text,started,expected',
    [
        ('reasoning</think>answer', True, ('reasoning', 'answer')),
        ('r</think>Use the </think> tag', True, ('r', 'Use the </think> tag')),
        ('partial reasoning', True, ('partial reasoning', '')),
        ('plain answer', False, (None, 'plain answer')),
    ],
)
def test_split_reasoning_contract(text, started, expected):
    assert split_reasoning(text, reasoning_started=started) == expected


def test_decode_message_splits_special_and_truncated_reasoning():

    class Processor:
        all_special_tokens = ['<channel|>']
        text = 'need weather<channel|>It is sunny.'

        def decode(self, token_ids, *, skip_special_tokens=True, **kwargs):
            return self.text.replace('<channel|>', '') if skip_special_tokens else self.text

    template = object.__new__(Template)
    template.processor = Processor()
    template.enable_thinking = True
    template.reasoning_open_marker = '<|channel>thought'
    template.reasoning_close_marker = '<channel|>'
    response = template.decode_message([], parse_tool_calls=False, reasoning_started=True)
    assert response.reasoning_content == 'need weather'
    assert response.content == 'It is sunny.'

    template.processor.text = 'unfinished reasoning'
    response = template.decode_message([], parse_tool_calls=False, reasoning_started=True)
    assert response.reasoning_content == 'unfinished reasoning'
    assert response.content == ''


def test_reasoning_state_comes_from_tokens_not_enable_flag():

    class Tokenizer:

        @staticmethod
        def encode(text, **kwargs):
            return [ord(char) for char in text]

    template = object.__new__(Template)
    template.processor = Tokenizer()
    template.reasoning_open_marker = '<think>'
    template.reasoning_close_marker = '</think>'
    assert template._reasoning_started(template.tokenizer.encode('prompt<think>partial'))
    assert not template._reasoning_started(template.tokenizer.encode('prompt<think>reason</think>answer'))
    assert not template._reasoning_started(template.tokenizer.encode('plain prompt'))


def test_generic_template_infers_media_types_from_config_and_processor():
    template = object.__new__(Template)
    template.config = type('Config', (), {'image_token_id': 1, 'video_token_id': 2})()
    template.processor = type('Processor', (), {'image_processor': object()})()
    assert template._infer_media_types() == frozenset({'image', 'video'})


def test_media_bundle_captures_top_level_and_inline_media():
    top_level = {'images': ['a.png'], 'messages': []}
    inline = {
        'messages': [{
            'role':
            'user',
            'content': [
                {
                    'type': 'text',
                    'text': 'look'
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': 'b.png'
                    }
                },
            ],
        }]
    }
    assert MediaBundle.from_sources(top_level).images == ('a.png', )
    assert MediaBundle.from_sources(inline).images == ('b.png', )


def test_render_delta_normalizes_tool_calls_in_history():

    class Tokenizer:

        def apply_chat_template(self, messages, **kwargs):
            text = ''
            for message in messages:
                calls = message.get('tool_calls') or ()
                for call in calls:
                    assert isinstance(call['function']['arguments'], dict)
                text += f"{message['role']}:{message.get('content') or ''}\n"
            if kwargs.get('add_generation_prompt'):
                text += 'assistant:'
            return text

        @staticmethod
        def encode(text, **kwargs):
            return [ord(char) for char in text]

    template = object.__new__(Template)
    template.processor = Tokenizer()
    template.enable_thinking = False
    history = [{
        'role': 'assistant',
        'content': '',
        'tool_calls': [{
            'type': 'function',
            'function': {
                'name': 'search',
                'arguments': '{"limit": 3}'
            },
        }],
    }]
    delta = template.render_delta(
        history, [{
            'role': 'tool',
            'content': 'done'
        }], gen_prompt_before=False, gen_prompt_after=True)
    assert delta


def test_render_delta_engine_ids_use_the_engine_hook():

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, **kwargs):
            return ''.join(message['content'] for message in messages)

        @staticmethod
        def encode(text, **kwargs):
            return [ord(char) for char in text]

    template = object.__new__(Template)
    template.processor = Tokenizer()
    template.enable_thinking = False
    template._engine_input_ids = lambda ids: [ids[0]] if ids else ids
    history = [{'role': 'user', 'content': 'a'}]
    appended = [{'role': 'assistant', 'content': 'bc'}]
    feature = template.render_delta(history, appended, gen_prompt_before=True, gen_prompt_after=False)
    engine = template.render_delta(history, appended, gen_prompt_before=True, gen_prompt_after=False, engine=True)
    assert len(feature) == 2
    assert engine == [feature[0]]


def test_tool_arguments_follow_json_schema():
    tools = [{
        'type': 'function',
        'function': {
            'name': 'search',
            'parameters': {
                'type': 'object',
                'properties': {
                    'limit': {
                        'type': 'integer'
                    },
                    'score': {
                        'type': 'number'
                    },
                    'exact': {
                        'type': 'boolean'
                    },
                    'ids': {
                        'type': 'array',
                        'items': {
                            'type': 'integer'
                        }
                    },
                },
            },
        },
    }]
    calls = [{
        'type': 'function',
        'function': {
            'name': 'search',
            'arguments': {
                'limit': '3',
                'score': '0.5',
                'exact': 'false',
                'ids': '[1, "2"]'
            },
        },
    }]
    arguments = coerce_tool_call_arguments(calls, tools)[0]['function']['arguments']
    assert arguments == {'limit': 3, 'score': 0.5, 'exact': False, 'ids': [1, 2]}


class TestTextTemplate:

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_qwen25_text_template_basic(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        messages = [
            Message(role='user', content='How are you?'),
            Message(role='assistant', content='I am fine, thank you!'),
        ]
        trajectory = Trajectory(messages=messages)

        encoded = template.encode(trajectory)

        assert 'input_ids' in encoded
        assert 'labels' in encoded
        assert len(encoded['input_ids']) > 0
        assert len(encoded['labels']) == len(encoded['input_ids'])

        input_ids = encoded['input_ids']
        labels = encoded['labels']

        assert isinstance(input_ids, np.ndarray)
        assert isinstance(labels, np.ndarray)

        assert (labels == -100).sum() > 0
        assert (labels != -100).sum() > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_qwen25_text_template_multiple_messages(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        messages = [
            Message(role='user', content='What is 1+1?'),
            Message(role='assistant', content='2'),
            Message(role='user', content='What is 2+2?'),
            Message(role='assistant', content='4'),
        ]
        trajectory = Trajectory(messages=messages)

        encoded = template.encode(trajectory)

        assert 'input_ids' in encoded
        assert 'labels' in encoded
        assert len(encoded['input_ids']) > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_qwen25_text_template_labels_correctness(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        messages = [Message(role='user', content='Hello'), Message(role='assistant', content='Hi there')]
        trajectory = Trajectory(messages=messages)

        encoded = template.encode(trajectory)

        input_ids = encoded['input_ids']
        labels = encoded['labels']

        assert len(input_ids) == len(labels)

        prompt_mask = labels == -100
        completion_mask = labels != -100

        assert prompt_mask.sum() > 0
        assert completion_mask.sum() > 0

        completion_tokens = input_ids[completion_mask]
        assert len(completion_tokens) > 0


class TestMultimodalTemplate:

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_qwen2vl_multimodal_template_basic(self):
        try:
            template = Template(
                model_id='Qwen/Qwen2-VL-7B-Instruct',
                max_length=8192,
                truncation_strategy='right',
                media_types={'image', 'video'},
            )
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        assert template.is_multimodal
        assert template.media_types == frozenset({'image', 'video'})

        image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
        messages = [
            Message(role='user', content='<image>\nWhat is in this image?', images=[image_url]),
            Message(role='assistant', content='This is a test image.'),
        ]
        trajectory = Trajectory(messages=messages)

        encoded = template.encode(trajectory)

        assert 'input_ids' in encoded
        assert 'labels' in encoded
        assert len(encoded['input_ids']) > 0
        assert len(encoded['labels']) == len(encoded['input_ids'])

        input_ids = encoded['input_ids']
        labels = encoded['labels']

        assert isinstance(input_ids, np.ndarray)
        assert isinstance(labels, np.ndarray)

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_qwen2vl_multimodal_template_with_placeholder(self):
        try:
            template = Template(
                model_id='Qwen/Qwen2-VL-7B-Instruct',
                max_length=8192,
                truncation_strategy='right',
                media_types={'image', 'video'},
            )
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
        messages = [
            Message(role='user', content='<image>\nDescribe this image.'),
            Message(role='assistant', content='The image shows a beautiful landscape.'),
        ]
        trajectory = Trajectory(messages=messages, images=[image_url])

        encoded = template.encode(trajectory)

        assert 'input_ids' in encoded
        assert 'labels' in encoded

        if 'pixel_values' in encoded:
            assert encoded['pixel_values'].shape[0] > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_qwen2vl_multimodal_template_labels_correctness(self):
        try:
            template = Template(
                model_id='Qwen/Qwen2-VL-7B-Instruct',
                max_length=8192,
                truncation_strategy='right',
                media_types={'image', 'video'},
            )
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
        messages = [
            Message(role='user', content='<image>\nWhat do you see?', images=[image_url]),
            Message(role='assistant', content='I see an image.'),
        ]
        trajectory = Trajectory(messages=messages)

        encoded = template.encode(trajectory)

        input_ids = encoded['input_ids']
        labels = encoded['labels']

        assert len(input_ids) == len(labels)

        prompt_mask = labels == -100
        completion_mask = labels != -100

        assert prompt_mask.sum() > 0
        assert completion_mask.sum() > 0

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_qwen2vl_multimodal_template_multiple_images(self):
        try:
            template = Template(
                model_id='Qwen/Qwen2-VL-7B-Instruct',
                max_length=16384,
                truncation_strategy='right',
                media_types={'image', 'video'},
            )
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        image_url = 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'
        messages = [
            Message(role='user', content='<image>\n<image>\nCompare these images.', images=[image_url, image_url]),
            Message(role='assistant', content='Both images are similar.'),
        ]
        trajectory = Trajectory(messages=messages)

        encoded = template.encode(trajectory)

        assert 'input_ids' in encoded
        assert 'labels' in encoded


class TestTemplateEdgeCases:

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_text_template_empty_assistant(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=512)
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        messages = [Message(role='user', content='Hello')]
        trajectory = Trajectory(messages=messages)

        encoded = template.encode(trajectory)

        assert 'input_ids' in encoded
        assert 'labels' in encoded

    @pytest.mark.skipif(SKIP_MODEL_DOWNLOAD, reason='Skipping tests that require model download')
    def test_text_template_max_length_truncation(self):
        try:
            template = Template(model_id='ms://Qwen/Qwen2.5-0.5B-Instruct', max_length=50, truncation_strategy='right')
        except Exception as e:
            pytest.skip(f'Failed to load template (may need network): {e}')

        long_text = 'Hello ' * 100
        messages = [Message(role='user', content=long_text), Message(role='assistant', content='Response')]
        trajectory = Trajectory(messages=messages)

        encoded = template.encode(trajectory)

        assert len(encoded['input_ids']) <= 50
