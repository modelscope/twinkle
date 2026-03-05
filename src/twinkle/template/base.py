# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
import numpy as np
import os
from collections.abc import Mapping
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

from twinkle.data_format import InputFeature, Message, Trajectory
from twinkle.hub import HubOperation
from twinkle.utils import load_image, to_device
from .utils import tokenize_with_assistant_labels, transfer_to_standard_message

if TYPE_CHECKING:
    import torch
    from PIL import Image

# Type aliases for multimodal data
ImageInput = Union[str, 'Image.Image', 'torch.Tensor']
VideoInput = Union[str, List['Image.Image'], 'torch.Tensor']
AudioInput = Union[str, np.ndarray, 'torch.Tensor']


class Template:

    # Placeholder tokens in user text
    image_placeholder: str = '<image>'
    video_placeholder: str = '<video>'
    audio_placeholder: str = '<audio>'

    def __init__(self,
                 model_id: str,
                 use_chat_template: bool = True,
                 max_length: Optional[int] = 8192,
                 truncation_strategy: Literal['raise', 'left', 'right', 'split'] = 'raise',
                 default_system: Optional[str] = None,
                 **kwargs):
        model_id = HubOperation.download_model(model_id, ignore_model=True)
        if os.path.exists(os.path.join(model_id, 'preprocessor_config.json')):
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        else:
            from transformers import AutoTokenizer
            self.processor = AutoTokenizer.from_pretrained(model_id, **kwargs)
        from transformers import AutoConfig
        self.config = AutoConfig.from_pretrained(model_id, **kwargs)

        self.use_chat_template = use_chat_template
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.default_system = default_system
        self._test_support_assistant_tokens_mask()
        self.pre_pipeline: List[Callable[[Trajectory], List[Trajectory]]] = [
            self._add_default_system,  # Add a default system field
            self._build_mm_messages,  # turn to standard mm messages
        ]
        self.post_pipeline: List[Callable[[InputFeature], List[InputFeature]]] = [
            self._check_max_length,  # Check and split input_features
            self._add_attention_fields,  # Add useful fields
            self._roll_labels,  # roll labels
        ]

    @property
    def tokenizer(self):
        tokenizer = self.processor
        if hasattr(tokenizer, 'tokenizer'):
            tokenizer = tokenizer.tokenizer
        return tokenizer

    @property
    def is_mm(self):
        from transformers import ProcessorMixin
        return isinstance(self.processor, ProcessorMixin)

    def _test_support_assistant_tokens_mask(self):
        # For VLM processors (is_mm=True), content must be list of dicts
        # For text-only processors, content can be a simple string
        if self.is_mm:
            dummy_inputs = [
                {
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': 'How are you?'
                    }]
                },
                {
                    'role': 'assistant',
                    'content': [{
                        'type': 'text',
                        'text': 'Fine.'
                    }]
                },
            ]
        else:
            dummy_inputs = [
                Message(role='user', content='How are you?'),
                Message(role='assistant', content='Fine.'),
            ]
        try:
            outputs = self.processor.apply_chat_template(
                dummy_inputs, return_assistant_tokens_mask=True, return_dict=True, tokenize=True)
            # Check if outputs is a dict (not all processors return dict even with return_dict=True)
            if isinstance(outputs, dict) and 'assistant_masks' in outputs:
                assistant_masks = outputs['assistant_masks']
                self._template_support_assistant_tokens_mask = (0 < np.array(assistant_masks).sum() <
                                                                len(assistant_masks))
            else:
                # Processor doesn't support return_dict properly
                self._template_support_assistant_tokens_mask = False
        except Exception:  # noqa
            # If any error occurs during testing, fall back to not supporting
            self._template_support_assistant_tokens_mask = False

    def preprocess_image(self, image: ImageInput) -> 'Image.Image':
        if isinstance(image, dict):
            if image.get('path'):
                image = image['path']
            else:
                image = image['bytes']
        return load_image(image)

    def preprocess_video(self, video: VideoInput) -> List['Image.Image']:
        return video

    def preprocess_audio(self, audio: AudioInput) -> np.ndarray:
        return audio

    def preprocess_images(self, images: List[ImageInput]) -> List['Image.Image']:
        """Preprocess a list of images."""
        return [self.preprocess_image(img) for img in images]

    def preprocess_videos(self, videos: List[VideoInput]) -> List[List['Image.Image']]:
        """Preprocess a list of videos."""
        return [self.preprocess_video(video) for video in videos]

    def preprocess_audios(self, audios: List[AudioInput]) -> List[np.ndarray]:
        """Preprocess a list of audio clips."""
        return [self.preprocess_audio(audio) for audio in audios]

    def _invoke_pre_pipeline(self, trajectories: List[Trajectory]) -> List[Trajectory]:
        current = trajectories
        for pipeline in self.pre_pipeline:
            next_batch = []
            for trajectory in current:
                next_batch.extend(pipeline(trajectory))
            current = next_batch
        return current

    def _invoke_post_pipeline(self, input_features: List[InputFeature]) -> List[InputFeature]:
        current = input_features
        for pipeline in self.post_pipeline:
            next_batch = []
            for input_feature in current:
                next_batch.extend(pipeline(input_feature))
            current = next_batch
        return current

    def concat_input_feature(self, prompt_input_feature: InputFeature, new_tokens: List[int]) -> InputFeature:
        import copy
        assert self.truncation_strategy != 'split', 'concat_input_feature does not support `truncation_strategy=split`'
        result = copy.deepcopy(prompt_input_feature)
        prompt_ids = result['input_ids']
        input_ids = list(prompt_ids) + new_tokens
        labels = [-100] * len(prompt_ids) + new_tokens
        result['input_ids'] = input_ids
        result['labels'] = labels
        new_input_feature = self._invoke_post_pipeline([result])[0]
        result.update(new_input_feature)
        messages: List[Message] = result.get('messages')
        if messages is not None:
            response_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            messages.append(Message(role='assistant', content=response_text))
            result['messages'] = messages
        return result

    def _add_default_system(self, trajectory: Trajectory) -> List[Trajectory]:
        if self.use_chat_template and self.default_system:
            if trajectory['messages'][0]['role'] == 'user':
                trajectory['messages'].insert(0, Message(role='system', content=self.default_system))
            for (_, messages) in trajectory.get('extend_message', []):
                if messages and messages[0]['role'] == 'user':
                    messages.insert(0, Message(role='system', content=self.default_system))
        return [trajectory]

    def _check_max_length(self, input_feature: InputFeature) -> List[InputFeature]:
        if self.max_length and len(input_feature['input_ids']) > self.max_length:
            if self.truncation_strategy == 'raise':
                raise ValueError(f'An input message(length: {len(input_feature["input_ids"])} '
                                 f'exceeds the maximum length({self.max_length})')
            elif self.truncation_strategy == 'left':
                return [InputFeature(**{key: value[-self.max_length:] for key, value in input_feature.items()})]
            elif self.truncation_strategy == 'right':
                return [InputFeature(**{key: value[:self.max_length] for key, value in input_feature.items()})]
            else:  # split
                result = []
                total_length = len(input_feature['input_ids'])
                for start in range(0, total_length, self.max_length):
                    end = min(start + self.max_length, total_length)
                    result.append(InputFeature(**{key: value[start:end] for key, value in input_feature.items()}))
                return result
        else:
            return [input_feature]

    def _add_attention_fields(self, input_feature: InputFeature) -> List[InputFeature]:
        input_ids = input_feature['input_ids']
        input_feature['attention_mask'] = np.ones_like(input_ids)
        input_feature['position_ids'] = np.arange(len(input_ids))
        input_feature['length'] = len(input_ids)
        return [input_feature]

    def _roll_labels(self, input_feature: InputFeature) -> List[InputFeature]:
        input_feature['labels'] = np.roll(input_feature['labels'], -1, axis=-1)
        return [input_feature]

    def _build_mm_messages(self, trajectory: Trajectory) -> List[Trajectory]:
        messages = trajectory['messages']
        new_messages = []
        for message in messages:
            message = copy(message)
            content = message['content']
            msg_images = message.get('images')
            msg_videos = message.get('videos')
            msg_audios = message.get('audios')
            if msg_images:
                message['images'] = self.preprocess_images(msg_images)
                assert len(message['images']) == content.count(self.image_placeholder)
            if msg_videos:
                message['videos'] = self.preprocess_videos(msg_videos)
                assert len(message['videos']) == content.count(self.video_placeholder)
            if msg_audios:
                message['audios'] = self.preprocess_audios(msg_audios)
                assert len(message['audios']) == content.count(self.audio_placeholder)
            new_messages.append(
                transfer_to_standard_message(message, self.image_placeholder, self.video_placeholder,
                                             self.audio_placeholder, self.is_mm))

        trajectory['messages'] = new_messages
        return [trajectory]

    def _apply_chat_template(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs):
        messages = [dict(message) for message in trajectory['messages']]
        tools = [dict(tool) for tool in trajectory.get('tools', [])]
        inputs = self.processor.apply_chat_template(
            messages,
            tools=tools,
            padding=False,
            tokenize=True,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors='pt',
            **kwargs)
        return inputs

    def encode(self, trajectory: Trajectory, add_generation_prompt: bool = False) -> InputFeature:
        if self.use_chat_template:
            if add_generation_prompt:
                # For inference: just get input_ids with generation prompt, no labels needed
                encoded = self._apply_chat_template(trajectory, add_generation_prompt=True)
                input_ids = encoded.pop('input_ids')
                if hasattr(input_ids, 'squeeze'):
                    input_ids = input_ids.squeeze(0)
                labels = np.full_like(input_ids, -100)  # No labels for inference
            elif self._template_support_assistant_tokens_mask:
                encoded = self._apply_chat_template(trajectory, return_assistant_tokens_mask=True)
                input_ids = encoded.pop('input_ids')
                assistant_masks = encoded.pop('assistant_masks')
                labels = np.where(assistant_masks, input_ids, -100)
            else:
                input_ids, labels, encoded = tokenize_with_assistant_labels(self.tokenizer, self._apply_chat_template,
                                                                            trajectory)
        else:
            assert len(trajectory['messages']) == 1 and trajectory['messages'][0]['role'] == 'user'
            text = trajectory['messages'][0]['content']
            input_ids = self.tokenizer.encode(text)
            encoded = {}
            labels = deepcopy(input_ids)
        return InputFeature(
            input_ids=np.array(input_ids),
            labels=np.array(labels),
            **encoded,
        )

    @staticmethod
    def map_col_to_row(trajectories: Dict[str, Any]):
        if not trajectories:
            return []
        rows = []
        total_count = len(trajectories[next(iter(list(trajectories.keys())))])
        for i in range(total_count):
            row = {}
            for key in trajectories:
                row[key] = trajectories[key][i]
            rows.append(row)
        return rows

    @staticmethod
    def map_row_to_col(rows: List[Union[Dict[str, Any], InputFeature]]) -> Dict[str, List[Any]]:
        if not rows:
            return {}

        columns: Dict[str, List[Any]] = {}
        keys = rows[0].keys()

        for key in keys:
            columns[key] = [row[key] for row in rows]

        return columns

    def batch_encode(self,
                     trajectories: Union[Dict[str, Any], List[Trajectory]],
                     add_generation_prompt: bool = False) -> List[InputFeature]:
        output = []
        _transfer = False
        if isinstance(trajectories, Mapping):
            _transfer = True
            trajectories = self.map_col_to_row(trajectories)
        trajectories = self._invoke_pre_pipeline(trajectories)
        for trajectory in trajectories:
            output.append(self.encode(trajectory, add_generation_prompt=add_generation_prompt))
        output = self._invoke_post_pipeline(output)
        if _transfer:
            output = self.map_row_to_col(output)
        return output

    def check(self, trajectory: Trajectory) -> Optional[Trajectory]:
        encoded = None
        try:
            encoded = self.batch_encode([trajectory])
            if not encoded:
                return None
            else:
                return trajectory
        except Exception as e:
            import traceback
            print(f'[Template.check] Error encoding trajectory: {e}')
            traceback.print_exc()
            return None
        finally:
            if encoded:
                del encoded

    def batch_check(self, trajectories: List[Trajectory]) -> List[Optional[Trajectory]]:
        output = []
        for trajectory in trajectories:
            output.append(self.check(trajectory))
        return output

    def decode(self, token_ids: List[int], **kwargs) -> str:
        return self.processor.decode(token_ids, **kwargs)

    def batch_decode(self, token_ids: List[List[int]], **kwargs) -> List[str]:
        return [self.processor.decode(_ids, **kwargs) for _ids in token_ids]

    def _get_vision_token_id(self) -> Optional[int]:
        if self.config is not None:
            return getattr(self.config, 'image_token_id', None)
        else:
            return self.processor.encode(self.image_placeholder)

    def _post_encode(self, model: 'torch.nn.Module', inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def pre_forward_hook(self, model: 'torch.nn.Module', args, kwargs):
        if not self.is_mm:
            return args, kwargs
        device = next(model.parameters()).device
        old_kwargs = to_device(kwargs, device)
        kwargs = to_device(self._post_encode(model, old_kwargs), device)
        for k, v in old_kwargs.items():
            if k in {
                    'input_ids', 'attention_mask', 'labels', 'position_ids', 'output_hidden_states', 'logits_to_keep',
                    'max_length_q', 'max_length_k', 'cu_seq_lens_q', 'cu_seq_lens_k'
            } and k not in kwargs:
                kwargs[k] = v
        if 'inputs_embeds' in kwargs:
            kwargs.pop('input_ids', None)

        from peft import PeftModel
        if isinstance(model, PeftModel):
            base_model = model.model
        else:
            base_model = model
        parameters = inspect.signature(base_model.forward).parameters
        if 'position_ids' not in parameters:
            kwargs.pop('position_ids', None)
        return args, kwargs
