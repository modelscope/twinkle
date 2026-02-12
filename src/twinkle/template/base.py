# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from collections.abc import Mapping
from copy import deepcopy
from typing import List, Optional, Dict, Any, Literal, Union, TYPE_CHECKING, Callable

import numpy as np


from twinkle.data_format import Trajectory, InputFeature, Message
from twinkle.hub import HubOperation
from .utils import tokenize_with_assistant_labels, transfer_to_standard_message
if TYPE_CHECKING:
    import torch
    from PIL import Image

# Type aliases for multimodal data
ImageInput = Union[str, 'Image.Image', "torch.Tensor"]
VideoInput = Union[str, List['Image.Image'], "torch.Tensor"]
AudioInput = Union[str, np.ndarray, "torch.Tensor"]


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

        self.use_chat_template = use_chat_template
        self.max_length = max_length
        self.truncation_strategy = truncation_strategy
        self.default_system = default_system
        self._test_support_assistant_tokens_mask()
        self.pre_pipeline: List[Callable[[Trajectory], List[Trajectory]]] = [
            self._add_default_system, # Add a default system field
            self._build_mm_messages, # turn to standard mm messages
        ]
        self.post_pipeline: List[Callable[[InputFeature], List[InputFeature]]] = [
            self._check_max_length, # Check and split input_features
            self._add_attention_fields, # Add useful fields
            self._roll_labels, # roll labels
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
                {'role': 'user', 'content': [{'type': 'text', 'text': 'How are you?'}]},
                {'role': 'assistant', 'content': [{'type': 'text', 'text': 'Fine.'}]},
            ]
        else:
            dummy_inputs = [
                Message(role='user', content='How are you?'),
                Message(role='assistant', content='Fine.'),
            ]
        try:
            outputs = self.processor.apply_chat_template(
                dummy_inputs,
                return_assistant_tokens_mask=True,
                return_dict=True,
                tokenize=True
            )
            # Check if outputs is a dict (not all processors return dict even with return_dict=True)
            if isinstance(outputs, dict) and 'assistant_masks' in outputs:
                assistant_masks = outputs['assistant_masks']
                self._template_support_assistant_tokens_mask = (
                    0 < np.array(assistant_masks).sum() < len(assistant_masks)
                )
            else:
                # Processor doesn't support return_dict properly
                self._template_support_assistant_tokens_mask = False
        except Exception: # noqa
            # If any error occurs during testing, fall back to not supporting
            self._template_support_assistant_tokens_mask = False

    def preprocess_image(self, image: ImageInput) -> 'Image.Image':
        return image

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
            else: # split
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
        # TODO code untested
        messages = trajectory['messages']
        # Get images/videos from trajectory level (common case) or message level
        traj_images = trajectory.get('images') or []
        traj_videos = trajectory.get('videos') or []
        
        # Preprocess all trajectory-level images and videos
        if traj_images and self.is_mm:
            traj_images = self.preprocess_images(traj_images)
        if traj_videos and self.is_mm:
            traj_videos = self.preprocess_videos(traj_videos)
        
        # Distribute trajectory-level images to messages that contain placeholders
        image_idx = 0
        video_idx = 0
        new_messages = []
        for message in messages:
            # If message already has images/videos at message level, use those
            msg_images = message.get('images')
            msg_videos = message.get('videos')
            
            # If not, assign from trajectory level based on placeholder count
            if msg_images is None and self.is_mm:
                content = message.get('content', '')
                if isinstance(content, str):
                    placeholder_count = content.count(self.image_placeholder)
                    if placeholder_count > 0 and image_idx < len(traj_images):
                        msg_images = traj_images[image_idx:image_idx + placeholder_count]
                        image_idx += placeholder_count
            elif msg_images and self.is_mm:
                # Preprocess message-level images
                msg_images = self.preprocess_images(msg_images)
            
            if msg_videos is None and self.is_mm:
                content = message.get('content', '')
                if isinstance(content, str):
                    placeholder_count = content.count(self.video_placeholder)
                    if placeholder_count > 0 and video_idx < len(traj_videos):
                        msg_videos = traj_videos[video_idx:video_idx + placeholder_count]
                        video_idx += placeholder_count
            elif msg_videos and self.is_mm:
                # Preprocess message-level videos
                msg_videos = self.preprocess_videos(msg_videos)
            
            # Create message with images/videos attached
            msg_with_media = dict(message)
            if msg_images:
                msg_with_media['images'] = msg_images
            if msg_videos:
                msg_with_media['videos'] = msg_videos
            
            new_messages.append(transfer_to_standard_message(
                msg_with_media, self.image_placeholder, self.video_placeholder, self.is_mm))
        
        trajectory['messages'] = new_messages
        return [trajectory]

    def _apply_chat_template(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs):
        messages = [dict(message) for message in trajectory['messages']]
        tools = [dict(tool) for tool in trajectory.get('tools', [])]
        inputs = self.processor.apply_chat_template(messages, tools=tools, padding=False,
                                           tokenize=True, return_dict=True,
                                           add_generation_prompt=add_generation_prompt, return_tensors='pt', **kwargs)
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
                input_ids, labels, encoded = tokenize_with_assistant_labels(self.tokenizer, self._apply_chat_template, trajectory)
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

    def batch_encode(self, trajectories: Union[Dict[str, Any], List[Trajectory]], add_generation_prompt: bool = False) -> List[InputFeature]:
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
            print(f"[Template.check] Error encoding trajectory: {e}")
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

    def post_encode(self, model: 'torch.nn.Module', inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform inputs for model forward.

        Default: use helper methods for embedding merge.
        Override if model handles internally (like Qwen3-VL).
        """
        input_ids = inputs.get('input_ids')
        if input_ids is None:
            return inputs

        text_embeds = self._get_text_embeddings(model, input_ids)
        vision_embeds = self._get_vision_embeddings(model, inputs)

        if vision_embeds is not None:
            inputs_embeds = self._merge_vision_embeddings(
                text_embeds, vision_embeds, input_ids, inputs
            )
        else:
            inputs_embeds = text_embeds

        result = {k: v for k, v in inputs.items() if k != 'input_ids'}
        result['inputs_embeds'] = inputs_embeds
        return result

    def _get_text_embeddings(self, model: 'torch.nn.Module', input_ids: 'torch.Tensor') -> 'torch.Tensor':
        """Get text embeddings from model."""
        embed_fn = None
        if hasattr(model, 'get_input_embeddings'):
            embed_fn = model.get_input_embeddings()
        elif hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_fn = model.model.embed_tokens
        elif hasattr(model, 'language_model') and hasattr(model.language_model, 'embed_tokens'):
            embed_fn = model.language_model.embed_tokens

        if embed_fn is None:
            raise ValueError("Cannot find embedding layer in model")

        return embed_fn(input_ids)

    def _get_vision_embeddings(self, model: 'torch.nn.Module', inputs: Dict[str, Any]) -> Optional['torch.Tensor']:
        """Get vision embeddings. Override in subclass."""
        return None

    def _get_vision_token_id(self) -> Optional[int]:
        """Get vision placeholder token ID. Override in subclass."""
        return self.processor.encode(self.image_placeholder)

    def _merge_vision_embeddings(
            self,
            text_embeds: 'torch.Tensor',
            vision_embeds: 'torch.Tensor',
            input_ids: 'torch.Tensor',
            inputs: Dict[str, Any]
    ) -> 'torch.Tensor':
        """Merge vision embeddings at placeholder positions."""
        vision_token_id = self._get_vision_token_id()
        if vision_token_id is None:
            return text_embeds

        vision_mask = (input_ids == vision_token_id).unsqueeze(-1).expand_as(text_embeds)
        vision_embeds = vision_embeds.to(device=text_embeds.device, dtype=text_embeds.dtype)
        vision_mask = vision_mask.to(device=text_embeds.device)

        return text_embeds.masked_scatter(vision_mask, vision_embeds)

    def _get_position_ids(self, inputs: Dict[str, Any]) -> Optional['torch.Tensor']:
        """Get position_ids. Override for models with special position encoding."""
        return None
