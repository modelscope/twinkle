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
from .utils import TokenizeByRound, transfer_to_standard_message

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
                 enable_thinking: bool = True,
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
        self.enable_thinking = enable_thinking
        self.truncation_strategy = truncation_strategy
        self.default_system = default_system
        self._test_support_assistant_tokens_mask()
        self.pre_pipeline: List[Callable[[Trajectory], List[Trajectory]]] = [
            self._add_default_system,  # Add a default system field
            self._to_standard_reasoning_content,  # Convert thinking to standard field
            self._build_standard_messages,  # turn to standard mm messages
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
            image = image.get('bytes') or image.get('path')
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
        import torch
        assert self.truncation_strategy != 'split', 'concat_input_feature does not support `truncation_strategy=split`'
        result = copy.deepcopy(prompt_input_feature)
        prompt_ids = result['input_ids']
        input_ids = list(prompt_ids) + new_tokens
        labels = [-100] * len(prompt_ids) + new_tokens
        result['input_ids'] = input_ids
        result['labels'] = labels
        if 'mm_token_type_ids' in result:
            token_ids_shape = result['mm_token_type_ids'].shape
            device = result['mm_token_type_ids'].device
            padded_tokens = torch.zeros((token_ids_shape[0], len(new_tokens))).to(device)
            result['mm_token_type_ids'] = torch.cat((result['mm_token_type_ids'], padded_tokens), dim=1)
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
        return [trajectory]

    def _to_standard_reasoning_content(self, trajectory: Trajectory) -> List[Trajectory]:

        def _extract_reasoning_content(messages: list[Message]) -> List[Message]:
            result = []
            for message in messages:
                message = message.copy()
                if message.get('role') == 'assistant':
                    content = message.get('content', '')
                    if 'reasoning_content' not in message and isinstance(content, str):
                        if '</think>' in content:
                            reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip(
                                '\n')
                            new_content = content.split('</think>')[-1].lstrip('\n')

                            message['reasoning_content'] = reasoning_content
                            message['content'] = new_content

                result.append(message)

            return result

        trajectory['messages'] = _extract_reasoning_content(trajectory['messages'])
        return [trajectory]

    def _truncate_feature(self, feature: InputFeature, strategy: str) -> InputFeature:
        """Truncate input_ids and labels in a single InputFeature."""
        length = len(feature['input_ids'])
        if length <= self.max_length:
            return feature
        if strategy == 'raise':
            raise ValueError(f'Input length {length} exceeds max_length {self.max_length}')
        result = dict(feature)
        if strategy == 'left':
            result['input_ids'] = result['input_ids'][-self.max_length:]
            if 'labels' in result:
                result['labels'] = result['labels'][-self.max_length:]
        elif strategy == 'right':
            result['input_ids'] = result['input_ids'][:self.max_length]
            if 'labels' in result:
                result['labels'] = result['labels'][:self.max_length]
        return InputFeature(**result)

    def set_mm_position_ids(self, input_feature: InputFeature):
        return np.arange(len(input_feature['input_ids']))

    def _check_max_length(self, input_feature: InputFeature) -> List[InputFeature]:
        if not self.max_length or 'input_ids' not in input_feature:
            return [input_feature]

        strategy = self.truncation_strategy

        # Split strategy
        if strategy == 'split':
            results = []
            for start in range(0, len(input_feature['input_ids']), self.max_length):
                end = min(start + self.max_length, len(input_feature['input_ids']))
                feat = dict(input_feature)
                feat['input_ids'] = feat['input_ids'][start:end]
                if 'labels' in feat:
                    feat['labels'] = feat['labels'][start:end]
                results.append(InputFeature(**feat))
            return results

        # left/right/raise
        return [self._truncate_feature(input_feature, strategy)]

    def _add_attention_fields(self, input_feature: InputFeature) -> List[InputFeature]:
        if 'input_ids' not in input_feature:
            return [input_feature]
        input_ids = input_feature['input_ids']
        input_feature['attention_mask'] = np.ones_like(input_ids)
        input_feature['position_ids'] = self.set_mm_position_ids(input_feature)
        input_feature['length'] = len(input_ids)
        return [input_feature]

    def _roll_labels(self, input_feature: InputFeature) -> List[InputFeature]:
        if 'input_ids' not in input_feature:
            return [input_feature]
        input_feature['labels'] = np.roll(input_feature['labels'], -1, axis=-1)
        return [input_feature]

    def _process_mm_messages(self, messages: List, images: List, videos: List, audios: List) -> List:
        """Process multimodal content with trajectory-level media.

        Args:
            messages: List of messages to process
            images: Trajectory-level images
            videos: Trajectory-level videos
            audios: Trajectory-level audios
        """
        # Determine format: list or string (check first non-system message)
        is_list_format = any(isinstance(m.get('content'), list) for m in messages if m.get('role') != 'system')

        if is_list_format:
            return self._process_mm_list_format(messages, images, videos, audios)
        else:
            return self._process_mm_string_format(messages, images, videos, audios)

    def _process_mm_list_format(self, messages: List, images: List, videos: List, audios: List) -> List:
        """Process list format content with trajectory-level media."""
        img_iter, vid_iter, aud_iter = iter(images), iter(videos), iter(audios)
        new_messages = []
        first_user_idx = None

        for idx, message in enumerate(messages):
            message = copy(message)
            content = message.get('content')

            if message.get('role') == 'user' and first_user_idx is None:
                first_user_idx = idx

            # Non-user messages: convert to list format but don't consume iterator
            if message.get('role') != 'user':
                if isinstance(content, str):
                    message['content'] = [{'type': 'text', 'text': content}] if content else []
                new_messages.append(message)
                continue

            # User messages: process list content and fill placeholders
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get('type')
                    # Check if block has inline data
                    has_data = any(block.get(k) for k in (btype, 'url', 'path'))
                    if has_data:
                        # Preprocess inline URL
                        for key in (btype, 'url', 'path'):
                            if key and block.get(key) is not None:
                                block[key] = getattr(self, f'preprocess_{btype}', lambda x: x)(block[key])
                                break
                    elif btype == 'image':
                        url = next(img_iter, None)
                        if url:
                            block['url'] = url
                    elif btype == 'video':
                        url = next(vid_iter, None)
                        if url:
                            block['url'] = url
                    elif btype == 'audio':
                        url = next(aud_iter, None)
                        if url:
                            block['url'] = url
            elif isinstance(content, str):
                # Convert string content to list format
                message['content'] = [{'type': 'text', 'text': content}] if content else []

            new_messages.append(message)

        # Prepend remaining media to first user message
        if first_user_idx is not None:
            prepend = [{'type': 'image', 'url': u} for u in img_iter]
            prepend += [{'type': 'video', 'url': u} for u in vid_iter]
            prepend += [{'type': 'audio', 'url': u} for u in aud_iter]
            if prepend:
                msg = new_messages[first_user_idx]
                content = msg.get('content', [])
                if not isinstance(content, list):
                    content = [{'type': 'text', 'text': content}] if content else []
                msg['content'] = prepend + content

        return new_messages

    def _process_mm_string_format(self, messages: List, images: List, videos: List, audios: List) -> List:
        """Process string format content with trajectory-level media."""
        # Count total placeholders across all messages
        total_img = sum(
            m.get('content', '').count(self.image_placeholder) for m in messages
            if isinstance(m.get('content'), str) and m.get('role') != 'system')
        total_vid = sum(
            m.get('content', '').count(self.video_placeholder) for m in messages
            if isinstance(m.get('content'), str) and m.get('role') != 'system')
        total_aud = sum(
            m.get('content', '').count(self.audio_placeholder) for m in messages
            if isinstance(m.get('content'), str) and m.get('role') != 'system')

        img_missing = len(images) - total_img
        vid_missing = len(videos) - total_vid
        aud_missing = len(audios) - total_aud

        # Find first user message index
        first_user_idx = next((i for i, m in enumerate(messages) if m.get('role') == 'user'), None)

        new_messages = []
        img_iter, vid_iter, aud_iter = iter(images), iter(videos), iter(audios)

        for idx, message in enumerate(messages):
            message = copy(message)
            content = message.get('content', '')

            # Non-user messages: convert to list format but don't consume iterator
            if message.get('role') != 'user':
                if isinstance(content, str):
                    message['content'] = [{'type': 'text', 'text': content}] if content else []
                new_messages.append(message)
                continue

            # User messages: skip non-string content
            if not isinstance(content, str):
                new_messages.append(message)
                continue

            # Prepend missing placeholders to first user message
            if idx == first_user_idx:
                if img_missing > 0:
                    content = self.image_placeholder * img_missing + content
                if vid_missing > 0:
                    content = self.video_placeholder * vid_missing + content
                if aud_missing > 0:
                    content = self.audio_placeholder * aud_missing + content

            # Count placeholders in this message and assign media
            msg_img_count = content.count(self.image_placeholder)
            msg_vid_count = content.count(self.video_placeholder)
            msg_aud_count = content.count(self.audio_placeholder)

            msg_images = [next(img_iter, None) for _ in range(msg_img_count)]
            msg_videos = [next(vid_iter, None) for _ in range(msg_vid_count)]
            msg_audios = [next(aud_iter, None) for _ in range(msg_aud_count)]

            message['content'] = content
            message['images'] = [i for i in msg_images if i is not None]
            message['videos'] = [v for v in msg_videos if v is not None]
            message['audios'] = [a for a in msg_audios if a is not None]

            message = transfer_to_standard_message(message, self.image_placeholder, self.video_placeholder,
                                                   self.audio_placeholder, self.is_mm)

            new_messages.append(message)

        return new_messages

    def _build_standard_messages(self, trajectory: Trajectory) -> List[Trajectory]:
        # Extract trajectory-level media
        images = self.preprocess_images(trajectory.pop('images', None) or [])
        videos = self.preprocess_videos(trajectory.pop('videos', None) or [])
        audios = self.preprocess_audios(trajectory.pop('audios', None) or [])

        trajectory['messages'] = self._process_mm_messages(trajectory['messages'], images, videos, audios)
        if not self.is_mm:
            for message in trajectory['messages']:
                message['content'] = message['content'][0]['text']
        return [trajectory]

    def _apply_chat_template(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs):
        messages = [dict(message) for message in trajectory['messages']]
        # Arrow serialization may pad content blocks with null keys (e.g. 'image': None
        # on text-only blocks). Jinja checks `'image' in item` on dict keys, so these
        # phantom keys cause wrong token counts. Strip them here.
        for msg in messages:
            if not isinstance(msg.get('content'), list):
                continue
            msg['content'] = [{
                k: v
                for k, v in b.items() if v is not None
            } for b in msg['content'] if isinstance(b, dict)]
        tools = [dict(tool) for tool in trajectory.get('tools', [])]
        if 'tokenize' not in kwargs:
            kwargs['tokenize'] = True
        if 'enable_thinking' not in kwargs:
            kwargs['enable_thinking'] = self.enable_thinking
        inputs = self.processor.apply_chat_template(
            messages,
            tools=tools,
            padding=False,
            return_dict=True,
            add_generation_prompt=add_generation_prompt,
            return_tensors='pt',
            **kwargs)
        return inputs

    def _encode_messages(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs) -> InputFeature:
        """Encode a single trajectory's messages into InputFeature."""
        labels = None
        input_ids = None
        if self.use_chat_template:
            if add_generation_prompt:
                # For inference: just get input_ids with generation prompt, no labels needed
                encoded = self._apply_chat_template(trajectory, add_generation_prompt=True, **kwargs)
                if 'input_ids' in encoded:
                    input_ids = encoded.pop('input_ids')
                    if hasattr(input_ids, 'squeeze'):
                        input_ids = input_ids.squeeze(0)
                    labels = np.full_like(input_ids, -100)  # No labels for inference
            elif self._template_support_assistant_tokens_mask:
                encoded = self._apply_chat_template(
                    trajectory, return_assistant_tokens_mask=kwargs.get('tokenize', True), **kwargs)
                if 'input_ids' in encoded:
                    input_ids = encoded.pop('input_ids')
                    assistant_masks = encoded.pop('assistant_masks')
                    labels = np.where(assistant_masks, input_ids, -100)
            else:
                if kwargs.get('tokenize', True):
                    input_ids, labels, encoded = TokenizeByRound.tokenize_with_assistant_labels(
                        self.tokenizer, self._apply_chat_template, trajectory, **kwargs)
                else:
                    encoded = self._apply_chat_template(trajectory, **kwargs)
        else:
            assert len(trajectory['messages']) == 1 and trajectory['messages'][0]['role'] == 'user'
            text = trajectory['messages'][0]['content']
            input_ids = self.tokenizer.encode(text, **kwargs)
            encoded = {}
            labels = deepcopy(input_ids)
        if isinstance(encoded, str):
            input_feature = {'prompt': encoded}
        else:
            input_feature = InputFeature(
                input_ids=np.array(input_ids),
                labels=np.array(labels),
                **encoded,
            )
        trajectory.update(input_feature)
        return trajectory

    def encode(self, trajectory: Trajectory, add_generation_prompt: bool = False, **kwargs) -> InputFeature:
        return self._encode_messages(trajectory, add_generation_prompt, **kwargs)

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

    def _is_trajectory(self, obj: Any) -> bool:
        """Check if an object is a Trajectory (has 'messages' key)."""
        return isinstance(obj, Mapping) and 'messages' in obj

    def _get_trajectory_keys(self, trajectories: Mapping, is_columnar: bool) -> List[str]:
        """Get keys whose values are lists of Trajectories."""
        keys = []
        if is_columnar:
            for k, v in trajectories.items():
                if isinstance(v, list) and v and self._is_trajectory(v[0]):
                    keys.append(k)
        else:
            for k, v in trajectories.items():
                if v is not None and self._is_trajectory(v):
                    keys.append(k)
        return keys

    def batch_encode(
        self,
        trajectories: Union[Dict[str, List[Any]], List[Trajectory]],
        add_generation_prompt: bool = False,
        **kwargs,
    ) -> Union[Dict[str, Any], List[InputFeature]]:
        """Encode trajectories into InputFeatures.

        Args:
            trajectories: Either List[Trajectory] or columnar Dict[str, List].
                For nested trajectories, columnar format with trajectory list columns
                (e.g., 'chosen'/'rejected') is supported.
            add_generation_prompt: Whether to add generation prompt.

        Returns:
            List[InputFeature] or columnar Dict[str, List[InputFeature]].
        """
        _transfer = False

        # Handle list input
        if isinstance(trajectories, list) and len(trajectories) > 0:
            # Check if first element has nested trajectories
            if isinstance(trajectories[0], Mapping) and len(self._get_trajectory_keys(trajectories[0], False)) > 0:
                # Convert row→columnar, process with columnar logic, convert back
                columnar = self.map_row_to_col(trajectories)
                encoded = self.batch_encode(columnar, add_generation_prompt)
                return self.map_col_to_row(encoded)

        if isinstance(trajectories, Mapping):
            _transfer = True
            # Check if it has nested trajectory columns
            traj_keys = self._get_trajectory_keys(trajectories, True)
            if traj_keys:
                # Nested format: encode each trajectory list separately, keep other columns
                return {
                    key:
                    self.batch_encode(trajectories[key], add_generation_prompt)
                    if key in traj_keys else trajectories[key]
                    for key in trajectories
                }
            else:
                # Standard columnar format
                trajectories = self.map_col_to_row(trajectories)

        # Process List[Trajectory]
        trajectories = self._invoke_pre_pipeline(trajectories)

        # Use thread pool for parallel encoding
        from concurrent.futures import ThreadPoolExecutor
        from functools import partial
        encode_fn = partial(
            self.encode,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        with ThreadPoolExecutor() as executor:
            output = list(executor.map(encode_fn, trajectories))

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
