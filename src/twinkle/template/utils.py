# Copyright (c) ModelScope Contributors. All rights reserved.
import inspect
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeVar

from twinkle.data_format import Message, Trajectory
from twinkle.utils import get_logger, to_device

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger()
_T = TypeVar('_T')


def _coerce_ids_to_list(ids: Any) -> List[int]:
    import torch
    if isinstance(ids, torch.Tensor):
        while ids.dim() > 1:
            if ids.shape[0] != 1:
                raise ValueError('_coerce_ids_to_list expects a single-sample tensor (leading dims of size 1); '
                                 f'got shape {tuple(ids.shape)}. Pass one trajectory at a time.')
            ids = ids[0]
        return ids.tolist()
    return ids


def _convert_to_vlm_format(messages: List[Dict]) -> List[Dict]:
    converted = []
    for msg in messages:
        new_msg = dict(msg)
        content = msg.get('content')
        # If content is a string, convert to list format for VLM processors
        if isinstance(content, str):
            new_msg['content'] = [{'type': 'text', 'text': content}]
        converted.append(new_msg)
    return converted


def _is_vlm_processor(tokenizer) -> bool:
    if hasattr(tokenizer, 'tokenizer') and hasattr(tokenizer, 'image_processor'):
        return True
    return False


def _load_image(img: Any) -> Optional[Any]:
    """Load images to PIL format."""
    import io
    from PIL import Image

    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, str):
        if img.startswith(('http://', 'https://')):
            import requests
            resp = requests.get(img, timeout=30)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content))
        else:
            return Image.open(img)
    elif isinstance(img, bytes):
        return Image.open(io.BytesIO(img))
    elif isinstance(img, dict) and 'bytes' in img:
        return Image.open(io.BytesIO(img['bytes']))
    else:
        return img


def _transfer_single_message(
    content: str,
    image_placeholder: str,
    video_placeholder: str,
    audio_placeholder: str,
    images: list | None = None,
    videos: list | None = None,
    audios: list | None = None,
) -> list[dict]:
    if not content:
        return []

    media_configs = [
        (image_placeholder, 'image', images or []),
        (video_placeholder, 'video', videos or []),
        (audio_placeholder, 'audio', audios or []),
    ]

    placeholders = []
    for placeholder, media_type, media_list in media_configs:
        if not placeholder:
            continue
        start = 0
        media_idx = 0
        while (pos := content.find(placeholder, start)) != -1:
            if media_idx >= len(media_list):
                # More placeholders than provided media entries: stop scanning so the extra
                # placeholder text is preserved verbatim instead of being silently consumed
                # (which would make user-supplied media references vanish without warning).
                logger.warning(f'placeholder {placeholder!r} appears more times than provided '
                               f'{media_type} entries ({len(media_list)}); extra occurrences are kept as literal text.')
                break
            url = media_list[media_idx]
            placeholders.append((pos, len(placeholder), media_type, url))
            media_idx += 1
            start = pos + len(placeholder)

    if not placeholders:
        return [{'type': 'text', 'text': content}] if content.strip() else []

    placeholders.sort(key=lambda x: x[0])

    result = []
    cursor = 0

    for pos, length, media_type, url in placeholders:
        text_segment = content[cursor:pos]
        if text_segment.strip():
            result.append({'type': 'text', 'text': text_segment})

        if url is not None:
            result.append({'type': media_type, 'url': url})

        cursor = pos + length

    trailing_text = content[cursor:]
    if trailing_text.strip():
        result.append({'type': 'text', 'text': trailing_text})

    return result


def transfer_to_standard_message(message: Message, image_placeholder, video_placeholder, audio_placeholder, is_mm):
    if is_mm:
        new_content = _transfer_single_message(message['content'], image_placeholder, video_placeholder,
                                               audio_placeholder, message.get('images'), message.get('videos'),
                                               message.get('audios'))
    else:
        new_content = [{'type': 'text', 'text': message['content']}]

    return Message(
        role=message['role'],
        content=new_content,
        tool_calls=message.get('tool_calls'),
        reasoning_content=message.get('reasoning_content'))


def get_inputs_embeds_hf(inputs_embeds, inputs, visual, processor, config):
    input_ids = inputs['input_ids']
    pixel_values = inputs.get('pixel_values')
    pixel_values_videos = inputs.get('pixel_values_videos')
    image_grid_thw = inputs.get('image_grid_thw')
    video_grid_thw = inputs.get('video_grid_thw')
    dtype = visual.dtype
    if pixel_values is None and pixel_values_videos is None:
        from PIL import Image
        images = [Image.new('RGB', (32, 32), (0, 0, 0))]
        media_inputs = processor.image_processor(images=images, return_tensors='pt')
        media_inputs = to_device(media_inputs, input_ids.device)
        pixel_values = media_inputs['pixel_values'].type(dtype)
        image_embeds = visual(pixel_values, grid_thw=media_inputs['image_grid_thw'])
        if hasattr(image_embeds, 'pooler_output'):
            image_embeds = image_embeds.pooler_output
        inputs_embeds = inputs_embeds + image_embeds.mean().to(device=inputs_embeds.device) * 0.
    else:
        import torch
        if pixel_values is None:
            pixel_values_mixed = pixel_values_videos
            grid_thw = video_grid_thw
        elif pixel_values_videos is None:
            pixel_values_mixed = pixel_values
            grid_thw = image_grid_thw
        else:
            pixel_values_mixed = torch.concat([pixel_values, pixel_values_videos], dim=0)
            grid_thw = torch.concat([image_grid_thw, video_grid_thw], dim=0)
        pixel_values_mixed = pixel_values_mixed.type(dtype)
        mixed_embeds = visual(pixel_values_mixed, grid_thw=grid_thw)
        if hasattr(mixed_embeds, 'pooler_output'):
            mixed_embeds = mixed_embeds.pooler_output
        if pixel_values is None:
            image_embeds = None
            video_embeds = mixed_embeds
        elif pixel_values_videos is None:
            image_embeds = mixed_embeds
            video_embeds = None
        else:
            merge_length = processor.image_processor.merge_size**2
            image_tokens = (image_grid_thw.prod(dim=-1) // merge_length).sum()
            image_embeds = mixed_embeds[:image_tokens]
            video_embeds = mixed_embeds[image_tokens:]

        if image_embeds is not None:
            image_mask = (input_ids == config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = image_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if video_embeds is not None:
            video_mask = (input_ids == config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            video_mask = video_mask.to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    return inputs_embeds


class TokenizeByRound:
    """Tokenize by encoding messages round-by-round.

    This approach handles <think></think> tags correctly by encoding each message
    incrementally, determining token boundaries by comparing consecutive encode results.

    For assistant messages, uses add_generation_prompt=True to exclude the assistant
    prefix (e.g., '<|im_start|>assistant\n') from training labels.
    """

    @staticmethod
    def tokenize_with_assistant_labels(tokenizer: 'PreTrainedTokenizer',
                                       encode_func: Callable,
                                       trajectory: Trajectory,
                                       train_indices: Optional[set] = None,
                                       **kwargs) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """Tokenize trajectory and generate labels for assistant turns.

        Args:
            tokenizer: The tokenizer (unused, kept for interface compatibility).
            encode_func: Function to encode a trajectory. Must support add_generation_prompt.
            trajectory: The trajectory containing messages.
            train_indices: If provided, only label assistant messages whose
                message index is in this set.  ``None`` means label all.

        Returns:
            Tuple of (input_ids, labels, extra_encoded_fields).
            Labels are -100 for non-assistant tokens, original token id for assistant content tokens.
            Assistant prefix tokens (e.g., '<|im_start|>assistant\n') are excluded from training.
        """
        kwargs.pop('add_generation_prompt', None)
        messages = trajectory['messages']

        # Encode full trajectory
        encoded = encode_func(trajectory, **kwargs)
        full_ids = _coerce_ids_to_list(encoded.pop('input_ids'))

        # Initialize labels: all -100 (not trained)
        labels = [-100] * len(full_ids)

        if not messages:
            return full_ids, labels, encoded

        for i, msg in enumerate(messages):
            if msg['role'] != 'assistant':
                continue
            if train_indices is not None and i not in train_indices:
                continue

            # Get position AFTER assistant prefix:
            # encode(messages[:i], add_generation_prompt=True) includes the prefix
            partial_trajectory = copy(trajectory)
            partial_trajectory['messages'] = list(messages[:i])
            partial_ids = _coerce_ids_to_list(
                encode_func(partial_trajectory, add_generation_prompt=True, **kwargs)['input_ids'])
            start_pos = len(partial_ids)

            # Get end position: encode(messages[:i+1]) includes full assistant turn
            partial_trajectory = copy(trajectory)
            partial_trajectory['messages'] = list(messages[:i + 1])
            partial_ids = _coerce_ids_to_list(encode_func(partial_trajectory, **kwargs)['input_ids'])
            end_pos = len(partial_ids)

            # Mark assistant CONTENT tokens as trainable (excluding prefix)
            for j in range(start_pos, min(end_pos, len(full_ids))):
                labels[j] = full_ids[j]

        return full_ids, labels, encoded


class TokenizeByPlaceHolder:

    PLACEHOLDER = '<<<ASSISTANT_PLACEHOLDER_7f3d2a1b>>>'

    @staticmethod
    def find_subsequence(seq: List[int], subseq: List[int], start: int = 0) -> int:
        """Find the first index of `subseq`"""
        subseq_len = len(subseq)
        for i in range(start, len(seq) - subseq_len + 1):
            if seq[i:i + subseq_len] == subseq:
                return i
        return -1

    @staticmethod
    def split_by_subsequence(seq: List[int], subseq: List[int]) -> List[List[int]]:
        """Split seq by subseq"""
        parts = []
        start = 0
        subseq_len = len(subseq)

        while True:
            pos = TokenizeByPlaceHolder.find_subsequence(seq, subseq, start)
            if pos == -1:
                parts.append(seq[start:])
                break
            parts.append(seq[start:pos])
            start = pos + subseq_len

        return parts

    @staticmethod
    def build_labels(
        full_ids: List[int],
        template_parts: List[List[int]],
    ) -> List[int]:
        labels = list(full_ids)
        pos = 0

        for part in template_parts:
            if not part:
                continue

            match_pos = TokenizeByPlaceHolder.find_subsequence(full_ids, part, pos)

            if match_pos == -1:
                # should not happen
                raise ValueError(f'Template part not found in full_ids at position {pos}')

            for i in range(match_pos, match_pos + len(part)):
                labels[i] = -100

            pos = match_pos + len(part)

        return labels

    @staticmethod
    def tokenize_with_assistant_labels(
        tokenizer: 'PreTrainedTokenizer',
        encode_func: Callable,
        trajectory: Trajectory,
    ) -> Tuple[List[int], List[int], Dict[str, Any]]:
        import torch
        placeholder: str = TokenizeByPlaceHolder.PLACEHOLDER
        messages = [dict(message) for message in trajectory['messages']]

        _dummy_messages = []
        assistant_count = 0
        for msg in messages:
            if msg['role'] == 'assistant':
                msg = deepcopy(msg)
                if isinstance(msg['content'], str):
                    msg['content'] = placeholder
                else:
                    text_items = [c for c in msg['content'] if isinstance(c, dict) and c.get('type') == 'text']
                    if len(text_items) != 1:
                        raise ValueError('TokenizeByPlaceHolder requires exactly one text item in assistant '
                                         f'content, got {len(text_items)} (content={msg["content"]!r}).')
                    text_items[0]['text'] = placeholder
                assistant_count += 1
            _dummy_messages.append(msg)

        encoded = encode_func(trajectory)
        full_ids = _coerce_ids_to_list(encoded.pop('input_ids'))

        _dummy_trajectory = copy(trajectory)
        _dummy_trajectory['messages'] = _dummy_messages
        template_ids = _coerce_ids_to_list(encode_func(_dummy_trajectory)['input_ids'])

        extra_kwargs = {}
        if 'add_special_tokens' in inspect.signature(tokenizer.encode).parameters:
            extra_kwargs['add_special_tokens'] = False
        placeholder_ids = tokenizer.encode(placeholder, **extra_kwargs)
        template_parts = TokenizeByPlaceHolder.split_by_subsequence(template_ids, placeholder_ids)

        if len(template_parts) != assistant_count + 1:
            raise ValueError(f'Expected {assistant_count + 1} parts, got {len(template_parts)}. '
                             'Placeholder might appear in original content.')

        try:
            labels = TokenizeByPlaceHolder.build_labels(full_ids, template_parts)
        except ValueError as e:
            newline_placeholder_ids = tokenizer.encode('\n' + placeholder, **extra_kwargs)
            template_parts = TokenizeByPlaceHolder.split_by_subsequence(template_ids, newline_placeholder_ids)
            if len(template_parts) == assistant_count + 1:
                labels = TokenizeByPlaceHolder.build_labels(full_ids, template_parts)
            else:
                raise e
        last_role = messages[-1]['role'] if messages else None
        if last_role == 'assistant' and labels and labels[-1] == -100:
            end_idx = len(labels)
            start_idx = end_idx - 1
            while start_idx > 0 and labels[start_idx - 1] == -100:
                start_idx -= 1

            for i in range(start_idx, end_idx):
                labels[i] = full_ids[i]

        return full_ids, labels, encoded
