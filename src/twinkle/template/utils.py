# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import List, Dict, Any, Tuple, Optional, Callable
from copy import deepcopy, copy

from transformers import PreTrainedTokenizer

from twinkle.data_format import Trajectory, Message

PLACEHOLDER = "<<<ASSISTANT_PLACEHOLDER_7f3d2a1b>>>"


def find_subsequence(seq: List[int], subseq: List[int], start: int = 0) -> int:
    """Find the first index of `subseq`"""
    subseq_len = len(subseq)
    for i in range(start, len(seq) - subseq_len + 1):
        if seq[i:i + subseq_len] == subseq:
            return i
    return -1


def split_by_subsequence(seq: List[int], subseq: List[int]) -> List[List[int]]:
    """Split seq by subseq"""
    parts = []
    start = 0
    subseq_len = len(subseq)

    while True:
        pos = find_subsequence(seq, subseq, start)
        if pos == -1:
            parts.append(seq[start:])
            break
        parts.append(seq[start:pos])
        start = pos + subseq_len

    return parts


def build_labels(
        full_ids: List[int],
        template_parts: List[List[int]],
) -> List[int]:
    labels = list(full_ids)
    pos = 0

    for part in template_parts:
        if not part:
            continue

        match_pos = find_subsequence(full_ids, part, pos)

        if match_pos == -1:
            # should not happen
            raise ValueError(f"Template part not found in full_ids at position {pos}")

        for i in range(match_pos, match_pos + len(part)):
            labels[i] = -100

        pos = match_pos + len(part)

    return labels


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


def tokenize_with_assistant_labels(
        tokenizer: PreTrainedTokenizer,
        encode_func: Callable,
        trajectory: Trajectory,
        placeholder: str = PLACEHOLDER,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    import torch
    messages = [dict(message) for message in trajectory['messages']]

    _dummy_messages = []
    assistant_count = 0
    for msg in messages:
        if msg['role'] == 'assistant':
            msg = deepcopy(msg)
            if isinstance(msg["content"], str):
                msg["content"] = placeholder
            else:
                msg["content"][0]['text'] = placeholder
            assistant_count += 1
        _dummy_messages.append(msg)

    encoded = encode_func(
        trajectory,
    )
    full_ids = encoded.pop('input_ids')
    if isinstance(full_ids, torch.Tensor):
        full_ids = full_ids.tolist()[0]

    _dummy_trajectory = copy(trajectory)
    _dummy_trajectory['messages'] = _dummy_messages
    template_ids = encode_func(
        _dummy_trajectory,
    )
    template_ids = template_ids['input_ids']
    if isinstance(template_ids, torch.Tensor):
        template_ids = template_ids.tolist()[0]

    placeholder_ids = tokenizer.encode(placeholder, add_special_tokens=False)
    template_parts = split_by_subsequence(template_ids, placeholder_ids)

    if len(template_parts) != assistant_count + 1:
        raise ValueError(
            f"Expected {assistant_count + 1} parts, got {len(template_parts)}. "
            "Placeholder might appear in original content."
        )

    labels = build_labels(full_ids, template_parts)
    if labels and labels[-1] == -100:
        end_idx = len(labels)
        start_idx = end_idx - 1
        while start_idx > 0 and labels[start_idx - 1] == -100:
            start_idx -= 1

        for i in range(start_idx, end_idx):
            labels[i] = full_ids[i]

    return full_ids, labels, encoded


def _load_image(img: Any) -> Optional[Any]:
    """Load images to PIL format."""
    from PIL import Image
    import io

    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, str):
        if img.startswith(('http://', 'https://')):
            import requests
            resp = requests.get(img, timeout=30)
            return Image.open(io.BytesIO(resp.content))
        else:
            return Image.open(img)
    elif isinstance(img, bytes):
        return Image.open(io.BytesIO(img))
    elif isinstance(img, dict) and 'bytes' in img:
        return Image.open(io.BytesIO(img['bytes']))
    else:
        return img


def _transfer_single_message(content: str, image_placeholder, video_placeholder, images, videos):
    image_idx = 0
    video_idx = 0
    remaining = content
    # Handle None images/videos
    images = images or []
    videos = videos or []
    has_image = image_placeholder in content
    has_video = video_placeholder in content
    new_content = []
    while remaining:
        img_pos = remaining.find(image_placeholder) if has_image else -1
        vid_pos = remaining.find(video_placeholder) if has_video else -1

        # Find next placeholder
        if img_pos == -1 and vid_pos == -1:
            if remaining.strip():
                new_content.append({'type': 'text', 'text': remaining})
            break

        # Determine which comes first
        if vid_pos == -1 or (img_pos != -1 and img_pos < vid_pos):
            # Image placeholder
            if remaining[:img_pos].strip():
                new_content.append({'type': 'text', 'text': remaining[:img_pos]})
            if image_idx < len(images):
                new_content.append({'type': 'image', 'url': images[image_idx]})
                image_idx += 1
            remaining = remaining[img_pos + len(image_placeholder):]
        else:
            # Video placeholder
            if remaining[:vid_pos].strip():
                new_content.append({'type': 'text', 'text': remaining[:vid_pos]})
            if video_idx < len(videos):
                new_content.append({'type': 'video', 'url': videos[video_idx]})
                video_idx += 1
            remaining = remaining[vid_pos + len(video_placeholder):]
    return new_content

def transfer_to_standard_message(message: Message, image_placeholder, video_placeholder, is_mm):
    if is_mm:
        new_content = _transfer_single_message(message['content'], image_placeholder, video_placeholder,
                                               message.get('images'), message.get('videos'))
    else:
        new_content = message['content']

    return Message(role=message['role'], content=new_content, tool_calls=message.get('tool_calls'),
                   reasoning_content=message.get('reasoning_content'))
