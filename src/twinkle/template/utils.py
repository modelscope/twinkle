# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib
import inspect
from copy import copy, deepcopy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, TypeVar

from twinkle.data_format import Message, Trajectory
from twinkle.utils import get_logger, to_device

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger()
_T = TypeVar('_T')


def construct_template(template_cls, *, _default_model_id=None, **kwargs):
    import twinkle.template
    from twinkle.template.base import Template
    from twinkle.utils import Plugin

    template_spec = kwargs.pop('template_spec', None)
    if isinstance(template_cls, Template):
        if template_spec is not None:
            raise TypeError('an existing template cannot also receive template_spec')
        return template_cls.replace(**kwargs) if kwargs else template_cls
    if isinstance(template_cls, str):
        if hasattr(twinkle.template, template_cls):
            template_cls = getattr(twinkle.template, template_cls)
        elif template_spec is not None:
            module, separator, name = template_cls.rpartition('.')
            if not separator:
                raise ValueError(f'Template {template_cls!r} is not registered in this worker')
            template_cls = getattr(importlib.import_module(module), name)
        elif template_cls.isidentifier():
            raise ValueError(f'Template {template_cls!r} is not registered in this worker')
        else:
            template_cls = Plugin.load_plugin(template_cls, Template)
    if not isinstance(template_cls, type) or not issubclass(template_cls, Template):
        raise TypeError('template_cls must be a Template instance, subclass or registered name')
    if template_spec is not None:
        if kwargs:
            raise TypeError(f'Unexpected TemplateSpec construction arguments: {sorted(kwargs)}')
        factory = getattr(template_cls, 'from_spec', None)
        if factory is None:
            raise TypeError(f'{template_cls!r} does not implement from_spec().')
        return factory(template_spec)
    if _default_model_id is not None:
        for base in template_cls.__mro__:
            parameters = inspect.signature(base.__init__).parameters
            if 'model_id' in parameters:
                kwargs.setdefault('model_id', _default_model_id)
                break
            if not any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
                break
    return template_cls(**kwargs)


def template_request(template_cls, kwargs):
    options = dict(kwargs)
    if isinstance(template_cls, str):
        return template_cls, options
    from twinkle.template.base import Template
    if isinstance(template_cls, Template):
        if options:
            template_cls = template_cls.replace(**options)
        options = template_cls._constructor_config()
        template_cls = type(template_cls)
    if isinstance(template_cls, type):
        if not issubclass(template_cls, Template):
            raise TypeError('template_cls must be a Template subclass')
        template_cls = getattr(template_cls, '_server_template_name', None) or template_cls.__name__
    if not isinstance(template_cls, str):
        raise TypeError('template_cls must be a Template instance, subclass or registered name')
    return template_cls, options


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
    from twinkle.utils.safe_media import checked_media_path, safe_http_get

    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img
    elif isinstance(img, str):
        if img.startswith(('http://', 'https://')):
            # safe_http_get refuses non-public targets (private nets, loopback, cloud
            # metadata) and re-validates every redirect hop: a URL from a dataset row or
            # a serving request must not turn this process into an SSRF pivot.
            resp = safe_http_get(img, timeout=30)
            return Image.open(io.BytesIO(resp.content))
        else:
            return Image.open(checked_media_path(img))
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

    return {**message, 'content': new_content}


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


@lru_cache(maxsize=128)
def _compile_assistant_template(chat_template):
    from jinja2 import nodes
    from jinja2.visitor import NodeTransformer
    from transformers.utils.chat_template_utils import _compile_jinja_template

    original = _compile_jinja_template(chat_template)
    environment = original.environment

    class TrackAssistant(NodeTransformer):
        def __init__(self):
            self.targets = []

        @staticmethod
        def boundary(target, kind):
            message = nodes.Name(target, 'load')
            selected = nodes.Call(nodes.Name('_assistant_is_selected', 'load'), [message], [], None, None)
            marker = nodes.Call(nodes.Name('_assistant_boundary', 'load'), [message, nodes.Const(kind)], [], None, None)
            return nodes.If(selected, [nodes.Output([marker])], [], [])

        def visit_For(self, node, *args, **kwargs):
            target = node.target.name if isinstance(node.target, nodes.Name) else None
            self.targets.append(target)
            node = self.generic_visit(node, *args, **kwargs)
            self.targets.pop()
            if target is not None:
                node.body = [self.boundary(target, 'S'), *node.body, self.boundary(target, 'E')]
            return node

        def visit_Continue(self, node, *args, **kwargs):
            if self.targets and self.targets[-1] is not None:
                return [self.boundary(self.targets[-1], 'E'), node]
            return node

        visit_Break = visit_Continue

    return original, environment.from_string(TrackAssistant().visit(environment.parse(chat_template)))


def jinja_assistant_ranges(tokenizer, messages, selected, expected_text, **kwargs):
    import re
    import uuid

    if not selected:
        return []
    template = tokenizer.get_chat_template(kwargs.pop('chat_template', None), tools=kwargs.get('tools'))
    original, compiled = _compile_assistant_template(template)
    for key in (
        'tokenize',
        'return_dict',
        'return_tensors',
        'return_assistant_tokens_mask',
        'padding',
        'truncation',
        'max_length',
        'continue_final_message',
        'tokenizer_kwargs',
    ):
        kwargs.pop(key, None)
    message_indices = {id(messages[index]): index for index in selected}
    selected_ids = set(message_indices)
    seen = set()
    prefix = f'<<ASSISTANT_{uuid.uuid4().hex}:'
    block_count = 0
    pending = {}
    owners = {}

    def is_selected(message):
        return id(message) in selected_ids

    def boundary(message, kind):
        nonlocal block_count
        message_id = id(message)
        if kind == 'S':
            index = block_count
            block_count += 1
            pending.setdefault(message_id, []).append(index)
            owners[index] = message_id
            seen.add(message_id)
        else:
            index = pending[message_id].pop()
        return f'{prefix}{kind}{index}>>'

    render_kwargs = {
        **tokenizer.special_tokens_map,
        'tools': None,
        'documents': None,
        'add_generation_prompt': False,
        **kwargs,
    }
    rendered = compiled.render(
        **{
            **render_kwargs,
            'messages': messages,
            '_assistant_is_selected': is_selected,
            '_assistant_boundary': boundary,
        }
    )
    parts, stack, spans = [], [], []
    productive = set()
    cursor, length = 0, 0
    for match in re.finditer(re.escape(prefix) + r'([SE])(\d+)>>', rendered):
        part = rendered[cursor : match.start()]
        parts.append(part)
        length += len(part)
        kind, index = match.group(1), int(match.group(2))
        if kind == 'S':
            stack.append((index, length))
        elif not stack or stack[-1][0] != index:
            raise ValueError('Jinja assistant ownership markers are not balanced')
        else:
            start = stack.pop()[1]
            spans.append((start, length, owners[index]))
            if length > start:
                productive.add(owners[index])
        cursor = match.end()
    parts.append(rendered[cursor:])
    if (
        stack
        or len(spans) != block_count
        or seen != selected_ids
        or productive != selected_ids
        or ''.join(parts) != expected_text
    ):
        raise ValueError('Jinja template cannot preserve assistant ownership without changing its rendering')
    openers = {}
    result = []
    for start, end, owner in sorted(spans):
        if start == end:
            continue
        if owner not in openers:
            options = {**render_kwargs, 'messages': messages[: message_indices[owner]], 'add_generation_prompt': False}
            before = original.render(**options)
            after = original.render(**{**options, 'add_generation_prompt': True})
            if not after.startswith(before):
                raise ValueError('Jinja generation opener cannot be isolated without changing its prefix rendering')
            openers[owner] = after[len(before) :], before, options
        opener, before, options = openers[owner]
        content = expected_text[start:end]
        if content.startswith(opener):
            cut = len(opener)
        else:
            chat_options = {**options, 'enable_thinking': False, 'thinking': False}
            chat_before = original.render(**chat_options)
            chat_after = original.render(**{**chat_options, 'add_generation_prompt': True})
            if chat_before != before or not chat_after.startswith(before):
                raise ValueError('Jinja history opener cannot be isolated without changing its prefix rendering')
            chat_opener = chat_after[len(before) :]
            if not content.startswith(chat_opener):
                raise ValueError('Jinja assistant opener does not match its generation prefix')
            cut = len(chat_opener)
        result.append((start + cut, end))
    return result


class TokenizeByRound:
    """Tokenize by encoding messages round-by-round.

    This approach handles <think></think> tags correctly by encoding each message
    incrementally, determining token boundaries by comparing consecutive encode results.

    For assistant messages, uses add_generation_prompt=True to exclude the assistant
    prefix (e.g., '<|im_start|>assistant\n') from training labels.

    Boundaries are prefix-encode LENGTHS, so the caller must hand in a normalized
    message list (multiple system messages already collapsed into one leading
    message, see Template._encode_messages): otherwise the full encode and the
    prefix slices disagree on where the head ends and labels silently shift.
    """

    @staticmethod
    def tokenize_with_assistant_labels(tokenizer: 'PreTrainedTokenizer',
                                       encode_func: Callable,
                                       trajectory: Trajectory,
                                       train_indices: Optional[set] = None,
                                       **kwargs) -> Tuple[List[int], List[int], Dict[str, Any]]:
        """Tokenize trajectory and generate labels for assistant turns.

        Args:
            tokenizer: Fast tokenizer for full-render offsets, or None for verified round encoders.
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

        spans = None
        if getattr(tokenizer, 'is_fast', False) and getattr(tokenizer, 'chat_template', None):
            options = {**kwargs, 'tools': list(trajectory.get('tools') or []),
                       'tokenize': False, 'return_dict': False, 'return_tensors': None,
                       'return_assistant_tokens_mask': False, 'add_generation_prompt': False}
            from jinja2 import TemplateError

            # Any template the marker pass cannot handle falls through to the round-by-round
            # encoder below, which is correct whenever the template is prefix-stable.
            try:
                text = tokenizer.apply_chat_template(messages, **options)
                tokenized = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
                if tokenized['input_ids'] == full_ids:
                    selected = {index for index, message in enumerate(messages)
                                if message['role'] == 'assistant' and (train_indices is None or index in train_indices)}
                    spans = jinja_assistant_ranges(tokenizer, messages, selected, text, **options)
            except (TemplateError, TypeError, ValueError):
                pass
        if spans is not None:
            for index, (start, end) in enumerate(tokenized['offset_mapping']):
                if end > start and any(left <= start and end <= right for left, right in spans):
                    labels[index] = full_ids[index]
            return full_ids, labels, encoded

        def render_length(prefix, add_generation_prompt=False):
            partial_trajectory = copy(trajectory)
            partial_trajectory['messages'] = list(prefix)
            partial_ids = _coerce_ids_to_list(encode_func(
                partial_trajectory, add_generation_prompt=add_generation_prompt, **kwargs)['input_ids'])
            if full_ids[:len(partial_ids)] != partial_ids:
                raise ValueError('Chat template is not prefix-stable and cannot provide safe assistant boundaries')
            return len(partial_ids)

        for i, msg in enumerate(messages):
            if msg['role'] != 'assistant' or train_indices is not None and i not in train_indices:
                continue
            # start: after the assistant prefix, which the add_generation_prompt encode includes
            start_pos = render_length(messages[:i], add_generation_prompt=True)
            end_pos = render_length(messages[:i + 1])
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
