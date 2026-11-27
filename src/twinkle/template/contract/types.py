# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple
from typing_extensions import Required, TypedDict

from twinkle.data_format import Message, Tool, Trajectory
EngineBackend = Literal['vllm', 'sglang', 'lmdeploy', 'transformers']


class MaxLengthError(ValueError):
    pass


@dataclass(frozen=True)
class EncodeIntent:
    add_generation_prompt: bool
    with_labels: bool
    expand_media: bool

    def __post_init__(self):
        if self.with_labels and self.add_generation_prompt:
            raise ValueError(
                'EncodeIntent(with_labels=True, add_generation_prompt=True) is contradictory: an open '
                'generation prompt has no response to supervise.')
        if not self.expand_media and self.with_labels:
            raise ValueError(
                'EncodeIntent(expand_media=False, with_labels=True) is contradictory: labels are aligned '
                'with the expanded sequence the model forward consumes.')

    def render_mode(self, engine: EngineBackend = 'vllm') -> str:
        check_backend(engine)
        return 'train' if self.with_labels else 'transformers' if self.expand_media else engine


TRAIN = EncodeIntent(False, True, True)
INFER = EncodeIntent(True, False, True)
ENGINE = EncodeIntent(True, False, False)
SCORE = EncodeIntent(False, False, True)


def check_backend(backend):
    if backend not in ('vllm', 'sglang', 'lmdeploy', 'transformers'):
        raise ValueError(f'Unsupported engine backend: {backend!r}')


def resolve_intent(intent, add_generation_prompt):
    if add_generation_prompt is not None and not isinstance(add_generation_prompt, bool):
        raise TypeError('add_generation_prompt must be a bool or None')
    if intent is None:
        return INFER if add_generation_prompt else TRAIN
    if not isinstance(intent, EncodeIntent):
        raise TypeError('intent must be an EncodeIntent')
    if add_generation_prompt is not None and add_generation_prompt != intent.add_generation_prompt:
        raise ValueError(f'add_generation_prompt={add_generation_prompt} contradicts {intent}.')
    return intent


class ConversationSample(Trajectory, total=False):
    messages: Required[List[Message]]
    system: Optional[str]
    label: bool
    objects: Dict[str, Any]
    chat_template_kwargs: Dict[str, Any]
    extra_kwargs: Dict[str, Any]
    mm_processor_kwargs: Dict[str, Any]


@dataclass(frozen=True)
class PreferenceSample:
    chosen: ConversationSample
    rejected: ConversationSample
    margin: Optional[float] = None


@dataclass(frozen=True)
class KTOSample:
    conversation: ConversationSample
    label: bool
    #: Optional KTO KL counterpart; without it, encode only the conversation.
    rejected: Optional[ConversationSample] = None


#: One row before (``InputOrderRow``) and after (``EncodedRow``) the labels shift; both are trajectories.
InputOrderRow = EncodedRow = Trajectory


@dataclass(frozen=True)
class MediaBundle:
    images: Tuple[Any, ...] = ()
    videos: Tuple[Any, ...] = ()
    audios: Tuple[Any, ...] = ()

    @classmethod
    def from_sources(cls, *sources):
        values = {'images': (), 'videos': (), 'audios': ()}
        for source in sources:
            if source is None:
                continue
            for key in values:
                if values[key]:
                    continue
                value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
                present = value is not None and (not isinstance(value, (list, tuple)) or bool(value))
                if not present and isinstance(source, dict):
                    value = cls._from_messages(source.get('messages') or (), key)
                    present = bool(value)
                if present:
                    values[key] = tuple(value if isinstance(value, (list, tuple)) else [value])
        return cls(**values)

    @staticmethod
    def _from_messages(messages, key):
        media_type = key[:-1]
        result = []
        for message in messages:
            message_media = message.get(key)
            if message_media is not None:
                result.extend(message_media if isinstance(message_media, (list, tuple)) else [message_media])
            content = message.get('content')
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict) or item.get('type') not in {media_type, f'{media_type}_url'}:
                    continue
                value = item.get(item['type'])
                if value is None:
                    value = item.get(media_type)
                if value is None:
                    value = item.get('url') or item.get('path')
                if isinstance(value, dict) and value.get('url') is not None:
                    value = value['url']
                if value is not None:
                    result.append(value)
        return result

    def __bool__(self):
        return bool(self.images or self.videos or self.audios)


@dataclass(frozen=True)
class DecodeContext:
    is_finished: bool = True
    first_token: bool = False
    template_inputs: Any = None
    skip_special_tokens: bool = True
    parse_tool_calls: bool = True
    reasoning_started: Optional[bool] = None
    tools: Optional[Tuple[Dict[str, Any], ...]] = None


@dataclass(frozen=True)
class DecodedResponse:
    content: Any
    tool_calls: Tuple[Dict[str, Any], ...] = ()
    errors: Tuple[str, ...] = ()
    reasoning_content: Optional[str] = None
    reasoning_started: Optional[bool] = None
