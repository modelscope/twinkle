# Copyright (c) ModelScope Contributors. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, NewType, Optional, Tuple
from typing_extensions import Required, TypedDict

from twinkle.data_format import Trajectory

Message = Dict[str, Any]
Tool = Dict[str, Any]


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


InputOrderLabels = NewType('InputOrderLabels', List[int])
OutputOrderLabels = NewType('OutputOrderLabels', List[int])
InputOrderLossScale = NewType('InputOrderLossScale', List[float])
OutputOrderLossScale = NewType('OutputOrderLossScale', List[float])
InputOrderCompletionMask = NewType('InputOrderCompletionMask', List[int])
OutputOrderCompletionMask = NewType('OutputOrderCompletionMask', List[int])


class InputOrderRow(Trajectory, total=False):
    input_ids: Required[List[int]]
    labels: InputOrderLabels
    loss_scale: InputOrderLossScale
    completion_mask: InputOrderCompletionMask


class EncodedRow(Trajectory, total=False):
    input_ids: Required[List[int]]
    labels: OutputOrderLabels
    loss_scale: OutputOrderLossScale
    completion_mask: OutputOrderCompletionMask


class PreparedRow(EncodedRow, total=False):
    position_ids: Required[Any]
    attention_mask: Any


@dataclass(frozen=True)
class PreferencePair:
    chosen: EncodedRow
    rejected: Optional[EncodedRow]
    margin: Optional[float] = None


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
