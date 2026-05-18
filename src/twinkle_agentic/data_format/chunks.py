import sys
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

if sys.version_info[:2] <= (3, 11):
    # Pydantic requirements.
    from typing_extensions import TypedDict
else:
    from typing import TypedDict

_MULTIMODAL_TYPES = ('image', 'video', 'audio')
_MEDIA_BUCKETS = (('images', 'image'), ('videos', 'video'), ('audios', 'audio'))


class Chunk(TypedDict, total=False):

    type: Literal['text', 'image', 'video', 'audio']
    content: Union[str, Any]
    raw: Union[str, Any]
    role: str
    round: int


@dataclass
class Chunks:

    chunks: List[Chunk]

    def to_trajectory(
            self,
            block_wrapper: Optional[Tuple[str, str]] = ('<block_{n}>', '</block_{n}>'),
    ) -> Dict[str, Any]:
        media: Dict[str, List[Any]] = {t: [] for t in _MULTIMODAL_TYPES}
        bound: List[Chunk] = []
        wrap_counter = 0
        for c in self.chunks:
            if c.get('type') in _MULTIMODAL_TYPES and not isinstance(c.get('raw'), dict):
                media[c['type']].append(c.get('content'))
                continue
            if (block_wrapper and c.get('type') == 'text' and c.get('role') != 'tool'):
                raw = c.get('raw')
                is_condensed = isinstance(raw, dict) and raw.get('condensed')
                content = c.get('content')
                if is_condensed and isinstance(content, str) and content:
                    wrap_counter += 1
                    prefix = block_wrapper[0].format(n=wrap_counter)
                    suffix = block_wrapper[1].format(n=wrap_counter)
                    c = {**c, 'content': f'{prefix}{content}{suffix}'}
            bound.append(c)

        # Merge consecutive same-role chunks into one message via groupby.
        messages = [
            self._group_to_message(role, list(grp))
            for role, grp in groupby(bound, key=lambda c: c.get('role') or 'user')
        ]

        trajectory: Dict[str, Any] = {'messages': messages}
        for plural, singular in _MEDIA_BUCKETS:
            if media[singular]:
                trajectory[plural] = media[singular]
        return trajectory

    @staticmethod
    def _group_to_message(role: str, group: List[Chunk]) -> Dict[str, Any]:
        """Fold a same-role run of chunks into one :class:`Message`.

        Preserves the intra-group order so mixed text / image / video / audio
        parts round-trip back into OpenAI-style structured ``content``.
        """
        reasoning: List[str] = []
        parts: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        tool_call_id: Optional[str] = None
        has_media = False

        for c in group:
            t, raw, content = c.get('type'), c.get('raw'), c.get('content')
            kind = raw.get('kind') if isinstance(raw, dict) else None
            # Any chunk in the group may carry the shared ``tool_call_id``.
            if isinstance(raw, dict) and raw.get('tool_call_id') and tool_call_id is None:
                tool_call_id = raw['tool_call_id']

            if t == 'text' and kind == 'reasoning_content' and content:
                reasoning.append(content)
            elif t == 'text' and kind == 'tool_call' and isinstance(raw.get('tool_call'), dict):
                tool_calls.append(dict(raw['tool_call']))
            elif t == 'text' and content:
                parts.append({'type': 'text', 'text': content})
            elif t in _MULTIMODAL_TYPES and isinstance(raw, dict):
                has_media = True
                # Drop condenser-only markers, keep the original part shape.
                parts.append({k: v for k, v in raw.items() if k != 'condensed'} or {'type': t, t: content})

        msg: Dict[str, Any] = {'role': role}
        if reasoning:
            msg['reasoning_content'] = '\n\n'.join(reasoning)
        if parts:
            msg['content'] = parts if has_media else '\n\n'.join(p['text'] for p in parts)
        if tool_calls:
            msg['tool_calls'] = tool_calls
        if tool_call_id is not None:
            msg['tool_call_id'] = tool_call_id
        return msg
