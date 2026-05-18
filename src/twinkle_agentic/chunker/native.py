# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from twinkle.data_format import Trajectory
from twinkle_agentic.data_format import Chunk, Chunks
from .base import Chunker

# Recursive separator list, coarsest → finest. The empty string at the
# end forces a hard character cut when nothing finer fits.
_DEFAULT_SEPARATORS: tuple = (
    '\n\n',
    '\n',
    '。',
    '．',
    '.',
    '！',
    '!',
    '？',
    '?',
    '；',
    ';',
    '，',
    ',',
    ' ',
    '',
)

_MULTIMODAL_TYPES = ('image', 'video', 'audio')

_SplitFn = Optional[Callable[[str], List[str]]]


class NativeChunker(Chunker):
    """Character-level recursive chunker for trajectories.
    TODO: Experimental feature, wait for testing
    Args:
        chunk_size: Soft upper bound (in characters) for every emitted
            text chunk. Must be positive.
        separators: Ordered separator list. The chunker tries each
            separator in turn; any piece still larger than
            ``chunk_size`` is re-split with the next one. A terminal
            ``''`` (hard character cut) is appended automatically if
            missing so the algorithm is guaranteed to terminate.
        passage_boundary_re: Optional regex (compiled with
            ``re.MULTILINE``) whose matches act as **hard, non-mergeable**
            passage boundaries on the first user message. The regex
            match is preserved at the start of the next piece (so
            ``''.join(pieces) == text``). Pieces that are already
            ``<= chunk_size`` are emitted as-is and are **never merged**
            across boundaries; only pieces that still exceed
            ``chunk_size`` fall back to the normal recursive split + merge.
            This is how you keep e.g. HotpotQA passages atomic per
            ``<block_N>``.
    """

    def __init__(
        self,
        chunk_size: int = 1024,
        separators: Sequence[str] | None = None,
        passage_boundary_re: str | None = None,
    ):
        if chunk_size <= 0:
            raise ValueError(f'chunk_size must be positive, got {chunk_size}')
        self.chunk_size = chunk_size
        seps = tuple(separators) if separators is not None else _DEFAULT_SEPARATORS
        if '' not in seps:
            seps += ('', )
        self.separators = seps
        self.passage_boundary_re: re.Pattern | None = (
            re.compile(passage_boundary_re, re.MULTILINE) if passage_boundary_re else None)

    # ------------------------------------------------------------------
    # public entry
    # ------------------------------------------------------------------
    def __call__(self, trajectory: Trajectory) -> Chunks:
        chunks: list[Chunk] = []
        first_user_done = False
        # ``round`` is 1-indexed at the first user message. Any messages
        # emitted before that (e.g., leading ``system``) carry round 0.
        round_idx = 0
        for msg in trajectory.get('messages') or []:
            is_user = msg.get('role') == 'user'
            if is_user:
                round_idx += 1
            split = (self._split_text if is_user and not first_user_done else None)
            if is_user:
                first_user_done = True
            for chunk in self._parts(msg, split):
                chunk['round'] = round_idx
                chunks.append(chunk)
        return Chunks(chunks=chunks)

    # ------------------------------------------------------------------
    # message → chunks decomposition
    # ------------------------------------------------------------------
    def _parts(self, message: dict[str, Any], split: _SplitFn) -> Iterator[Chunk]:
        role = message.get('role') or 'user'
        tcid = message.get('tool_call_id')

        rc = message.get('reasoning_content')
        if rc:
            yield _text_chunk(role, rc, kind='reasoning_content', tool_call_id=tcid)

        content = message.get('content')
        if isinstance(content, str):
            yield from self._emit_text(role, content, split, tcid)
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get('type')
                if ptype == 'text':
                    yield from self._emit_text(role, part.get('text') or '', split, tcid)
                elif ptype in _MULTIMODAL_TYPES:
                    # Keep raw part so Chunks.to_trajectory can rebuild
                    # the original OpenAI-style entry verbatim.
                    yield {  # type: ignore[misc]
                        'type': ptype, 'content': part.get(ptype),
                        'raw': dict(part), 'role': role,
                    }

        for tc in message.get('tool_calls') or []:
            yield _text_chunk(role, '', kind='tool_call', tool_call=tc, tool_call_id=tcid)

    def _emit_text(self, role: str, text: str, split: _SplitFn, tool_call_id: str | None) -> Iterator[Chunk]:
        if not text:
            return
        pieces = split(text) if split is not None else [text]
        for piece in pieces:
            if piece:
                yield _text_chunk(role, piece, tool_call_id=tool_call_id)

    # ------------------------------------------------------------------
    # recursive text splitter
    # ------------------------------------------------------------------
    def _split_text(self, text: str) -> list[str]:
        if not text:
            return []
        if self.passage_boundary_re is None:
            if len(text) <= self.chunk_size:
                return [text]
            return self._merge(self._recursive_split(text, list(self.separators)))
        # Force-split first; each forced piece is kept intact when it is
        # already short enough, and is recursively re-split (but NOT
        # merged with sibling passages) when it exceeds ``chunk_size``.
        out: list[str] = []
        for piece in self._force_split(text):
            if not piece or not piece.strip():
                continue
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._merge(self._recursive_split(piece, list(self.separators))))
        return out

    def _force_split(self, text: str) -> list[str]:
        """Split ``text`` at every ``passage_boundary_re`` match; the
        match itself sticks to the start of the **next** piece, so
        ``''.join(_force_split(text)) == text``.
        """
        assert self.passage_boundary_re is not None
        matches = list(self.passage_boundary_re.finditer(text))
        if not matches:
            return [text]
        out: list[str] = []
        prev = 0
        for m in matches:
            start = m.start()
            if start > prev:
                out.append(text[prev:start])
            prev = start
        if prev < len(text):
            out.append(text[prev:])
        return out

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text else []
        # Terminal: no more separators, or next one is the hard-cut sentinel.
        if not separators or separators[0] == '':
            return _hard_cut(text, self.chunk_size)

        sep, *rest = separators
        out: list[str] = []
        for piece in _split_keep(text, sep):
            if not piece:
                continue
            if len(piece) <= self.chunk_size:
                out.append(piece)
            else:
                out.extend(self._recursive_split(piece, rest))
        return out

    def _merge(self, pieces: list[str]) -> list[str]:
        """Greedy concatenation: small fragments fuse up to ``chunk_size``
        without exceeding it. Relative order is preserved.
        """
        merged: list[str] = []
        buf = ''
        for p in pieces:
            if not p:
                continue
            if buf and len(buf) + len(p) > self.chunk_size:
                merged.append(buf)
                buf = ''
            buf += p
        if buf:
            merged.append(buf)
        return merged


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _split_keep(text: str, sep: str) -> list[str]:
    """``str.split(sep)`` but the separator stays glued to the end of
    each left-hand piece, so ``''.join(result) == text``.
    """
    if not sep or sep not in text:
        return [text] if text else []
    out: list[str] = []
    start, n = 0, len(sep)
    while (i := text.find(sep, start)) != -1:
        out.append(text[start:i + n])
        start = i + n
    if start < len(text):
        out.append(text[start:])
    return out


def _hard_cut(text: str, size: int) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] if text else []


def _text_chunk(
    role: str,
    content: str,
    *,
    kind: str | None = None,
    tool_call: Any = None,
    tool_call_id: str | None = None,
) -> Chunk:
    raw: dict[str, Any] = {}
    if kind is not None:
        raw['kind'] = kind
    if tool_call is not None:
        raw['tool_call'] = tool_call
    if tool_call_id is not None:
        raw['tool_call_id'] = tool_call_id
    chunk: Chunk = {'type': 'text', 'content': content, 'role': role}  # type: ignore[assignment]
    if raw:
        chunk['raw'] = raw
    return chunk
