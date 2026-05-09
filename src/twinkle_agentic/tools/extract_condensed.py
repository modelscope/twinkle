# Copyright (c) ModelScope Contributors. All rights reserved.
import json
from typing import Any, Dict, List, Optional

from twinkle.data_format.message import Tool as ToolInfo
from twinkle_agentic.data_format import Chunks

from .base import Tool


TOOL_NAME = 'extract_condensed'


class ExtractCondensed(Tool):
    """Return the original text behind a ``<block_N>`` compressed segment.

    Args:
        chunks: The :class:`Chunks` object emitted by a condenser
            (post-compression). Each condensed chunk should carry
            ``raw.original`` holding the pre-compression text; if that
            snapshot is missing the block is still enumerated (so
            numbering stays aligned with ``<block_N>``) but the tool
            returns an explicit error on lookup rather than silently
            handing back the compressed stand-in.

    The block enumeration rule mirrors :meth:`Chunks.to_trajectory`
    exactly: only text chunks with ``raw.condensed=True``,
    ``role != 'tool'`` and non-empty content are indexed, in chunk
    order, starting from ``1``. This guarantees the block numbers this
    tool accepts match the ``<block_N>`` tags the model actually sees.
    """

    def __init__(self, chunks: Chunks):
        self._blocks: Dict[int, Optional[str]] = {}
        counter = 0
        for c in chunks.chunks:
            if c.get('type') != 'text':
                continue
            content = c.get('content')
            if not isinstance(content, str) or not content:
                continue
            if c.get('role') == 'tool':
                continue
            raw = c.get('raw')
            if not (isinstance(raw, dict) and raw.get('condensed')):
                continue
            counter += 1
            original = raw.get('original')
            self._blocks[counter] = (
                original if isinstance(original, str) and original else None)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------
    def tool_info(self) -> ToolInfo:
        return {
            'tool_name': TOOL_NAME,
            'description': (
                'Recover the full, uncompressed text of one or more '
                'previously condensed passages, identified by their '
                '<block_N> tags. Use this tool whenever you need to '
                're-read the original detail of compressed blocks.'),
            'parameters': json.dumps({
                'blocks': ('int OR list[int], the 1-indexed block number(s) '
                           'N appearing inside <block_N>...</block_N>. '
                           'Pass a single int to expand one block, or a '
                           'list of ints to expand several in one call '
                           '(e.g. 3 or [1, 3, 5]).'),
            }),
        }

    def __call__(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            return (f'Error: arguments must be an object, got '
                    f'{type(arguments).__name__}.')
        # Accept the new preferred name ``blocks`` first, fall back to the
        # legacy singular ``block`` for backward compatibility with callers
        # that were built against the int-only interface.
        if 'blocks' in arguments:
            raw = arguments['blocks']
            key = 'blocks'
        elif 'block' in arguments:
            raw = arguments['block']
            key = 'block'
        else:
            return 'Error: missing required argument "blocks".'

        # Normalise to a list of integers. Single int / str-int → 1-element
        # list; list/tuple → validate every element. Preserve order,
        # deduplicate while keeping first occurrence.
        if isinstance(raw, (list, tuple)):
            items = list(raw)
        else:
            items = [raw]

        seen: Dict[int, None] = {}
        parsed: List[int] = []
        for i, item in enumerate(items):
            # ``bool`` subclasses ``int`` (``int(True) == 1``) and ``float``
            # coerces silently (``int(1.9) == 1``); reject both up front.
            if isinstance(item, bool) or isinstance(item, float):
                return (f'Error: "{key}" item at position {i} must be an '
                        f'integer, got {type(item).__name__} {item!r}.')
            try:
                n = int(item)
            except (TypeError, ValueError):
                return (f'Error: "{key}" item at position {i} must be an '
                        f'integer, got {item!r}.')
            if n in seen:
                continue
            seen[n] = None
            parsed.append(n)

        if not parsed:
            return f'Error: "{key}" must contain at least one block number.'

        # Single-block path preserves the legacy bare-text return shape so
        # existing callers / prompts keep working unchanged.
        if len(parsed) == 1 and not isinstance(raw, (list, tuple)):
            return self._lookup_one(parsed[0])

        # Multi-block path wraps each result in <block_N>...</block_N> so
        # the model can tell them apart in the returned tool message.
        parts: List[str] = []
        for n in parsed:
            value = self._lookup_one(n)
            parts.append(f'<block_{n}>\n{value}\n</block_{n}>')
        return '\n\n'.join(parts)

    def _lookup_one(self, n: int) -> str:
        """Return the original text for block ``n`` or an ``Error: ...`` string."""
        if n not in self._blocks:
            available = ', '.join(str(k) for k in sorted(self._blocks))
            return (f'Error: block {n} not found. '
                    f'Available blocks: {available or "(none)"}.')
        value = self._blocks[n]
        if value is None:
            return (f'Error: block {n} has no original-text snapshot. '
                    f'The upstream condenser must populate raw.original '
                    f'before registering ExtractCondensed.')
        return value

    # ------------------------------------------------------------------
    # Introspection helpers (handy for debugging / tests)
    # ------------------------------------------------------------------
    @property
    def blocks(self) -> List[int]:
        """Sorted list of block indices available to this tool."""
        return sorted(self._blocks)

    def __len__(self) -> int:
        return len(self._blocks)

    def __contains__(self, n: Any) -> bool:
        try:
            return int(n) in self._blocks
        except (TypeError, ValueError):
            return False
