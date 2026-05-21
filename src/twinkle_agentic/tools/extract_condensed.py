# Copyright (c) ModelScope Contributors. All rights reserved.
from typing import Any, Dict, List, Optional

from twinkle.data_format.message import Tool as ToolInfo
from twinkle_agentic.data_format import Chunks
from .base import Tool

TOOL_NAME = 'extract_condensed'


class ExtractCondensed(Tool):
    """Return the original text behind a ``<block_N>`` compressed segment.

    TODO: Experimental feature, wait for testing
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
    ``role != 'tool'`` and non-empty content are indexed via a
    1-based monotonic counter in chunk order. The block ids this
    tool accepts therefore match the ``<block_N>`` tags the model
    actually sees.
    """

    def __init__(self, chunks: Chunks):
        self._blocks: Dict[int, Optional[str]] = {}
        # Trajectory-bound set of block ids already returned in full.
        self._already_expanded: set = set()
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
            self._blocks[counter] = (original if isinstance(original, str) and original else None)

    # ------------------------------------------------------------------
    # Tool interface
    # ------------------------------------------------------------------
    def tool_info(self) -> ToolInfo:
        return {
            'type': 'function',
            'function': {
                'name':
                TOOL_NAME,
                'description': ('Recover the full, uncompressed text of ONE previously '
                                'condensed passage, identified by its <block_N> tag. Use '
                                'this tool whenever you need to re-read the original '
                                'detail of a compressed block. Each call expands exactly '
                                'one block; issue separate calls for additional blocks, '
                                'and do not request the same block twice.'),
                'parameters': {
                    'blocks': ('int, the 1-indexed block number N appearing '
                               'inside <block_N>...</block_N>. Exactly one '
                               'block per call (e.g. 3); lists are rejected.'),
                },
            },
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

        # Single-block-per-call contract. Reject list/tuple up front so a
        # hallucinated ``blocks=[1..200]`` cannot balloon the tool response.
        if isinstance(raw, (list, tuple)):
            return (f'Error: "{key}" must be a single integer; only one '
                    f'block may be expanded per call. Issue a separate '
                    f'extract_condensed call for each block you need.')

        # ``bool`` subclasses ``int`` (``int(True) == 1``) and ``float``
        # coerces silently (``int(1.9) == 1``); reject both up front.
        if isinstance(raw, bool) or isinstance(raw, float):
            return (f'Error: "{key}" must be an integer, got '
                    f'{type(raw).__name__} {raw!r}.')
        try:
            n = int(raw)
        except (TypeError, ValueError):
            return f'Error: "{key}" must be an integer, got {raw!r}.'

        # Short existence check. Deliberately do NOT list every available
        # id -- when the policy hallucinates a large range, echoing the
        # full list back multiplies the error into thousands of tokens.
        if n not in self._blocks:
            if not self._blocks:
                return f'Error: block {n} not found; no blocks available.'
            ids = sorted(self._blocks)
            return (f'Error: block {n} not found; valid block ids are '
                    f'{ids}.')

        # Trajectory-bound idempotency. The raw text is already in the
        # conversation as a prior tool response -- returning it again would
        # just double the non-trainable footprint.
        if n in self._already_expanded:
            return (f'Block {n} was already expanded earlier in this '
                    f'trajectory; re-read the previous tool response '
                    f'instead of requesting it again.')

        value = self._blocks[n]
        if value is None:
            return (f'Error: block {n} has no original-text snapshot. '
                    f'The upstream condenser must populate raw.original '
                    f'before registering ExtractCondensed.')

        self._already_expanded.add(n)
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
