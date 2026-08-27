# Copyright (c) ModelScope Contributors. All rights reserved.
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ToolCallParser(ABC):
    """Single-format tool-call parser."""

    name: str = ''
    open_marker: Optional[str] = None
    close_marker: Optional[str] = None

    @abstractmethod
    def detect(self, text: str) -> bool:
        """Cheap pre-check: does ``text`` carry this format's markup?"""

    @abstractmethod
    def parse(self, text: str) -> List[Dict[str, Any]]:
        """Return OpenAI-shape tool_calls. ``arguments`` is a dict (jinja-friendly)."""

    @abstractmethod
    def clean(self, text: str) -> str:
        """Strip parser-specific markup; return plain content text."""

    def parse_errors(self, text: str) -> List[str]:
        """Why markup this parser recognised produced no call.

        ``detect`` saying yes while ``parse`` returns nothing means the model did
        try to call a tool and the markup did not survive parsing. Without this
        the caller cannot tell that apart from a reply that called nothing, so it
        ends the episode and the model is never told its call was dropped.
        Measured on one challenger run: 6 of 59 episodes ended that way, each
        with a well-formed ``<tool_call>`` block whose JSON carried a Python-style
        ``\\'`` escape or a raw newline.

        One string per block that failed, carrying the parser's own reason (for a
        JSON block, the ``json.JSONDecodeError`` text). Default empty: a parser
        whose ``parse`` is the same regex as its ``detect`` cannot fail this way.
        """
        return []

    def extract_tool_result(self, text: str) -> Optional[str]:
        """If ``text`` is a tool-result message of this protocol, return the
        body with the protocol-specific prefix stripped; otherwise return ``None``.

        Default returns ``None`` — only protocols carrying their own tool-result
        framing (e.g. Cline) need to override this.
        """
        return None


class ToolCallRegistry:
    """Global ordered registry of :class:`ToolCallParser` instances."""

    _parsers: List[ToolCallParser] = []

    @classmethod
    def register(cls, parser: ToolCallParser) -> ToolCallParser:
        for p in cls._parsers:
            if p.name == parser.name:
                return p
        cls._parsers.append(parser)
        return parser

    @classmethod
    def parsers(cls) -> List[ToolCallParser]:
        return list(cls._parsers)

    @classmethod
    def detect_first(cls, text: str) -> Optional[ToolCallParser]:
        if not text:
            return None
        for p in cls._parsers:
            if p.detect(text):
                return p
        return None
