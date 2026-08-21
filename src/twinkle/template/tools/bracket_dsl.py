# Copyright (c) ModelScope Contributors. All rights reserved.
import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from .base import ToolCallParser


class BracketDslParser(ToolCallParser):
    """Parser for the bracketed call list used by ToolACE-style prompts.

    The system prompt of these datasets asks the model to answer with a python
    call list instead of markup, e.g.::

        [Text Analysis(text="great service"), UserID(username="alex")]
        [quarterly_data(stock_symbols=["AAPL", "TSLA"])]

    Function names may contain spaces, dots and dashes ("Get All Strains",
    "database.insert_data"). Argument values may themselves contain brackets and
    parentheses (list arguments), so the call list is located by scanning with a
    depth counter rather than by a bracket-free regex. Argument values are read
    as python literals, falling back to the raw text when they are not literals.
    """

    name = 'bracket_dsl'
    open_marker = None
    close_marker = None

    # A call opens with a name directly followed by '('; used for cheap detection
    # and to find call starts inside a located block. Names may carry spaces,
    # dots, dashes and apostrophes ("Get Today's Prices").
    _CALL_START_RE = re.compile(r"([A-Za-z_][\w.\-' ]*?)\s*\(")
    _DETECT_RE = re.compile(r"\[\s*[A-Za-z_][\w.\-' ]*?\s*\(")
    # Split an argument body on top-level commas only (values may hold commas).
    _ARG_NAME_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*(.*)$', re.DOTALL)

    def detect(self, text: str) -> bool:
        return bool(self._DETECT_RE.search(text or ''))

    @staticmethod
    def _find_blocks(text: str) -> List[Tuple[int, int]]:
        """Locate ``[ ... ]`` spans that start a call list, honouring nesting.

        Only a '[' immediately followed by ``name(`` opens a block, so plain
        prose lists ("[1, 2, 3]") are ignored. Quotes are honoured only inside an
        argument body (paren depth > 0) so that an apostrophe in a function name
        ("Get Today's Prices") does not start a string.
        """
        spans: List[Tuple[int, int]] = []
        i, n = 0, len(text or '')
        while i < n:
            if text[i] != '[':
                i += 1
                continue
            if not BracketDslParser._DETECT_RE.match(text, i):
                i += 1
                continue
            depth, j, quote, paren = 0, i, None, 0
            while j < n:
                ch = text[j]
                if quote:
                    if ch == '\\':
                        j += 2
                        continue
                    if ch == quote:
                        quote = None
                elif ch in '"\'' and paren > 0:
                    quote = ch
                elif ch == '(':
                    paren += 1
                elif ch == ')':
                    paren -= 1
                elif ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        spans.append((i, j + 1))
                        break
                j += 1
            i = (spans[-1][1] if spans and spans[-1][0] == i else i + 1)
        return spans

    @staticmethod
    def _match_paren(text: str, open_idx: int) -> Optional[int]:
        """Index of the ')' matching the '(' at ``open_idx``."""
        depth, j, quote = 0, open_idx, None
        n = len(text)
        while j < n:
            ch = text[j]
            if quote:
                if ch == '\\':
                    j += 2
                    continue
                if ch == quote:
                    quote = None
            elif ch in '"\'' and depth > 0:
                quote = ch
            elif ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    return j
            j += 1
        return None

    @staticmethod
    def _split_top_level(body: str) -> List[str]:
        """Split on commas that are not inside quotes, brackets or parens."""
        parts, buf = [], []
        depth, quote = 0, None
        for ch in body or '':
            if quote:
                if ch == quote:
                    quote = None
                buf.append(ch)
                continue
            if ch in '"\'':
                quote = ch
            elif ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth -= 1
            elif ch == ',' and depth == 0:
                parts.append(''.join(buf))
                buf = []
                continue
            buf.append(ch)
        if buf:
            parts.append(''.join(buf))
        return parts

    def _parse_args(self, body: str) -> Dict[str, Any]:
        args: Dict[str, Any] = {}
        for chunk in self._split_top_level(body):
            m = self._ARG_NAME_RE.match(chunk)
            if not m:
                continue
            key = m.group(1)
            raw = m.group(2).strip()
            try:
                args[key] = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                args[key] = raw.strip('"\'')
        return args

    def parse(self, text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        text = text or ''
        for start, end in self._find_blocks(text):
            block = text[start:end]
            pos = 1  # skip the opening '['
            while pos < len(block):
                m = self._CALL_START_RE.search(block, pos)
                if not m:
                    break
                close = self._match_paren(block, m.end() - 1)
                if close is None:
                    break
                name = m.group(1).strip()
                if name:
                    calls.append({
                        'type': 'function',
                        'function': {
                            'name': name,
                            'arguments': self._parse_args(block[m.end():close]),
                        },
                    })
                pos = close + 1
        return calls

    def clean(self, text: str) -> str:
        text = text or ''
        out, last = [], 0
        for start, end in self._find_blocks(text):
            out.append(text[last:start])
            last = end
        out.append(text[last:])
        return ''.join(out).rstrip()
