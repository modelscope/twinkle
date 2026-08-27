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

    Fenced code blocks are excluded, and so is anything the model wrote inside
    ``<think>``: this format has no markup of its own, so a python expression is
    otherwise indistinguishable from a call list. Two further rules keep code out:
    a block only counts as a call list when every argument in it is a keyword
    argument (``name=value``), which no comprehension is, and a reply cut off
    mid-thought leaves ``<think>`` unterminated, so that region runs to the end of
    the text.

    Getting this wrong is expensive and quiet: ``[int(v) for v in raw]`` in a
    reply parses as a call to ``int`` with no arguments, the tool the model
    actually meant to call never runs, and the episode ends having done nothing.
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
    # A fence runs to its closing delimiter, or to the end of a truncated reply.
    _FENCE_RE = re.compile(r'```.*?(?:```|\Z)', re.DOTALL)
    # So does a thinking block: a reply truncated inside one never closes it.
    _THINK_RE = re.compile(r'<think>.*?(?:</think>|\Z)', re.DOTALL)

    @staticmethod
    def _fenced_spans(text: str) -> List[Tuple[int, int]]:
        return [m.span() for m in BracketDslParser._FENCE_RE.finditer(text or '')]

    @staticmethod
    def _skip_spans(text: str) -> List[Tuple[int, int]]:
        """Regions where a call list is quoted code or private thought, not a call."""
        text = text or ''
        return (BracketDslParser._fenced_spans(text)
                + [m.span() for m in BracketDslParser._THINK_RE.finditer(text)])

    @staticmethod
    def _in_spans(index: int, spans: List[Tuple[int, int]]) -> bool:
        return any(start <= index < end for start, end in spans)

    @classmethod
    def _is_keyword_body(cls, body: str) -> bool:
        """Is every argument in this body a ``name=value`` pair?

        An empty body qualifies -- ``[get_time()]`` is a call list. A positional
        argument does not: that is what a comprehension or a nested expression
        looks like.
        """
        chunks = [c for c in cls._split_top_level(body) if c.strip()]
        return all(cls._ARG_NAME_RE.match(c) for c in chunks)

    @classmethod
    def _looks_like_call_list(cls, block: str) -> bool:
        """Does ``[...]`` hold calls with keyword arguments, and nothing else?"""
        pos, seen = 1, 0
        while pos < len(block):
            m = cls._CALL_START_RE.search(block, pos)
            if not m:
                break
            close = cls._match_paren(block, m.end() - 1)
            if close is None:
                return False
            if not cls._is_keyword_body(block[m.end():close]):
                return False
            seen += 1
            pos = close + 1
        return seen > 0

    def detect(self, text: str) -> bool:
        # Via _find_blocks, so that detect and parse cannot disagree: a parser
        # that claims a reply and then finds nothing in it denies the remaining
        # parsers their turn.
        return bool(self._find_blocks(text or ''))

    @staticmethod
    def _find_blocks(text: str) -> List[Tuple[int, int]]:
        """Locate ``[ ... ]`` spans that start a call list, honouring nesting.

        Only a '[' immediately followed by ``name(`` opens a block, so plain
        prose lists ("[1, 2, 3]") are ignored. Quotes are honoured only inside an
        argument body (paren depth > 0) so that an apostrophe in a function name
        ("Get Today's Prices") does not start a string.
        """
        spans: List[Tuple[int, int]] = []
        skip = BracketDslParser._skip_spans(text)
        i, n = 0, len(text or '')
        while i < n:
            if text[i] != '[' or BracketDslParser._in_spans(i, skip):
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
                        if BracketDslParser._looks_like_call_list(text[i:j + 1]):
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
        return self._scan(text)[0]

    def parse_errors(self, text: str) -> List[str]:
        return self._scan(text)[1]

    def _scan(self, text: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Calls and failures from one pass, so the two cannot disagree."""
        calls: List[Dict[str, Any]] = []
        errors: List[str] = []
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
                    errors.append(f'{m.group(1).strip()}( is never closed by a '
                                  f'matching )')
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
                else:
                    errors.append('a call in the list has an empty function name')
                pos = close + 1
        return calls, errors

    def clean(self, text: str) -> str:
        text = text or ''
        out, last = [], 0
        for start, end in self._find_blocks(text):
            out.append(text[last:start])
            last = end
        out.append(text[last:])
        return ''.join(out).rstrip()
