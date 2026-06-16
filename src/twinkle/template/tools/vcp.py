# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Dict, List

from .base import ToolCallParser

_VCP_OPEN = '<<<[TOOL_REQUEST]>>>'
_VCP_CLOSE = '<<<[END_TOOL_REQUEST]>>>'

_VCP_BLOCK_RE = re.compile(
    r'<<<\[TOOL_REQUEST\]>>>(.*?)<<<\[END_TOOL_REQUEST\]>>>',
    re.DOTALL,
)

# `「始ESCAPE」...「末ESCAPE」` is the nesting-safe variant; pair them strictly
# so an escaped value is not closed by a bare `「末」` from an inner block.
_VCP_KV_RE = re.compile(
    r'(?P<key>[A-Za-z_]\w*)\s*:\s*'
    r'(?:「始ESCAPE」(?P<val_esc>.*?)「末ESCAPE」'
    r'|「始」(?P<val>.*?)「末」)',
    re.DOTALL,
)


class VCPParser(ToolCallParser):
    """VCPChat / VCPSystem custom tool-call format.

    Outer markers ``<<<[TOOL_REQUEST]>>> ... <<<[END_TOOL_REQUEST]>>>`` wrap
    one call; parameters use full-width brackets ``「始」value「末」`` (escape
    variant ``「始ESCAPE」...「末ESCAPE」`` permits nested outer markers).
    The canonical function name lives in the ``tool_name`` field.
    """

    name = 'vcp'
    open_marker = _VCP_OPEN
    close_marker = _VCP_CLOSE

    def detect(self, text: str) -> bool:
        return _VCP_OPEN in (text or '')

    def parse(self, text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for block in _VCP_BLOCK_RE.findall(text or ''):
            args: Dict[str, Any] = {}
            name = ''
            for m in _VCP_KV_RE.finditer(block):
                k = m.group('key')
                v = m.group('val_esc') if m.group('val_esc') is not None else m.group('val')
                if k == 'tool_name':
                    name = (v or '').strip()
                else:
                    args[k] = v
            if not name:
                continue
            calls.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': args,
                },
            })
        return calls

    def clean(self, text: str) -> str:
        return _VCP_BLOCK_RE.sub('', text or '').rstrip()
