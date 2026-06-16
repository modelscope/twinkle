# Copyright (c) ModelScope Contributors. All rights reserved.
import re
from typing import Any, Dict, List

from .base import ToolCallParser

_ACTION_RE = re.compile(
    r'^\s*Action\s*:\s*(?P<name>[\w\-./]+)\s*\[(?P<args>.*?)\]\s*$',
    re.MULTILINE,
)


class ReActParser(ToolCallParser):
    name = 'react'

    def detect(self, text: str) -> bool:
        return bool(_ACTION_RE.search(text or ''))

    def parse(self, text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for m in _ACTION_RE.finditer(text or ''):
            calls.append({
                'type': 'function',
                'function': {
                    'name': m.group('name'),
                    'arguments': {
                        'input': m.group('args')
                    },
                },
            })
        return calls

    def clean(self, text: str) -> str:
        return _ACTION_RE.sub('', text or '').rstrip()
