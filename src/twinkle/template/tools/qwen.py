# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Any, Dict, List

from .base import ToolCallParser


class HermesQwenParser(ToolCallParser):
    name = 'hermes_qwen'
    open_marker = '<tool_call>'
    close_marker = '</tool_call>'

    _BLOCK_RE = re.compile(r'<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|\Z)')
    _FUNCTION_RE = re.compile(r'<function=([^>]+)>([\s\S]*?)</function>')
    _PARAMETER_RE = re.compile(r'<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>')
    _STRIP_RE = re.compile(r'<tool_call>[\s\S]*?(?:</tool_call>|\Z)')

    def detect(self, text: str) -> bool:
        return self.open_marker in text

    def parse(self, text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for block_m in self._BLOCK_RE.finditer(text or ''):
            block = block_m.group(1)
            func_m = self._FUNCTION_RE.search(block)
            if func_m:
                args: Dict[str, Any] = {}
                for pm in self._PARAMETER_RE.finditer(func_m.group(2)):
                    key = pm.group(1).strip()
                    val = pm.group(2).strip()
                    try:
                        args[key] = json.loads(val)
                    except (json.JSONDecodeError, ValueError):
                        args[key] = val
                calls.append({
                    'type': 'function',
                    'function': {
                        'name': func_m.group(1).strip(),
                        'arguments': args,
                    },
                })
                continue
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                continue
            name = data.get('name') or data.get('tool_name', '')
            if not name:
                continue
            args = data.get('arguments', {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {}
            calls.append({
                'type': 'function',
                'function': {
                    'name': name,
                    'arguments': args if isinstance(args, dict) else {},
                },
            })
        return calls

    def clean(self, text: str) -> str:
        return self._STRIP_RE.sub('', text or '').rstrip()
