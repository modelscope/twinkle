# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Any, Dict, List

from twinkle import remote_class
from twinkle.template import Template


@remote_class()
class QwenTemplate(Template):

    _BLOCK_RE = re.compile(r'<tool_call>\s*([\s\S]*?)\s*(?:</tool_call>|\Z)')
    _FUNCTION_RE = re.compile(r'<function=([^>]+)>([\s\S]*?)</function>')
    _PARAMETER_RE = re.compile(r'<parameter=([^>]+)>\s*([\s\S]*?)\s*</parameter>')
    _STRIP_RE = re.compile(r'<tool_call>[\s\S]*?(?:</tool_call>|\Z)')

    def parse(self, decoded: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for block_m in self._BLOCK_RE.finditer(decoded or ''):
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
            # JSON fallback: ``{"name": ..., "arguments": ...}`` inside the block.
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

    def clean(self, decoded: str) -> str:
        return self._STRIP_RE.sub('', decoded or '').rstrip()

    def parse_tool_call(self, decoded: str) -> List[Dict[str, Any]]:
        """Parse tool calls from the assistant's decoded output.

        Dispatches by model family on ``self.model_id``; the actual
        wire-format logic lives in :mod:`.tool_call_parser`.
        """
        mid = (self.model_id or '').lower()
        if 'qwen' in mid:
            return self.parse(decoded)
        # TODO: Other models (Llama3, OpenAI JSON, …) — add a parser in
        # ``tool_call_parser.py`` and extend this dispatch.
        return []

    def clean_tool_call(self, decoded: str) -> str:
        """Strip family-specific tool-call markup from assistant text."""
        mid = (self.model_id or '').lower()
        if 'qwen' in mid:
            return self.clean(decoded)
        # TODO: Other models
        return (decoded or '').rstrip()
