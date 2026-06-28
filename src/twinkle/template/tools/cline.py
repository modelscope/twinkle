# Copyright (c) ModelScope Contributors. All rights reserved.
"""Cline / OpenClaw text-embedded XML tool-call format.

Wire format (Layer-B agent app protocol — lives in plain ``content``,
not in the OpenAI ``tool_calls`` field):

    <read_file><path>src/foo.py</path></read_file>
    <execute_command>
      <command>ls -la</command>
      <requires_approval>false</requires_approval>
    </execute_command>

Detection is **structural** (no hardcoded tool-name whitelist):

* outer tag is snake_case ``[a-z][a-z0-9_]*`` and not in :data:`_DENY`
* outer block contains at least one nested ``<key>VAL</key>`` child

Recognised blocks are extracted only on full-text :meth:`parse` (e.g. by
``AgentTraceFilter`` after trajectory assembly).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import ToolCallParser

# Common HTML-like / template tags that are NOT Cline tool calls. Outer
# tags falling here are skipped to prevent false positives.
_DENY = frozenset({
    # twinkle-internal / model-internal markers
    'think',
    'answer',
    'tool_call',
    'tool_response',
    'function',
    'parameter',
    'parameters',
    'tools',
    'tool',
    'system',
    'user',
    'assistant',
    'message',
    'messages',
    'content',
    'response',
    'output',
    'role',
    'reasoning_content',
    # html / markdown
    'p',
    'a',
    'b',
    'i',
    'em',
    'strong',
    'div',
    'span',
    'pre',
    'code',
    'br',
    'hr',
    'ul',
    'ol',
    'li',
    'h1',
    'h2',
    'h3',
    'h4',
    'h5',
    'h6',
    'table',
    'tr',
    'td',
    'th',
    'tbody',
    'thead',
    'img',
    'video',
    'audio',
})

# Outer tool-call block: matched-pair via backreference. Body is non-greedy.
_BLOCK_RE = re.compile(r'<(?P<tool>[a-z][a-z0-9_]*)>(?P<body>[\s\S]*?)</(?P=tool)>')
# Inner parameter: matched-pair via backreference.
_PARAM_RE = re.compile(r'<(?P<key>[a-z][a-z0-9_]*)>(?P<val>[\s\S]*?)</(?P=key)>')

# Cline tool-result: [tool_name for 'path/args'] Result:
_RESULT_RE = re.compile(
    r'^\[(?P<tool>[a-z][a-z0-9_]*)\s+for\s+\'[^\']*\'\]\s*Result:\s*',
    re.DOTALL,
)


class ClineParser(ToolCallParser):
    name = 'cline'
    # Outer tag varies per tool — no fixed marker.
    open_marker = None
    close_marker = None

    def detect(self, text: str) -> bool:
        if not text or '<' not in text:
            return False
        for m in _BLOCK_RE.finditer(text):
            if m.group('tool') in _DENY:
                continue
            if _PARAM_RE.search(m.group('body')):
                return True
        return False

    def parse(self, text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for m in _BLOCK_RE.finditer(text or ''):
            tool = m.group('tool')
            if tool in _DENY:
                continue
            args: dict[str, Any] = {}
            for pm in _PARAM_RE.finditer(m.group('body')):
                args[pm.group('key')] = pm.group('val').strip()
            if not args:
                continue
            calls.append({
                'type': 'function',
                'function': {
                    'name': tool,
                    'arguments': args
                },
            })
        return calls

    def clean(self, text: str) -> str:
        if not text:
            return text or ''
        spans: list[tuple] = []
        for m in _BLOCK_RE.finditer(text):
            if m.group('tool') in _DENY:
                continue
            if not _PARAM_RE.search(m.group('body')):
                continue
            spans.append((m.start(), m.end()))
        if not spans:
            return text.rstrip()
        out: list[str] = []
        last = 0
        for s, e in spans:
            out.append(text[last:s])
            last = e
        out.append(text[last:])
        return ''.join(out).rstrip()

    def extract_tool_result(self, text: str) -> str | None:
        m = _RESULT_RE.match(text or '')
        return text[m.end():] if m else None
