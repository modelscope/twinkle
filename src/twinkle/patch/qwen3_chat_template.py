# Copyright (c) ModelScope Contributors. All rights reserved.
"""Patch Qwen3.x official chat_template to fix two robustness bugs.

Upstream jinja block (see HF Qwen3 chat_template):

    {%- if '</think>' in content %}
        {%- set reasoning_content = content.split('</think>')[0]
                                       .rstrip('\\n')
                                       .split('<think>')[-1]
                                       .lstrip('\\n') %}
        {%- set content = content.split('</think>')[-1].lstrip('\\n') %}
    {%- endif %}

Two defects:

1. ``split('</think>')[-1]`` silently drops text between the first and last
   ``</think>`` when the content has multiple (e.g. one stray/hallucinated
   closing tag). This causes irrecoverable data loss during re-rendering.
2. ``split('<think>')[-1]`` returns the entire first chunk when ``<think>``
   is absent — treating model output that happens to contain a lone
   ``</think>`` as if it were a reasoning block.

Combined, these make the template asymmetric: content that was produced by
``enable_thinking=False`` (no opening ``<think>``) but with a hallucinated
orphan ``</think>`` gets mis-parsed into ``reasoning_content``, producing a
rendered string that diverges byte-wise from the actual token stream. This
breaks downstream consumers that rely on template round-trip (e.g. multi-turn
bridge text computation in the agentic rollout).

The patch narrows the parse branch to require a matching opening ``<think>``
at the start of content, uses ``split('</think>', 1)`` to preserve any
trailing orphans inside content, and extracts ``reasoning_content`` via
``split('<think>', 1)[1]`` (safe after startswith check).
"""
import warnings

from twinkle.patch import Patch

# Exact upstream block from Qwen3/Qwen3.5 chat_template. Indentation
# (12-space outer, 16-space inner) matches the shipped template.
_OLD = (
    "            {%- if '</think>' in content %}\n"
    "                {%- set reasoning_content = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}\n"  # noqa: E501
    "                {%- set content = content.split('</think>')[-1].lstrip('\\n') %}\n"
    '            {%- endif %}')

_NEW = ("            {%- if content.startswith('<think>') and '</think>' in content %}\n"
        "                {%- set _parts = content.split('</think>', 1) %}\n"
        "                {%- set reasoning_content = _parts[0].split('<think>', 1)[1].strip('\\n') %}\n"
        "                {%- set content = _parts[1].lstrip('\\n') %}\n"
        '            {%- endif %}')


class Qwen3ChatTemplate(Patch):
    """Patch tokenizer.chat_template in-place to fix Qwen3.x parse defects.

    Idempotent via pattern-presence check (no class-level flag needed: each
    tokenizer instance carries its own ``chat_template`` string, and a
    previously-patched string already contains ``_NEW``).

    Failure mode: if ``_OLD`` is not found (e.g. upstream fixed the template
    in a future release), emits a warning and leaves the tokenizer untouched
    so training keeps running.
    """

    def __call__(self, tokenizer, *args, **kwargs):
        tmpl = getattr(tokenizer, 'chat_template', None)
        if not tmpl or not isinstance(tmpl, str):
            return False
        if _NEW in tmpl:
            return False  # already patched in this process
        if _OLD not in tmpl:
            warnings.warn(
                'Qwen3ChatTemplate patch: expected OLD parse block not found '
                'in tokenizer.chat_template. Upstream template may have been '
                'updated or diverged; skipping patch. Verify manually if '
                'bridge text alignment issues reappear.',
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        tokenizer.chat_template = tmpl.replace(_OLD, _NEW, 1)
        return True
